"""Stage 3-4: preference matching, contrastive learning, and joint loss.

Implements Eqs. (19)-(29) of the manuscript.

Stage 3 - Bilinear interaction and InfoNCE (Eqs. 19-27):
  (19) linear projections of the user preference representation and the product
       multimodal joint representation into a shared matching space,
  (20) low-rank bilinear interaction capturing second-order feature cross-links,
  (21) MLP non-linear transformation over [user; product; bilinear],
  (22) matching score,
  (23) sigmoid preference probability,
  (24) L2 normalisation of user / product representations,
  (25) temperature-scaled cosine similarity,
  (26) user-conditional InfoNCE loss,
  (27) symmetric cross-modal (visual-textual) contrastive loss.

Stage 4 - Joint loss (Eqs. 28-29):
  (28) binary cross-entropy preference prediction loss,
  (29) L = L_pred + lambda_1 * L_InfoNCE + lambda_2 * L_cross.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreferenceMatching(nn.Module):
    """Bilinear interaction head mapping a user-product pair to a preference score."""

    def __init__(self, dim, bilinear_dim=256, mlp_hidden=512, dropout=0.1):
        super().__init__()
        # Eq. (19): independent projections into the matching space.
        self.user_proj = nn.Linear(dim, dim)
        self.product_proj = nn.Linear(dim, dim)

        # Eq. (20): low-rank bilinear interaction. U and V project the two sides
        # into a common interaction space; their element-wise product encodes the
        # second-order cross relationship (a compact W_b formulation).
        self.U = nn.Linear(dim, bilinear_dim, bias=False)
        self.V = nn.Linear(dim, bilinear_dim, bias=False)

        # Eqs. (21)-(22): MLP over [user; product; bilinear] -> matching score.
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + bilinear_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_fc = nn.Linear(mlp_hidden // 2, 1)

    def project(self, user_repr, product_repr):
        """Eq. (19): project both sides into the matching space."""
        return self.user_proj(user_repr), self.product_proj(product_repr)

    def bilinear(self, p_user, p_product):
        """Eq. (20): second-order bilinear interaction."""
        return self.U(p_user) * self.V(p_product)

    def forward(self, user_repr, product_repr):
        """Return predicted probability, raw matching score, and bilinear vector."""
        p_user, p_product = self.project(user_repr, product_repr)
        z = self.bilinear(p_user, p_product)                       # Eq. (20)
        x = torch.cat([user_repr, product_repr, z], dim=-1)        # Eq. (21) concat
        h = self.mlp(x)                                            # Eq. (21)
        score = self.score_fc(h).squeeze(-1)                       # Eq. (22)
        prob = torch.sigmoid(score)                                # Eq. (23)
        return prob, score, z


def l2_normalize(x, eps=1e-8):
    """Eq. (24): L2 normalisation along the last dimension."""
    return F.normalize(x, p=2, dim=-1, eps=eps)


def temperature_similarity(a, b, tau):
    """Eqs. (24)-(25): cosine similarity scaled by the temperature coefficient."""
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return torch.matmul(a_n, b_n.t()) / tau


def user_product_infonce(user_repr, product_repr, pos_idx, tau=0.07):
    """Eq. (26): user-conditional InfoNCE over the training batch.

    user_repr:    (B, D) individual user preference representations
    product_repr: (B, D) multimodal joint representations of candidate products
    pos_idx:      (B,) index of the positive (high-preference) product for each user
    """
    sim = temperature_similarity(user_repr, product_repr, tau)     # (B, B)
    # Positive score for each user: similarity to its own positive product.
    batch_idx = torch.arange(sim.size(0), device=sim.device)
    pos_sim = sim[batch_idx, pos_idx]                              # (B,)
    # Denominator: similarity to all products in the batch (incl. negatives).
    denom = torch.logsumexp(sim, dim=-1)                           # (B,)
    return (-pos_sim + denom).mean()


def user_product_infonce_pooled(user_repr, pos_repr, neg_pool, tau=0.07):
    """Training-time variant of Eq. (26) with a pooled negative set.

    Each user has an explicit positive product (high preference). The negative pool
    is assembled from the low-preference products of the whole batch. This avoids
    requiring a positive to be present inside the in-batch candidate set and keeps a
    strong, well-defined contrastive signal.

    user_repr: (B, D)    individual user preference representations
    pos_repr:   (B, D)    high-preference product representations (one per user)
    neg_pool:  (B, M, D)  low-preference product pool (M negatives per user)
    """
    u_n = l2_normalize(user_repr)                 # (B, D)
    p_n = l2_normalize(pos_repr)                  # (B, D)
    n_n = l2_normalize(neg_pool)                  # (B, M, D)
    s_pos = torch.sum(u_n * p_n, dim=-1) / tau             # (B,)
    s_neg = torch.einsum("bd,bmd->bm", u_n, n_n) / tau     # (B, M)
    num = torch.exp(s_pos)
    den = num + torch.exp(s_neg).sum(dim=-1)
    return (-torch.log(num / den)).mean()


def cross_modal_contrast(vis_repr, txt_repr, tau=0.07):
    """Eq. (27): symmetric InfoNCE between visual and textual product reps."""
    s_vt = temperature_similarity(vis_repr, txt_repr, tau)         # (B, B)
    s_tv = s_vt.t()
    labels = torch.arange(s_vt.size(0), device=s_vt.device)
    loss_v = F.cross_entropy(s_vt, labels)
    loss_t = F.cross_entropy(s_tv, labels)
    return 0.5 * (loss_v + loss_t)


def preference_bce(prob, label):
    """Eq. (28): binary cross-entropy preference prediction loss."""
    return F.binary_cross_entropy(prob, label)


def joint_loss(prob, label, user_repr, product_repr, pos_idx,
               vis_repr, txt_repr, tau=0.07, lambda1=0.5, lambda2=0.3):
    """Eq. (29): weighted combination of the three objectives."""
    l_pred = preference_bce(prob, label)
    l_infonce = user_product_infonce(user_repr, product_repr, pos_idx, tau)
    l_cross = cross_modal_contrast(vis_repr, txt_repr, tau)
    total = l_pred + lambda1 * l_infonce + lambda2 * l_cross
    return total, l_pred, l_infonce, l_cross
