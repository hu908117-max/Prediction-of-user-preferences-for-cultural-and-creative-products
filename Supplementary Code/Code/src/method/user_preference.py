"""Stage 2: cross-modal feature extraction guided by aesthetic-emotional preference.

Implements Eqs. (9)-(18) of the manuscript:
  (9)      concatenation of the emotion-label embedding and the interpretation
           text vector for one piece of feedback,
  (10)     MLP mapping of heterogeneous feedback to a unified preference space
           (local preference query),
  (11)-(12) Key/Value projections of the product token-level joint representation,
  (13)     query-space projection of the preference query,
  (14)     scaled dot-product attention weights,
  (15)     token-level Key/Value matrices,
  (16)     preference-guided cross-attention (attention-weighted product features),
  (17)     learnable importance of each historical product,
  (18)     weighted pooling into the user preference representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserPreferenceModule(nn.Module):
    """Builds the user preference representation from historical feedback."""

    def __init__(self, num_emotions, text_dim, joint_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # Learnable emotion-label embedding (one-hot / learnable embedding, Eq. 9).
        self.emotion_embedding = nn.Embedding(num_emotions + 1, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Eq. (10): MLP over [emotion embedding ; interpretation text vector].
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Eqs. (11)-(13): Key/Value projections of product tokens, query projection.
        self.wk = nn.Linear(joint_dim, hidden_dim)
        self.wv = nn.Linear(joint_dim, hidden_dim)
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.attn_dropout = nn.Dropout(dropout)

        # Eq. (17): learnable preference aggregation parameters.
        self.agg_param = nn.Parameter(torch.randn(hidden_dim))

    def encode_feedback(self, emotion_labels, feedback_text_vec):
        """Eqs. (9)-(10): (B, H) labels + (B, H, Dtxt) text vectors -> (B, H, D) queries."""
        e = self.emotion_embedding(emotion_labels)          # (B, H, D)
        t = self.text_proj(feedback_text_vec)               # (B, H, D)
        fused = torch.cat([e, t], dim=-1)                   # Eq. (9)
        return self.mlp(fused)                              # Eq. (10)

    def attend_product(self, query, product_tokens):
        """Eqs. (11)-(16): preference query attends to a product's token features.

        query:         (B, D) single preference query, or (B, H, D) per-history queries
        product_tokens:(..., Nv, Dj) token-level joint representation
        returns preference-guided product feature (..., D)
        """
        k = self.wk(product_tokens)                         # (..., Nv, D)  Eq. (11)/(15)
        v = self.wv(product_tokens)                         # (..., Nv, D)  Eq. (12)/(15)
        q = self.wq(query).unsqueeze(-2)                    # (..., 1, D)   Eq. (13)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)                         # Eq. (14)
        attn = self.attn_dropout(attn)
        ctx = torch.matmul(attn, v).squeeze(-2)             # Eq. (16)
        return ctx, attn

    def aggregate_history(self, ctx_list):
        """Eqs. (17)-(18): learnable weighted pooling over historical products.

        ctx_list: (B, H, D) context features of the historical products.
        returns user preference representation (B, D).
        """
        scores = torch.einsum("bhd,d->bh", ctx_list, self.agg_param)   # Eq. (17)
        beta = scores.softmax(dim=-1)                                  # Eq. (17)
        user_repr = torch.einsum("bh,bhd->bd", beta, ctx_list)         # Eq. (18)
        return user_repr, beta

    def forward(self, emotion_labels, feedback_text_vec, history_tokens):
        """Full stage-2 forward.

        emotion_labels:    (B, H)     historical emotion-label ids
        feedback_text_vec: (B, H, Dt) historical interpretation text vectors
        history_tokens:    (B, H, Nv, Dj) token-level joint representations
        returns (user_repr, beta, attn_stack)
        """
        queries = self.encode_feedback(emotion_labels, feedback_text_vec)   # (B, H, D)
        b, h, _, _ = history_tokens.shape
        ctx_stack = []
        attn_stack = []
        for j in range(h):
            ctx, attn = self.attend_product(queries[:, j], history_tokens[:, j])
            ctx_stack.append(ctx)
            attn_stack.append(attn)
        ctx_stack = torch.stack(ctx_stack, dim=1)                            # (B, H, D)
        user_repr, beta = self.aggregate_history(ctx_stack)
        return user_repr, beta, attn_stack


def gelu(x):
    return F.gelu(x)
