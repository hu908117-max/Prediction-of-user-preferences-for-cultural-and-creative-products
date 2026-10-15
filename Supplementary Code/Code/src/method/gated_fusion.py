"""Stage 1: multimodal feature extraction and gated fusion representation.

Implements Eqs. (1)-(8) of the manuscript:
  (1)-(2) regional visual features and text token features from the backbone,
  (3)     independent linear projections into a shared latent space,
  (4)-(5) cross-modal attention between visual regions and cultural semantics,
  (6)     modality-level global pooling,
  (7)     dynamic gating vector from the concatenated modal representations,
  (8)     element-wise gated joint representation of the product.
"""

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Cross-attention from queries to keys/values (Eqs. 4-5)."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # stored for analysis (association weights)

    def forward(self, query, key, value):
        b, nq, d = query.shape
        nk = key.shape[1]
        h = self.num_heads

        q = self.wq(query).view(b, nq, h, self.head_dim).transpose(1, 2)
        k = self.wk(key).view(b, nk, h, self.head_dim).transpose(1, 2)
        v = self.wv(value).view(b, nk, h, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, h, nq, nk)
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach().mean(dim=1)  # average over heads

        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).reshape(b, nq, d)
        return self.proj(out)


class GatedFusion(nn.Module):
    """Gated fusion of visual and cultural-textual token features (Eqs. 3-8)."""

    def __init__(self, vis_dim, txt_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        # Eq. (3): independent linear projections into the shared latent space.
        self.vis_proj = nn.Linear(vis_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        # Eqs. (4)-(5): cross-attention between visual regions and textual semantics.
        self.v2t_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.t2v_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)

        # Eq. (7): gating network on the concatenated modality-level vectors.
        self.gate_fc = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, vis_tokens, txt_tokens):
        """vis_tokens: (B, Nv, Dv), txt_tokens: (B, Nt, Dt) -> joint (B, Nv, Dh)."""
        v = self.vis_proj(vis_tokens)          # (B, Nv, Dh)
        t = self.txt_proj(txt_tokens)          # (B, Nt, Dh)

        # Bidirectional cross-modal semantic association.
        v_sem = self.norm_v(v + self.v2t_attn(v, t, t))   # visual queries attend to text
        t_sem = self.norm_t(t + self.t2v_attn(t, v, v))   # text queries attend to visuals

        # Eq. (6): modality-level representations via global average pooling.
        v_pool = v_sem.mean(dim=1)             # (B, Dh)
        t_pool = t_sem.mean(dim=1)             # (B, Dh)

        # Eq. (7): dynamic gating vector.
        g = torch.sigmoid(self.gate_fc(torch.cat([v_pool, t_pool], dim=-1)))  # (B, Dh)

        # Eq. (8): element-wise gated joint representation, anchored on visual tokens.
        joint = g.unsqueeze(1) * v_sem + (1.0 - g).unsqueeze(1) * t_sem.mean(dim=1, keepdim=True)
        return joint, g, v_pool, t_pool
