"""End-to-end museum cultural-product user-preference prediction model.

Composes the manuscript stages:

  Stage 1  Multimodal feature extraction + gated fusion        (backbone + GatedFusion)
  Stage 2  User-preference-guided cross-attention              (UserPreferenceModule)
  Stage 3  Bilinear interaction + InfoNCE matching             (PreferenceMatching)
  Stage 4  Joint loss + LoRA-efficient fine-tuning of Qwen2.5-VL

For the Qwen2.5-VL backbone, LoRA adapters are injected into the language-side
attention and feed-forward projections (method/lora.py) and only those adapters plus
the added task modules are trained; the visual encoder stays frozen. For the
lightweight backbone used in CPU / demo runs, all parameters are compact and trained
directly, exercising the identical downstream computation graph.
"""

import torch
import torch.nn as nn

from ..method.gated_fusion import GatedFusion
from ..method.user_preference import UserPreferenceModule
from ..method.preference_matching import PreferenceMatching
from ..method.lora import inject_lora, lora_parameters
from .backbone import build_backbone


class MuseumPreferenceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        hidden = cfg["hidden_dim"]

        # Stage 1: backbone -> token streams, then gated fusion.
        self.backbone = build_backbone(
            name=cfg.get("backbone", "lightweight"),
            vis_in=cfg["vis_in"], txt_in=cfg["txt_in"], hidden_dim=hidden,
            num_vis_tokens=cfg.get("num_vis_tokens", 16),
            num_txt_tokens=cfg.get("num_txt_tokens", 16),
        )
        self.fusion = GatedFusion(
            vis_dim=hidden, txt_dim=hidden, hidden_dim=hidden,
            num_heads=cfg.get("num_heads", 4),
            dropout=cfg.get("dropout", 0.1),
        )

        # Stage 2: user preference representation.
        num_emotions = 8  # eight kept ArtEmis sentiments
        self.user_module = UserPreferenceModule(
            num_emotions=num_emotions, text_dim=cfg["txt_in"],
            joint_dim=hidden, hidden_dim=hidden,
            dropout=cfg.get("dropout", 0.1),
        )

        # Stage 3: bilinear interaction matching head.
        self.matcher = PreferenceMatching(
            dim=hidden, bilinear_dim=cfg.get("bilinear_dim", 256),
            mlp_hidden=cfg.get("mlp_hidden", 512),
            dropout=cfg.get("dropout", 0.1),
        )

        # Stage 4: LoRA for the Qwen backbone (no-op for lightweight backbone).
        self.lora_replaced = []
        if cfg.get("use_qwen", False):
            self.lora_replaced = inject_lora(
                self.backbone.model if hasattr(self.backbone, "model") else self.backbone,
                target_substrings=("q_proj", "k_proj", "v_proj", "o_proj", "ffn"),
                rank=cfg.get("lora_rank", 8),
                alpha=cfg.get("lora_alpha", 16.0),
                dropout=cfg.get("lora_dropout", 0.05),
            )

    # ------------------------------------------------------------------ #
    def encode_product(self, vis, txt):
        """Stage 1 for a single product batch. vis/txt: (B, Din) -> fusion outputs."""
        vis_tokens, txt_tokens = self.backbone(vis, txt)
        joint, gate, v_pool, t_pool = self.fusion(vis_tokens, txt_tokens)
        return joint, gate, v_pool, t_pool

    def _encode_history(self, hist_vis, hist_txt):
        """Encode a batch of historical products. Shapes: (B, Hh, Din)."""
        b, hh = hist_vis.shape[:2]
        fv = hist_vis.reshape(b * hh, -1)
        ft = hist_txt.reshape(b * hh, -1)
        joint, gate, v_pool, t_pool = self.encode_product(fv, ft)
        d = joint.shape[-1]
        joint = joint.reshape(b, hh, -1, d)
        return joint

    def forward(self, batch):
        """Full forward. Returns a dict of representations, scores, and artifacts."""
        # Candidate (target) product.
        tgt_joint, _, tgt_vpool, tgt_tpool = self.encode_product(
            batch["target_vis"], batch["target_txt"])
        product_repr = tgt_joint.mean(dim=1)           # (B, H)  multimodal joint rep

        # Positive / negative products for InfoNCE.
        pos_joint, _, _, _ = self.encode_product(batch["pos_vis"], batch["pos_txt"])
        neg_joint, _, _, _ = self.encode_product(batch["neg_vis"], batch["neg_txt"])
        pos_repr = pos_joint.mean(dim=1)               # (B, H)
        neg_repr = neg_joint.mean(dim=1)               # (B, H)

        # Historical products -> token-level joint representations.
        hist_joint = self._encode_history(batch["hist_vis"], batch["hist_txt"])

        # Stage 2: user preference representation from history.
        user_repr, beta, attn = self.user_module(
            batch["hist_emo"], batch["hist_txt"], hist_joint)

        # Stage 3: preference prediction.
        prob, score, z = self.matcher(user_repr, product_repr)

        return {
            "user_repr": user_repr,
            "product_repr": product_repr,
            "pos_repr": pos_repr,
            "neg_repr": neg_repr,
            "vis_repr": tgt_vpool,        # for cross-modal contrast (Eq. 27)
            "txt_repr": tgt_tpool,
            "beta": beta,
            "attn": attn,
            "gate": None,
            "prob": prob,
            "score": score,
            "bilinear": z,
        }

    def trainable_parameters(self):
        """Return the parameters updated during training.

        For the Qwen backbone only the LoRA adapters are returned; for the
        lightweight backbone the whole module is returned alongside the task modules.
        """
        if self.lora_replaced:
            return lora_parameters(self.backbone.model
                                  if hasattr(self.backbone, "model") else self.backbone)
        return [p for p in self.parameters() if p.requires_grad]
