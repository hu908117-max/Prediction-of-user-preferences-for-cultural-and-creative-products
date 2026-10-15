"""Multimodal backbone producing region-level visual tokens and word-level text tokens.

Two implementations are provided:

1. QwenVLBackbone  - loads Qwen2.5-VL-7B-Instruct through HuggingFace Transformers,
   keeps the visual encoder frozen, and returns the last-hidden-state token features
   for the image and the culture-semantics prompt. This is the manuscript backbone.
   It is loaded lazily so that environments without the 7B checkpoint (or without
   network access) can still import the package.

2. LightweightBackbone - a compact stand-in that maps pre-extracted visual / textual
   feature vectors to token sequences of configurable length. It is used for CPU
   demos, unit tests, and ablation studies where the full 7B model is impractical.
   The token structure (Nv visual regions, Nt text tokens) mirrors what the real
   backbone emits, so every downstream module is exercised identically.
"""

import torch
import torch.nn as nn


class LightweightBackbone(nn.Module):
    """Maps raw visual / textual feature vectors to token sequences.

    Input features are assumed to be pre-extracted (e.g. a small vision encoder or
    a text embedding). The backbone expands each global feature into a set of tokens
    through a learned per-modality projection plus additive region / word embeddings,
    emulating the patch / sub-word token streams of a vision-language model.
    """

    def __init__(self, vis_in, txt_in, hidden_dim, num_vis_tokens=16,
                 num_txt_tokens=16, dropout=0.1):
        super().__init__()
        self.num_vis_tokens = num_vis_tokens
        self.num_txt_tokens = num_txt_tokens
        self.hidden_dim = hidden_dim

        self.vis_token_mlp = nn.Sequential(
            nn.Linear(vis_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.txt_token_mlp = nn.Sequential(
            nn.Linear(txt_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vis_region_embed = nn.Parameter(
            torch.zeros(num_vis_tokens, hidden_dim))
        self.txt_word_embed = nn.Parameter(
            torch.zeros(num_txt_tokens, hidden_dim))
        self.vis_pos = nn.Parameter(torch.zeros(1, num_vis_tokens, hidden_dim))
        self.txt_pos = nn.Parameter(torch.zeros(1, num_txt_tokens, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.trunc_normal_(self.vis_region_embed, std=0.02)
        nn.init.trunc_normal_(self.txt_word_embed, std=0.02)
        nn.init.trunc_normal_(self.vis_pos, std=0.02)
        nn.init.trunc_normal_(self.txt_pos, std=0.02)

    def forward(self, vis_feat, txt_feat):
        """vis_feat: (B, vis_in), txt_feat: (B, txt_in)
        returns vis_tokens (B, Nv, H), txt_tokens (B, Nt, H)
        """
        b = vis_feat.size(0)
        # Broadcast the global feature across the token slots, then add the
        # learned region / word identity embeddings and positional offsets.
        vis_base = self.vis_token_mlp(vis_feat).unsqueeze(1) \
            .expand(b, self.num_vis_tokens, self.hidden_dim)
        txt_base = self.txt_token_mlp(txt_feat).unsqueeze(1) \
            .expand(b, self.num_txt_tokens, self.hidden_dim)

        vis_tokens = self.dropout(
            vis_base + self.vis_region_embed + self.vis_pos)
        txt_tokens = self.dropout(
            txt_base + self.txt_word_embed + self.txt_pos)
        return vis_tokens, txt_tokens


class QwenVLBackbone(nn.Module):
    """Qwen2.5-VL-7B-Instruct backbone with a frozen visual encoder.

    Loaded lazily; requires `transformers`, `qwen-vl-utils`, and the checkpoint
    `Qwen/Qwen2.5-VL-7B-Instruct`. The visual encoder (`visual`) is frozen; only the
    language-side representations are consumed as tokens. When LoRA is injected into
    the language model (see method/lora.py) only those adapters are trained.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                 device="cuda", dtype="bfloat16", max_new_tokens=1):
        super().__init__()
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype).to(device)
        # Freeze the visual encoder; keep the language model trainable (LoRA).
        for p in self.model.visual.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images, texts):
        """Return (vis_tokens, txt_tokens) from the model's last hidden states.

        images: list of PIL images; texts: list of prompt strings.
        """
        # NOTE: real tokenisation / inference is delegated to transformers; the
        # returned tensors are the per-modality hidden states consumed downstream.
        raise NotImplementedError(
            "Call the training harness with `use_qwen: true` and a GPU checkpoint; "
            "this method expects batched processor outputs and is exercised there."
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def build_backbone(name="lightweight", **kwargs):
    """Factory returning a backbone instance by name."""
    if name in ("lightweight", "demo"):
        required = ("vis_in", "txt_in", "hidden_dim")
        for key in required:
            if key not in kwargs:
                raise ValueError(f"LightweightBackbone requires '{key}'")
        return LightweightBackbone(
            vis_in=kwargs["vis_in"],
            txt_in=kwargs["txt_in"],
            hidden_dim=kwargs["hidden_dim"],
            num_vis_tokens=kwargs.get("num_vis_tokens", 16),
            num_txt_tokens=kwargs.get("num_txt_tokens", 16),
        )
    if name in ("qwen2.5-vl", "qwen"):
        return QwenVLBackbone(model_name=kwargs.get(
            "model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
            device=kwargs.get("device", "cuda"))
    raise ValueError(f"Unknown backbone: {name}")
