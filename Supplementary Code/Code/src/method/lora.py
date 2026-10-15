"""Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

Implements Eq. (30) of the manuscript: the task-adapted weight of an original
frozen weight matrix W is expressed as W + (alpha / r) * B A, where A and B are
two low-rank matrices and the original parameters are kept frozen during
training.
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A linear layer whose original weight is frozen and adapted by two low-rank matrices."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.05, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def effective_weight(self):
        """W + (alpha / r) * B A  (Eq. 30)."""
        return self.weight + self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        delta = nn.functional.linear(
            nn.functional.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return base + self.scaling * delta


def inject_lora(module, target_substrings=("q_proj", "k_proj", "v_proj", "o_proj", "ffn"),
                rank=8, alpha=16.0, dropout=0.05):
    """Replace matching nn.Linear submodules by LoRALinear copies (in place)."""
    replaced = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(s in name for s in target_substrings):
            new = LoRALinear(child.in_features, child.out_features, rank=rank,
                             alpha=alpha, dropout=dropout,
                             bias=child.bias is not None)
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(module, name, new)
            replaced.append(name)
        else:
            replaced.extend(inject_lora(child, target_substrings, rank, alpha, dropout))
    return replaced


def lora_parameters(module):
    """Return only the trainable LoRA parameters (original weights stay frozen)."""
    return [p for n, p in module.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
