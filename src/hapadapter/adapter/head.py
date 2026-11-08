#!/usr/bin/env python3
"""
hapadapter.adapter.head -- the HapAdapter head: hap1/hap2 trunk embeddings -> personalized prediction
"""

import torch

from hapadapter.model_utils import (
    build_prediction_head,
    get_activation_module,
    neff_weight,
)

# a_raw = ln(e - 1) -> softplus(a_raw) == 1, so twin readout starts as id map
DIST_A_INIT = 0.5413

def build_head_modules(
    d_in: int,
    head_variant: str = "twin",
    proj_dim: int = 128,
    num_labels: int = 1,
    head_num_layers: int = 1,
    head_hidden_size: int = -1,
    head_activation: str = "gelu",
    head_dropout: float = 0.1,
) -> dict:
    if head_variant not in ("twin", "single"):
        raise ValueError(f"head_variant must be 'twin' or 'single', got {head_variant!r}.")
    
    if num_labels != 1:
        raise ValueError(
            f"the HapAdapter head is currently only configured for binary classification. "
            f"{num_labels} is not supported."
        )

    head_modules = dict(main_head=None, proj=None, dist_a=None, dist_b=None)

    if head_variant == "single":
        head_modules["main_head"] = build_prediction_head(
            input_size=d_in,
            output_size=1,
            num_layers=head_num_layers,
            hidden_size=head_hidden_size,
            activation=head_activation,
            dropout=head_dropout,
        )
        return head_modules

    # else: twin
    if proj_dim <= 0:
        raise ValueError(f"proj_dim must be > 0 for the twin head, got {proj_dim}.")
    
    head_modules["proj"] = build_prediction_head(
        input_size=d_in,
        output_size=proj_dim,
        num_layers=head_num_layers,
        hidden_size=head_hidden_size,
        activation=head_activation,
        dropout=head_dropout,
    )

    head_modules["dist_a"] = torch.nn.Parameter(torch.tensor(DIST_A_INIT))
    head_modules["dist_b"] = torch.nn.Parameter(torch.tensor(0.0))
    return head_modules

def twin_logit(proj, dist_a, dist_b, dropout, h1, h2):
    """logit P(ASB) = softplus(dist_a) * ||proj(h1) - proj(h2)|| + dist_b"""
    z1 = proj(dropout(h1))
    z2 = proj(dropout(h2))
    s = torch.linalg.vector_norm(z1 - z2, dim=-1)      # >= 0, symmetric in (h1, h2)
    a = torch.nn.functional.softplus(dist_a)           # a > 0: logit monotone in the distance
    return a * s + dist_b

def asb_loss(ell, labels, depth, neff_s):
    """
    Precision-weighted BCE on the ASB logit
    """
    y = labels.float().view_as(ell)
    w = neff_weight(depth, ell, neff_s)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(ell, y, reduction="none")
    return (w * bce).mean()

class HapAdapterHead(torch.nn.Module):
    """HapAdapter: pooled embeddings -> binary logit"""

    def __init__(self, d_in: int, head_variant: str = "twin", proj_dim: int = 128,
                 head_num_layers: int = 1, head_hidden_size: int = -1,
                 head_activation: str = "gelu", head_dropout: float = 0.1,
                 pool_dropout: float = 0.1):
        super().__init__()
        self.task = "classification"
        self.head_variant = head_variant
        self.d_in = int(d_in)
        self.proj_dim = int(proj_dim)
        self.head_num_layers = int(head_num_layers)
        self.head_hidden_size = int(head_hidden_size)
        self.head_activation = head_activation
        self.head_dropout = float(head_dropout)
        self.dropout = torch.nn.Dropout(pool_dropout)
        for name, mod in build_head_modules(
                d_in=d_in, head_variant=head_variant, proj_dim=proj_dim,
                head_num_layers=head_num_layers, head_hidden_size=head_hidden_size,
                head_activation=head_activation, head_dropout=head_dropout).items():
            setattr(self, name, mod)

    def forward(self, h1, h2=None):
        """h1, h2 are pooled haplotype embeddings with shape (B, d_in)"""
        if self.head_variant == "twin":
            if h2 is None:
                raise RuntimeError("the twin head needs both haplotypes (got h2=None).")
            return twin_logit(self.proj, self.dist_a, self.dist_b, self.dropout, h1, h2)
        out = self.main_head(self.dropout(h1))
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def config(self) -> dict:
        """Everything needed to rebuild an instantiation of the HapAdapter head"""
        return dict(d_in=self.d_in,
                    task=self.task,
                    head_variant=self.head_variant,
                    proj_dim=self.proj_dim,
                    head_num_layers=self.head_num_layers,
                    head_hidden_size=self.head_hidden_size,
                    head_activation=self.head_activation,
                    head_dropout=self.head_dropout,
                    pool_dropout=float(self.dropout.p))