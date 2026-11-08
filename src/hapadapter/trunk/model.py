#!/usr/bin/env python3
"""
hapadapter.trunk.model —- functional fine-tuning of a pretrained DNA encoder

Encoder -> pooled representation -> prediction head

  * task="regression": mu = head(pool(seq))
        - One window -> one score
        - MSE, optionally precision-weighted by depth
        - Single-track (scalar) or multi-track (one score per tissue)

  * task="classification": ell = head(pool(seq))
        - One window -> one logit
        - BCE, optionally precision-weighted
"""
from typing import Optional

import torch
import transformers
from transformers.modeling_outputs import SequenceClassifierOutput

from hapadapter.model_utils import (
    build_prediction_head,
    get_activation_module,
    neff_weight,
)

class FunctionalFinetuneForSequencePrediction(torch.nn.Module):
    """
    A pretrained DNA encoder fine-tuned on a functional genomics signal:
    * Encoder (any HF AutoModel) -> pooled representation h -> main_head -> score *
    """

    def __init__(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        pooling_mode: str = "center_mean",
        center_pool_width: int = 5,
        head_num_layers: int = 1,
        head_hidden_size: int = -1,
        head_activation: str = "gelu",
        head_dropout: float = 0.1,
        neff_s: float = 20.0,
        task: str = "regression",
        num_labels: int = 1,
    ):
        super().__init__()
        if pooling_mode not in ("cls", "center_mean", "mean"):
            raise ValueError(f"pooling_mode must be 'cls', 'center_mean', or 'mean', got {pooling_mode!r}.")
        if pooling_mode == "center_mean" and center_pool_width % 2 == 0:
            raise ValueError("center_pool_width must be odd for symmetric center pooling.")
        if neff_s <= 0:
            raise ValueError(f"neff_s must be > 0, got {neff_s}.")
        if task not in ("regression", "classification"):
            raise ValueError(f"task must be 'regression' or 'classification', got {task!r}.")

        num_labels = int(num_labels) # head width T (multi-track target)
        if num_labels < 1:
            raise ValueError(f"num_labels must be >= 1, got {num_labels}.")

        if task == "classification" and num_labels != 1:
            raise ValueError(
                f"The current classification head only supports binary labels (one logit) -> num_labels must be 1, got {num_labels}."
            )
            
        self.pooling_mode = pooling_mode
        self.center_pool_width = int(center_pool_width)
        self.neff_s = float(neff_s)
        self.task = task
        self.num_labels = num_labels
    
        # Pretrained encoder (any HF AutoModel with a tokenizer that emits input_ids)
        self.backbone = transformers.AutoModel.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, trust_remote_code=True,
        )
        hidden_size = self.backbone.config.hidden_size
        dropout_prob = getattr(self.backbone.config, "hidden_dropout_prob", 0.1)
        self.dropout = torch.nn.Dropout(dropout_prob)

        self.main_head = build_prediction_head(
            input_size=hidden_size,
            output_size=self.num_labels,
            num_layers=head_num_layers,
            hidden_size=head_hidden_size,
            activation=head_activation,
            dropout=head_dropout,
        )

    def init_backbone_from(self, checkpoint_path: str, map_location="cpu", include_head: bool = False) -> int:
        """
        Load the `backbone.*` weights from a fine-tuned checkpoint into this model
        ** returns the number of tensors loaded **

        include_head=True also restores `main_head.*`
        (defaults to False because the intended use is transferring 
        a conditioned encoder to a NEW task, whose head must start fresh)
        """
        sd = torch.load(checkpoint_path, map_location=map_location)
        
        if isinstance(sd, dict) and "state_dict" in sd and "backbone.embeddings" not in str(sd.keys()):
            sd = sd["state_dict"]
        
        prefixes = ("backbone.", "main_head.") if include_head else ("backbone.",)
        backbone_sd = {k: v for k, v in sd.items() if k.startswith(prefixes)}
        
        if not any(k.startswith("backbone.") for k in backbone_sd):
            raise ValueError(
                f"No 'backbone.*' keys found in {checkpoint_path}; got prefixes {sorted({k.split('.')[0] for k in sd})}"
            )
        
        if include_head and not any(k.startswith("main_head.") for k in backbone_sd):
            raise ValueError(
                f"include_head=True but {checkpoint_path} has no 'main_head.*' keys; "
                f"got prefixes {sorted({k.split('.')[0] for k in sd})}."
            )

        missing, unexpected = self.load_state_dict(backbone_sd, strict=False)
        # 'missing' lists the fresh head params (these should happen)
        # 'unexpected' should be empty
        if unexpected:
            raise ValueError(f"Unexpected keys while loading backbone: {unexpected[:5]} ...")
        if include_head:
            still = [k for k in missing if k.startswith("main_head.")]
            if still:
                raise ValueError(
                    f"main_head weights did NOT load: {still[:4]}. The head geometry differs from "
                    f"the checkpoint's -- build this model with the trunk's own head_num_layers / "
                    f"head_hidden_size / head_activation."
                )
        return len(backbone_sd)

    # ---- Pooling -----------------------------------------------------------
    def _pool(self, backbone_outputs, attention_mask=None):
        if hasattr(backbone_outputs, "last_hidden_state") and backbone_outputs.last_hidden_state is not None:
            seq = backbone_outputs.last_hidden_state
        elif isinstance(backbone_outputs, dict) and "last_hidden_state" in backbone_outputs:
            seq = backbone_outputs["last_hidden_state"]
        elif isinstance(backbone_outputs, (tuple, list)) and torch.is_tensor(backbone_outputs[0]) \
                and backbone_outputs[0].ndim == 3:
            seq = backbone_outputs[0]
        else:
            raise ValueError(f"Cannot extract token hidden states from {type(backbone_outputs)}.")

        if self.pooling_mode == "cls":
            return seq[:, 0, :]

        if self.pooling_mode == "mean":
            # masked all-token mean over the valid (non-pad) tokens of each window
            if attention_mask is None:
                return seq.mean(dim=1)
            m = attention_mask.unsqueeze(-1).to(seq.dtype)          # (B, L, 1)
            summed = (seq * m).sum(dim=1)                           # (B, H)
            denom = m.sum(dim=1).clamp(min=1.0)                     # (B, 1); avoid divide by 0 with clamp
            return summed / denom

        # center_mean: mean-pool center_pool_width tokens around the middle of each valid window
        bsz, max_len, _ = seq.shape
        half = self.center_pool_width // 2
        pooled = []
        for b in range(bsz):
            valid = int(attention_mask[b].sum().item()) if attention_mask is not None else max_len
            valid = max(valid, 1)
            center = valid // 2
            start = max(0, center - half)
            end = min(valid, center + half + 1)
            pooled.append(seq[b, start:end, :].mean(dim=0))
        return torch.stack(pooled, dim=0)

    def _pool_one(self, input_ids, attention_mask, apply_dropout=True, **kwargs):
        """One window -> pooled representation h in R^hidden"""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)
        pooled = self._pool(out, attention_mask=attention_mask)
        return self.dropout(pooled) if apply_dropout else pooled

    def _score_one(self, input_ids, attention_mask, **kwargs):
        """
        One window -> logit(s) + fused representation
        ** returns (B,) when num_labels = 1 and (B, T) when num_labels = T > 1 **
        """
        pooled = self._pool_one(input_ids, attention_mask, **kwargs)
        out = self.main_head(pooled)  # (B, T)
        if self.num_labels == 1:
            out = out.squeeze(-1)     # (B,)
        return out, pooled

    # (Optional) precision weighting
    def _neff_weight(self, depth, like):
        """Wrapper over model_utils.neff_weight, bound to this model's neff_s"""
        return neff_weight(depth, like, self.neff_s)

    def forward(
        self,
        input_ids=None,            # hap1 window (sequence1)
        attention_mask=None,
        input_ids_alt=None,        # hap2 window (sequence2)
        attention_mask_alt=None,
        labels=None,               # regression: binding label, (B,) scalar or (B,T) per-track; classification: 0/1
        depth=None,                # n = depth (privileged; weight only)
        label_mask=None,           # (B,T) 0/1 observed mask for multi-track targets (no mask => all observed)
        **kwargs,
    ):
        if self.task == "classification":
            ell, _ = self._score_one(input_ids, attention_mask, **kwargs)
            loss = None
            if labels is not None:
                y = labels.float().view_as(ell)
                w = self._neff_weight(depth, ell)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(ell, y, reduction="none")
                loss = (w * bce).mean()
            return SequenceClassifierOutput(loss=loss, logits=ell.unsqueeze(-1))

        # regression: single-window functional score
        mu, _ = self._score_one(input_ids, attention_mask, **kwargs) # (B,) or (B,T)

        # multi-track (num_labels > 1): masked multi-output MSE over observed tissues
        if self.num_labels > 1:
            loss = None
            if labels is not None:
                y = labels.float().view_as(mu)
                if label_mask is not None:
                    m = label_mask.float().view_as(mu)
                else:
                    m = torch.ones_like(mu)
                sq = m * (mu - y) ** 2
                denom = m.sum().clamp(min=1.0)
                loss = sq.sum() / denom
            return SequenceClassifierOutput(loss=loss, logits=mu)  # (B, T)

        # scalar path (num_labels == 1): single-track functional score
        loss = None
        if labels is not None:
            y = labels.float().view_as(mu)
            w = self._neff_weight(depth, mu)
            loss = (w * (mu - y) ** 2).mean()

        return SequenceClassifierOutput(loss=loss, logits=mu.unsqueeze(-1))
