#!/usr/bin/env python3

"""
hapadapter.trunk.loading -- encoder trunk I/O
"""

import glob
import json
import os
from typing import Optional

import torch
import transformers

from hapadapter.trunk.model import FunctionalFinetuneForSequencePrediction

DEFAULT_MODEL_MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_run_config(checkpoint_dir: str) -> dict:
    """Read run_config.json written by the trainer (errors if missing)"""
    path = os.path.join(checkpoint_dir, "run_config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No run_config.json in {checkpoint_dir}. It should have written by the trainer"
            f"at save time, but you can also pass task/head fields explicitly instead."
        )
    with open(path) as f:
        return json.load(f)

def apply_overrides(run_config: dict, overrides: dict) -> dict:
    """Override run_config fields with any non-None values (from CLI)"""
    rc = dict(run_config)
    for k, v in (overrides or {}).items():
        if v is not None:
            rc[k] = v
    return rc

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

def find_weights_file(checkpoint_dir: str) -> str:
    """
    Locate the saved weights
    - first searches for top-level "best model" (written iff load_best_model_at_end is on)
    - falls back to latest checkpoint
    """
    candidates = [
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
        os.path.join(checkpoint_dir, "model.safetensors"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
    )

    for ckpt in reversed(ckpts):
        for name in ("pytorch_model.bin", "model.safetensors"):
            c = os.path.join(ckpt, name)
            if os.path.exists(c):
                return c

    raise FileNotFoundError(
        f"No pytorch_model.bin / model.safetensors found in {checkpoint_dir} "
        f"or its checkpoint-* subdirectories."
    )

def _load_state_dict(weights_path: str) -> dict:
    # model.safetensors
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(weights_path)
    # pytorch_model.bin
    return torch.load(weights_path, map_location="cpu")

def load_model_weights(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    """Load a state dict into model, reporting missing/unexpected keys"""
    state_dict = _load_state_dict(weights_path)
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))

    print(f"Loaded weights from {weights_path}")
    print(f"  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print("  first missing:", missing[:10])
    if unexpected:
        print("  first unexpected:", unexpected[:10])

    # Derive the head prefix from the model's own parameters
    task = getattr(model, "task", "regression")
    head_prefixes = tuple(n for n in ("main_head", "proj", "dist_a", "dist_b")
                          if any(k.split(".")[0] == n for k in model.state_dict()))

    if not head_prefixes:
        raise RuntimeError(
            f"{type(model).__name__} exposes none of the expected head parameters "
            "(main_head / proj / dist_a / dist_b). Is this a composed model? "
            "Try loading the encoder trunk and the head separately "
            "by calling hapadapter.loading.load_model_and_tokenizer"
        )

    head_missing = [k for k in missing if k.split(".")[0] in head_prefixes]
    if head_missing:
        raise RuntimeError(
            f"{task} head weights did not load ({head_missing[:5]}...). The checkpoint and the "
            f"run_config architecture likely disagree. Refusing to run on an untrained head!"
        )

    return model

# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def build_model(run_config: dict, device: str = "cpu") -> torch.nn.Module:
    """Instantiate the trained architecture from a (possibly overridden) run_config"""
    if run_config.get("use_lora"):
        raise NotImplementedError(
            "run_config has use_lora=True, but LoRA checkpoints are not handled by the current "
            "version of this codebase."
        )

    head_num_layers = run_config["head_num_layers"]
    if head_num_layers >= 2 and "head_activation" not in run_config:
        raise KeyError(
            "run_config.json is missing 'head_activation', which this checkpoint needs: "
            f"head_num_layers={head_num_layers} builds an MLP head that requires a nonlinearity! "
            "Add the activation used at training time."
        )

    model = FunctionalFinetuneForSequencePrediction(
        model_name_or_path=run_config["model_name_or_path"],
        cache_dir=run_config.get("cache_dir"),
        pooling_mode=run_config["pooling_mode"],
        center_pool_width=run_config.get("center_pool_width", 5),
        head_num_layers=head_num_layers,
        head_hidden_size=run_config["head_hidden_size"],
        head_activation=run_config.get("head_activation", "gelu"),
        head_dropout=run_config.get("head_dropout", 0.1),
        neff_s=run_config.get("neff_s", 20.0),
        task=run_config.get("task", "regression"),
        num_labels=run_config.get("num_labels", 1),
    )
    return model.to(device)

def load_tokenizer(run_config: dict):
    tok = transformers.AutoTokenizer.from_pretrained(
        run_config["model_name_or_path"],
        cache_dir=run_config.get("cache_dir"),
        model_max_length=run_config.get("model_max_length", DEFAULT_MODEL_MAX_LENGTH),
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    # collator pads with pad_token_id and derives attention_mask from it
    if tok.pad_token_id is None:
        raise ValueError(
            f"{run_config['model_name_or_path']}: tokenizer has no pad_token_id, which the "
            f"collator requires -- every batch is padded to its longest member. Encoders "
            f"that come without one need it set here.")
    return tok

def load_model_and_tokenizer(checkpoint_dir: str, device: str = "cpu", overrides: dict = None):
    run_config = apply_overrides(load_run_config(checkpoint_dir), overrides or {})
    model = build_model(run_config, device=device)
    load_model_weights(model, find_weights_file(checkpoint_dir))
    model.eval()
    tokenizer = load_tokenizer(run_config)
    return model, tokenizer, run_config