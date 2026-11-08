#!/usr/bin/env python3
"""
hapadapter.model_utils -- generic MLP construction
"""
import torch

def get_activation_module(name: str) -> torch.nn.Module:
    name = name.lower()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported head activation {name!r}:"
                        " choose from gelu, relu, tanh, silu, or add new activation"
                        " to hapadapter.model_utils.get_activation_module")

def build_prediction_head(
    input_size: int,
    output_size: int = 1,
    num_layers: int = 1,
    hidden_size: int = -1,
    activation: str = "gelu",
    dropout: float = 0.1,
) -> torch.nn.Module:
    """
    num_layers counts total Linear layers:
      1  -> Linear(input_size -> output_size)
      2  -> Linear(input->hidden) + activation + dropout + Linear(hidden->output)
      3+ -> deeper MLP
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
    if hidden_size == -1:
        hidden_size = input_size
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive or -1, got {hidden_size}.")
    if not (0 <= dropout < 1):
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    # Linear head
    if num_layers == 1:
        return torch.nn.Linear(input_size, output_size)

    # MLP head
    layers = [torch.nn.Linear(input_size, hidden_size),
              get_activation_module(activation),
              torch.nn.Dropout(dropout)]

    for _ in range(num_layers - 2):
        layers += [torch.nn.Linear(hidden_size, hidden_size),
                   get_activation_module(activation),
                   torch.nn.Dropout(dropout)]
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return torch.nn.Sequential(*layers)

def neff_weight(depth, like, s: float):
    """
    Precision weight w_i = n_eff(n_i) = n(1+s)/(n+s)
    ** normalised to mean 1 (LR invariant) **
    ** defaults to 1's when depth is None **
    """
    import torch
    if depth is None:
        return torch.ones_like(like)
    n = depth.float().view_as(like).clamp(min=1.0)
    w = n * (1.0 + s) / (n + s)
    return w / w.mean()