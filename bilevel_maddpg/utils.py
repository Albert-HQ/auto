"""Utility helpers for bilevel MADDPG agents."""
from __future__ import annotations

import os
from typing import Sequence

import torch
from torch import nn


def _get_sequence_value(seq, index: int) -> int:
    """Return the integer value at index from a sequence-like argument."""
    if isinstance(seq, Sequence):
        if len(seq) == 0:
            raise IndexError("Empty sequence provided for dimension lookup")
        if index >= len(seq):
            raise IndexError(f"Index {index} is out of bounds for sequence of length {len(seq)}")
        return int(seq[index])
    return int(seq)


def agent_obs_dim(args, agent_id: int) -> int:
    """Return observation dimension for the given agent."""
    if not hasattr(args, "obs_shape"):
        raise AttributeError("args must define obs_shape before agent initialisation")
    return _get_sequence_value(args.obs_shape, agent_id)


def agent_action_dim(args, agent_id: int) -> int:
    """Return action dimension for the given agent."""
    if not hasattr(args, "action_shape"):
        raise AttributeError("args must define action_shape before agent initialisation")
    return _get_sequence_value(args.action_shape, agent_id)


def safe_load_state_dict(module: nn.Module, file_path: str, device, label: str) -> bool:
    """Load a state_dict if it exists, guarding against shape mismatches."""
    if not os.path.exists(file_path):
        return False
    try:
        state_dict = torch.load(file_path, map_location=device)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] Loading {label}: missing keys {missing}, unexpected keys {unexpected}")
        return True
    except RuntimeError as err:
        print(f"[WARN] Failed to load {label} from {file_path}: {err}")
    except Exception as err:  # pylint: disable=broad-except
        print(f"[WARN] Unexpected error loading {label} from {file_path}: {err}")
    return False
