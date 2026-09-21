# -*- coding: utf-8 -*-
"""Tensor helpers shared by the inference and training paths (stage2.md sec.5.2)."""

import torch


def to_bool(x):
    """Convert a Genesis return value to a Python bool (single environment)."""
    return bool(x.item() if hasattr(x, "item") else x)


def to_float(x):
    """Convert a Genesis return value to a Python float (single environment)."""
    return float(x.item() if hasattr(x, "item") else x)


def force_magnitude(force):
    """
    Compute the magnitude of contact forces.

    Accepts any tensor whose last dim is 3: (3,), (n_contacts, 3),
    (n_envs, n_contacts, 3); correspondingly returns a scalar,
    (n_contacts,) or (n_envs, n_contacts).
    """
    if hasattr(force, "shape") and force.shape[-1] == 3:
        return torch.norm(force, dim=-1)
    return force
