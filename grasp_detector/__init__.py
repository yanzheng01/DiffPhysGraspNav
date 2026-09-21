# -*- coding: utf-8 -*-
"""
grasp_detector: collision-based grasp detection with a differentiable
training path, per doc/stage2.md (v1.2, Genesis 1.3.3).

Two working modes:
  - inference (GraspDetector): boolean judgment of
    "bilateral contact + effective force + lift confirmation";
  - training (DiffGraspLoss): differentiable penetration-proxy loss over
    force-controlled rollouts, backpropagated via scene.backward(loss).

Quick start:
    import genesis as gs
    from grasp_detector import GraspDetector, DiffGraspLoss
"""

from .detector import GraspDetector, detect_parallel
from .diff_loss import DiffGraspLoss, make_force_parameter, ramp_force
from .utils import to_bool, to_float, force_magnitude

__version__ = "1.2.0"

__all__ = [
    "GraspDetector",
    "DiffGraspLoss",
    "detect_parallel",
    "make_force_parameter",
    "ramp_force",
    "to_bool",
    "to_float",
    "force_magnitude",
]
