# -*- coding: utf-8 -*-
"""Training-mode grasp loss: force control + geometric penetration proxy.

Genesis 1.3.3 differentiability constraints (verified empirically against
the installed source; see doc/stage2.md v1.2 sec.4.2):

1. Only state queries via ``scene.get_state()`` carry gradients
   (``qpos`` / ``dofs_vel`` / ``links_pos`` / ``links_quat`` of
   RigidSolverState).  ``get_contacts()``,
   ``get_links_net_contact_force()``, ``get_dofs_position()`` and
   ContactForce sensors all return *detached* tensors, so a loss built on
   measured contact forces has no gradient path.
2. PD position targets (``control_dofs_position``) are replayed for primal
   correctness but have no input-gradient path (explicit exception:
   "Gradients with respect to PD control targets are not supported yet").
   The only differentiable control entry is ``control_dofs_force``.
3. The taped force tensor must be a *sceneless* ``gs.Tensor``
   (``gs.zeros(..., requires_grad=True)`` / ``gs.tensor(...)`` and any
   arithmetic on them).  Plain ``torch.Tensor`` results lack
   ``_backward_from_qd`` and fail the tape replay with AttributeError;
   scene-attached tensors would re-enter ``scene._backward()`` from
   ``gs.Tensor.backward`` and raise "Multiple backward calls not allowed".

Because measured contact forces are not differentiable-queryable, the
grasp loss uses a geometric penetration proxy computed from the
differentiable ``links_pos``.  With finger origins ``p_L``, ``p_R`` and
object center ``p_O``:

    u    = (p_L - p_R) / |p_L - p_R|            (closing axis)
    d_L  = (p_L - p_O) . u,  d_R = (p_O - p_R) . u
    pen_i = relu(D_TOUCH - d_i)                 (penetration depth, m)

    loss  = mean(relu(PEN* - pen_L) + relu(PEN* - pen_R))   squeeze to target
          + w_sym * mean(|pen_L - pen_R|)                    symmetry

``D_TOUCH`` is the finger-origin-to-object-center projection distance at
first touch; for the mini gripper + 0.04 m cube it is 0.025 m
(cube half-width 0.02 + finger inner offset 0.005).

This is the contact-stage loss of the two-stage scheme (sec.4.4): run the
pose-guidance stage first (bring the fingers near the object), then
optimize the finger forces with this loss.  The gradient entry point is
``scene.backward(loss)`` -- NOT ``loss.backward()``.
"""

import torch

import genesis as gs


def make_force_parameter(values):
    """
    Create a sceneless ``gs.Tensor`` leaf for differentiable force control.

    Must be called after ``gs.init``.  The result (and any arithmetic on
    it) keeps the ``gs.Tensor`` type, so ``control_dofs_force`` can tape
    it and replay input gradients in ``scene.backward``.
    """
    return gs.tensor(list(values), requires_grad=True)


def ramp_force(base, delta, step, ramp_steps):
    """
    Linear force ramp ``base + a * delta`` with ``a = min(1, (step+1)/ramp)``.

    Stays a ``gs.Tensor`` so the taped control input remains
    differentiable; use for a gentle close instead of a force step.
    """
    a = min(1.0, (step + 1) / max(int(ramp_steps), 1))
    return base + a * delta


class DiffGraspLoss:
    """
    Differentiable grasp loss for gradient-based optimization.

    Parameters
    ----------
    scene : gs.Scene
        Scene built with ``SimOptions(requires_grad=True)`` (diffsim on;
        ``Scene.requires_grad`` is read-only and cannot be toggled later).
    robot_entity : gs.RigidEntity
        Robot (gripper) entity.
    object_entity : gs.RigidEntity
        Target object entity (its first link is used as the object proxy).
    left_finger_name, right_finger_name : str
        Finger link names whose origins define the closing axis.
    contact_distance : float
        ``D_TOUCH`` [m]: finger-origin-to-object-center projection
        distance at first touch (penetration proxy zero point).
    target_penetration : float
        ``PEN*`` [m]: saturation level of the squeeze term (2-4x the
        nominal contact penetration; the loss stops decreasing once both
        sides reach it, preventing unbounded squeezing).
    symmetry_weight : float
        ``w_sym`` weighting the left/right symmetry term.
    """

    def __init__(self, scene, robot_entity, object_entity,
                 left_finger_name, right_finger_name, *,
                 contact_distance=0.025, target_penetration=0.001,
                 symmetry_weight=0.5):
        self.scene = scene
        self.robot_entity = robot_entity
        self.object_entity = object_entity
        self.l_link = robot_entity.get_link(left_finger_name)
        self.r_link = robot_entity.get_link(right_finger_name)
        self.o_link = object_entity.links[0]
        self.contact_distance = contact_distance      # D_TOUCH (m)
        self.target_penetration = target_penetration  # PEN* (m)
        self.symmetry_weight = symmetry_weight        # w_sym

    def _rigid_state(self):
        """RigidSolverState of the current scene state (differentiable)."""
        for st in self.scene.get_state().solvers_state:
            if st is not None and hasattr(st, "qpos"):
                return st
        raise RuntimeError("no rigid solver state in scene.get_state()")

    def finger_distances(self):
        """
        Differentiable finger-object distances ``(d_L, d_R)`` [m].

        Projections of the finger-origin-to-object-center vectors onto the
        closing axis ``u`` (unit vector from the right to the left finger
        origin).  Both are (B,) tensors carrying the autograd graph; they
        under-estimate true surface distance by the finger inner offset,
        which ``contact_distance`` absorbs.
        """
        links_pos = self._rigid_state().links_pos  # (B, n_links, 3)
        pL = links_pos[:, self.l_link.idx, :]
        pR = links_pos[:, self.r_link.idx, :]
        pO = links_pos[:, self.o_link.idx, :]
        u = (pL - pR) / (pL - pR).norm(dim=-1, keepdim=True).clamp(min=1e-6)
        dL = ((pL - pO) * u).sum(-1)
        dR = ((pO - pR) * u).sum(-1)
        return dL, dR

    def compute_loss(self):
        """
        Differentiable scalar grasp loss; backprop via
        ``scene.backward(loss)`` after a force-controlled rollout.
        """
        dL, dR = self.finger_distances()
        penL = torch.relu(self.contact_distance - dL)
        penR = torch.relu(self.contact_distance - dR)
        # Squeeze term: reach the target penetration on both sides,
        # saturated at PEN* (zero gradient beyond -- prevents unbounded
        # squeezing / object damage).
        loss_squeeze = torch.mean(
            torch.relu(self.target_penetration - penL)
            + torch.relu(self.target_penetration - penR))
        # Symmetry term: balanced penetration prevents single-side slip-out.
        loss_symmetry = torch.mean(torch.abs(penL - penR))
        return loss_squeeze + self.symmetry_weight * loss_symmetry
