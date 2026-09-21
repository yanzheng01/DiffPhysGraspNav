# -*- coding: utf-8 -*-
"""Inference-mode grasp detection: boolean judgment (stage2.md sec.7.1 / sec.9)."""

import torch

from .utils import to_bool, to_float


class GraspDetector:
    """
    Boolean grasp-success detector for a single (non-batched) environment.

    Success criterion (stage2.md sec.1.3, three factors):

        grasp success = bilateral contact (left AND right finger touching)
                        AND effective force (both sides above threshold)
                        AND lift confirmation (object has left the support
                        surface, bilateral contact and force still holding)

    The two judgment instants:
      - closing phase: call ``detect(confirm_lift=False)`` after the gripper
        has closed and settled -- judges the first two factors only;
      - lift phase: call ``detect()`` (confirm_lift=True) after a short lift
        (5-10 cm) -- judges all three factors. The "off the support" signal
        must NOT be used before the lift: a table-top object touches the
        support until it is actually picked up.

    Parameters
    ----------
    scene : gs.Scene
        Genesis scene (single environment, ``scene.build(n_envs=0)``).
    robot_entity : gs.RigidEntity
        Robot (gripper) entity.
    object_entity : gs.RigidEntity
        Target object entity.
    left_finger_name, right_finger_name : str
        Finger-tip link names (verify with
        ``[l.name for l in robot.links]`` first).
    support_entity : gs.RigidEntity, optional
        Support surface entity, used by lift confirmation to verify the
        object has left it. If None, the lift-confirmation factor is
        skipped and ``on_support`` is reported as None.
    force_threshold : float
        Per-side contact force threshold [N]. Recommended
        ``mass * g * 1.5`` (stage2.md appendix B).
    """

    def __init__(self, scene, robot_entity, object_entity,
                 left_finger_name, right_finger_name,
                 support_entity=None, force_threshold=0.5):
        self.scene = scene
        self.robot_entity = robot_entity
        self.object_entity = object_entity
        self.support_entity = support_entity
        self.force_threshold = force_threshold
        self.l_link = robot_entity.get_link(left_finger_name)
        self.r_link = robot_entity.get_link(right_finger_name)

    def _finger_force(self, contacts, link):
        """
        Sum of robot-object contact forces acting on one finger link.

        In a contact pair (A, B), force_a acts on the A-side geom and
        force_b on the B-side geom; the robot may appear on either side,
        so both sides are matched by the global link index. Works for
        single-env layouts (n_contacts, ...) and batched layouts
        (n_envs, n_contacts, ...) -- invalid slots masked via valid_mask.
        Returns (3,) or (n_envs, 3).
        """
        wa = (contacts["link_a"] == link.idx).unsqueeze(-1)
        wb = (contacts["link_b"] == link.idx).unsqueeze(-1)
        if "valid_mask" in contacts:  # parallel envs: mask invalid slots
            wa = wa & contacts["valid_mask"].unsqueeze(-1)
            wb = wb & contacts["valid_mask"].unsqueeze(-1)
        return (contacts["force_a"] * wa
                + contacts["force_b"] * wb).sum(dim=-2)

    def _touching(self, contacts, link):
        """Boolean: any contact pair involves this finger link."""
        wa = contacts["link_a"] == link.idx
        wb = contacts["link_b"] == link.idx
        if "valid_mask" in contacts:
            wa = wa & contacts["valid_mask"]
            wb = wb & contacts["valid_mask"]
        return to_bool(wa.any() or wb.any())

    def detect(self, confirm_lift=True, return_debug_info=False):
        """
        Judge grasp success.

        Parameters
        ----------
        confirm_lift : bool
            If True (default), apply the full three-factor criterion -- call
            this only after the lift. If False, judge bilateral contact and
            force threshold only (closing phase).
        return_debug_info : bool
            If True, return ``(success, debug_info_dict)``.

        Returns
        -------
        bool or (bool, dict)
        """
        contacts = self.robot_entity.get_contacts(
            with_entity=self.object_entity)

        touching_left = self._touching(contacts, self.l_link)
        touching_right = self._touching(contacts, self.r_link)
        force_left = to_float(torch.norm(
            self._finger_force(contacts, self.l_link)))
        force_right = to_float(torch.norm(
            self._finger_force(contacts, self.r_link)))
        force_ok = (force_left > self.force_threshold
                    and force_right > self.force_threshold)

        # Lift confirmation: the object must have left the support surface.
        if self.support_entity is not None:
            sup = self.object_entity.get_contacts(
                with_entity=self.support_entity)
            if "valid_mask" in sup:
                on_support = to_bool(sup["valid_mask"].any())
            else:
                on_support = sup["force_a"].shape[0] > 0
        else:
            on_support = None

        success = touching_left and touching_right and force_ok
        if confirm_lift and self.support_entity is not None:
            success = success and (not on_support)

        if return_debug_info:
            debug = {
                "touching_left": touching_left,
                "touching_right": touching_right,
                "force_left": force_left,
                "force_right": force_right,
                "force_threshold_passed": force_ok,
                "on_support": on_support,
            }
            return success, debug
        return success


def detect_parallel(robot, obj, support, l_link, r_link,
                    force_threshold=1.0, confirm_lift=True):
    """
    Batched judgment for parallel environments (stage2.md sec.9).

    All logic stays in tensor form; no ``.item()`` per-environment loops.
    Call after the lift when ``confirm_lift=True``.

    Parameters
    ----------
    robot, obj : gs.RigidEntity
        Robot and object entities (batched scene, n_envs >= 1).
    support : gs.RigidEntity or None
        Support surface entity for lift confirmation (None skips it).
    l_link, r_link : RigidLink
        Finger links (``robot.get_link(name)``).
    force_threshold : float
        Per-side contact force threshold [N].
    confirm_lift : bool
        Require the object to have left the support surface.

    Returns
    -------
    torch.Tensor
        Indices of the environments where the grasp succeeded.
    """
    contacts = robot.get_contacts(with_entity=obj)
    valid = contacts["valid_mask"]                    # (n_envs, n_contacts)
    # Finger contact masks: the robot may appear on side a or b of a pair,
    # so match the global link index on both sides.
    la = (contacts["link_a"] == l_link.idx) & valid
    lb = (contacts["link_b"] == l_link.idx) & valid
    ra = (contacts["link_a"] == r_link.idx) & valid
    rb = (contacts["link_b"] == r_link.idx) & valid
    touch_l = (la | lb).any(dim=-1)                   # (n_envs,)
    touch_r = (ra | rb).any(dim=-1)
    fa, fb = contacts["force_a"], contacts["force_b"] # (n_envs, n_contacts, 3)
    # force_a acts on the a-side geom, force_b on the b-side geom; accumulate
    # per side according to the link match.
    force_l = (fa * la.unsqueeze(-1) + fb * lb.unsqueeze(-1)).sum(dim=1).norm(dim=-1)  # (n_envs,)
    force_r = (fa * ra.unsqueeze(-1) + fb * rb.unsqueeze(-1)).sum(dim=1).norm(dim=-1)
    is_success = (touch_l & touch_r
                  & (force_l > force_threshold)
                  & (force_r > force_threshold))
    if confirm_lift and support is not None:
        # Lift confirmation: the object must have left the support surface
        # (call this mode only after the short lift).
        on_support = obj.get_contacts(with_entity=support)["valid_mask"].any(dim=-1)  # (n_envs,)
        is_success = is_success & (~on_support)
    return torch.where(is_success)[0]  # indices of successful environments
