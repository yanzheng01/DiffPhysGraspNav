#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end training example for the grasp_detector package
(doc/stage2.md v1.2 sec.6.2, force-control design).

Two-stage optimization (sec.4.4):
  Stage 1 - pose guidance (PD position control, NOT differentiable):
            drive the fingers to a light pre-grasp touch. Zero contact
            gives zero gradient, so the approach phase runs open-loop.
            The reached state is registered as the new init state via
            ``scene.reset(pre_state)``.
  Stage 2 - force optimization (differentiable): optimize the finger
            forces with DiffGraspLoss over force-controlled rollouts.
            ``control_dofs_force`` is the only differentiable control
            entry in Genesis 1.3.3, and the taped force must be a
            sceneless gs.Tensor (see grasp_detector.diff_loss).
  Final    - inference validation: GraspDetector judges the optimized
            grasp after a PD-position lift (lift channel position-
            controlled, fingers kept under the optimized forces).

Usage:
    mamba activate diffphy
    python grasp_detector/examples/train_example.py \
        [--backend cpu|cuda] [--iters N] [--viewer]

Verified env: diffphy (Python 3.12, genesis 1.3.3).
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(PKG)
sys.path.insert(0, ROOT)

import genesis as gs  # noqa: E402

from grasp_detector import (  # noqa: E402
    DiffGraspLoss,
    GraspDetector,
    make_force_parameter,
    ramp_force,
)

URDF_PATH = os.path.join(ROOT, "assets", "mini_gripper.urdf")

DT = 0.01
CUBE_SIZE = 0.04
CUBE_MASS = 0.1
CUBE_POS = (0.5, 0.0, 0.0205)
ROBOT_POS = (0.5, 0.0, 0.10)

KP = {"lift_joint": 3000.0, "left_finger_joint": 800.0,
      "right_finger_joint": 800.0}
KV = {"lift_joint": 80.0, "left_finger_joint": 40.0,
      "right_finger_joint": 40.0}

Q_PRE = 0.0148        # stage-1 pre-grasp: fingers ~0.2 mm before touch
LIFT = 0.10           # final lift height (m)
F_BASE = [0.0, 2.0, 2.0]   # [lift, left, right] base forces (N)
D_TOUCH = 0.025       # finger-origin projection distance at touch (m)
TARGET_PEN = 0.001    # squeeze saturation PEN* (m)
HORIZON = 15          # stage-2 rollout steps
RAMP_STEPS = 8        # force ramp inside the rollout
LR = 500.0            # gradient-descent learning rate (N)


def build_scene(show_viewer=False):
    scene = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(dt=DT, requires_grad=True),
    )
    plane = scene.add_entity(
        gs.morphs.Plane(), material=gs.materials.Rigid(friction=2.0))
    robot = scene.add_entity(
        gs.morphs.URDF(file=URDF_PATH, pos=ROBOT_POS, fixed=True),
        material=gs.materials.Rigid(friction=1.0),
    )
    cube = scene.add_entity(
        gs.morphs.Box(pos=CUBE_POS, size=(CUBE_SIZE,) * 3),
        material=gs.materials.Rigid(friction=1.0),
    )
    scene.build()
    cube.set_mass(CUBE_MASS)  # 1.3.3: morphs.Box has no mass kwarg
    names = [j.name for j in robot.joints]
    robot.set_dofs_kp([KP[n] for n in names])
    robot.set_dofs_kv([KV[n] for n in names])
    return scene, robot, cube, plane, names


def main():
    ap = argparse.ArgumentParser(
        description="Two-stage grasp-force training example")
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--precision", choices=["32", "64"], default="64")
    ap.add_argument("--iters", type=int, default=20,
                    help="stage-2 gradient-descent iterations")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    gs.init(backend=getattr(gs, args.backend), precision=args.precision,
            seed=0)
    scene, robot, cube, plane, names = build_scene(args.viewer)
    i_lift = names.index("lift_joint")
    finger_idx = [names.index("left_finger_joint"),
                  names.index("right_finger_joint")]
    print(f"Training example - dof layout {names} "
          f"(backend={args.backend}, precision={args.precision})")

    # ------------------------------------------------------------------
    # Stage 1: pose guidance -- PD position control to a light pre-grasp
    # (position targets are replayed for primal correctness but carry no
    # gradient path, so this stage is run open-loop).
    # ------------------------------------------------------------------
    print("\n--- Stage 1: pose guidance (PD position control) ---")
    q0 = np.zeros(len(names))
    q_pre = np.zeros(len(names))
    q_pre[finger_idx[0]] = Q_PRE
    q_pre[finger_idx[1]] = Q_PRE
    for t in range(60):
        a = (t + 1) / 60.0
        robot.control_dofs_position(q0 + a * (q_pre - q0))
        scene.step()
    for _ in range(40):
        robot.control_dofs_position(q_pre)
        scene.step()
    pre_state = scene.get_state()
    q_now = robot.get_dofs_position().tolist()
    print(f"pre-grasp dofs: {np.round(q_now, 5).tolist()} "
          f"(target {q_pre.tolist()})")

    # Register the pre-grasp state as the rollout init: stage-2 rollouts
    # start from here (scene.reset() with no argument re-arms forward
    # stepping after each scene.backward).
    scene.reset(pre_state)

    # ------------------------------------------------------------------
    # Stage 2: force optimization with DiffGraspLoss (differentiable).
    # ------------------------------------------------------------------
    print("\n--- Stage 2: force optimization (DiffGraspLoss) ---")
    diff = DiffGraspLoss(
        scene, robot, cube,
        "left_finger_link", "right_finger_link",
        contact_distance=D_TOUCH,
        target_penetration=TARGET_PEN,
        symmetry_weight=0.5,
    )
    f_base = gs.tensor(F_BASE)
    delta_f = make_force_parameter([0.0, 0.0, 0.0])

    def rollout():
        scene.reset()  # rewinds to the registered pre-grasp state
        for t in range(HORIZON):
            robot.control_dofs_force(ramp_force(f_base, delta_f, t,
                                                RAMP_STEPS))
            scene.step()
        return diff.compute_loss()

    for it in range(args.iters):
        delta_f.zero_grad()
        loss = rollout()
        scene.backward(loss)  # snapshot -> autograd -> sim backward
        with torch.no_grad():
            delta_f -= LR * delta_f.grad
        print(f"  iter {it + 1}/{args.iters}: loss={float(loss):.6f} "
              f"delta_f={np.round(delta_f.tolist(), 4).tolist()}")
    print(f"optimized forces: lift={F_BASE[0]:.2f} N, "
          f"left={F_BASE[1] + float(delta_f[1]):.3f} N, "
          f"right={F_BASE[2] + float(delta_f[2]):.3f} N")

    # ------------------------------------------------------------------
    # Final: inference validation -- lift with PD position control on the
    # lift channel while the fingers keep the optimized forces, then let
    # GraspDetector judge the grasp.
    # ------------------------------------------------------------------
    print("\n--- Final: inference validation (GraspDetector) ---")
    detector = GraspDetector(
        scene=scene, robot_entity=robot, object_entity=cube,
        left_finger_name="left_finger_link",
        right_finger_name="right_finger_link",
        support_entity=plane,
        force_threshold=1.5,
    )
    scene.reset(pre_state)
    f_opt = np.array([F_BASE[1] + float(delta_f[1]),
                      F_BASE[2] + float(delta_f[2])])
    q_lift = np.zeros(len(names))
    q_lift[i_lift] = LIFT
    q_start = robot.get_dofs_position().tolist()
    q_start = np.array(q_start)
    for t in range(100):
        a = min(1.0, (t + 1) / 60.0)
        robot.control_dofs_position(
            np.array([q_start[i_lift] + a * (LIFT - q_start[i_lift])]),
            dofs_idx_local=[i_lift])
        robot.control_dofs_force(f_opt, dofs_idx_local=finger_idx)
        scene.step()
    for _ in range(40):
        robot.control_dofs_position(np.array([LIFT]), dofs_idx_local=[i_lift])
        robot.control_dofs_force(f_opt, dofs_idx_local=finger_idx)
        scene.step()

    ok, info = detector.detect(confirm_lift=True, return_debug_info=True)
    z = float(cube.get_pos()[2])
    print(f"detect(confirm_lift=True) -> {ok}")
    print(f"  contact L/R: {info['touching_left']}/{info['touching_right']}"
          f"  force L/R: {info['force_left']:.3f}/{info['force_right']:.3f} N"
          f"  on_support: {info['on_support']}  cube z: {z:.3f} m")
    print("\nRESULT:", "SUCCESS - optimized grasp holds through the lift"
          if ok else "FAILED - grasp did not survive the lift")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
