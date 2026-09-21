#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 check: validation of the grasp_detector package (doc/stage2.md sec.10).

Part A - inference mode (GraspDetector, sec.10.1), seven unit cases:
    1. gripper fully open, away from the object       -> False
    2. only the left finger touching                   -> False
    3. bilateral light touch (force below threshold)   -> False
    4. firm bilateral contact, closing phase
       detect(confirm_lift=False)                      -> True
    5. same state, object still on support,
       detect(confirm_lift=True)                       -> False
    6. after a 10 cm lift: contact + force hold,
       object off the support                          -> True (sustained)
    7. object slips off after one finger opens         -> False
  Plus a 5-frame ring-buffer debouncing check (sec.10.3).

Part B - training mode (DiffGraspLoss, sec.10.2; force-control design,
         stage2.md v1.2 -- PD targets are not differentiable in Genesis
         1.3.3 and measured contact forces are detached):
    - diffsim scene (SimOptions(requires_grad=True));
    - force-controlled rollout: control_dofs_force(f_base + ramp*delta_f)
      with delta_f a sceneless gs.Tensor leaf;
    - scene.backward(loss) -> assert delta_f.grad nonzero, lift component
      negligible, finger components negative (squeeze more);
    - finite-difference cross-check on one component (sign + magnitude);
    - a few gradient-descent steps reduce the loss.

Scene: ground plane + 0.1 kg cube (0.04 m) + assets/mini_gripper.urdf
(fixed base, lift prismatic joint + two prismatic fingers).
Dof layout [lift, left_finger, right_finger] is verified at runtime.

Usage:
    mamba activate diffphy
    python scripts/stage2_grasp_detector_check.py \
        [--backend cpu|cuda] [--precision 32|64] [--iters N] [--viewer]

Verified env: diffphy (Python 3.12, genesis 1.3.3).
"""

import argparse
import os
import sys
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Genesis 1.3.3 note: the zerocopy fast path in collider.get_contacts()
# fed torch.gather an int32 index ("Expected dtype int64 for index");
# this is patched in the installed genesis (upcast to int64), so no
# GS_ENABLE_ZEROCOPY workaround is needed here.

import genesis as gs  # noqa: E402

from grasp_detector import (  # noqa: E402
    DiffGraspLoss,
    GraspDetector,
    force_magnitude,
    make_force_parameter,
    ramp_force,
)

URDF_PATH = os.path.join(ROOT, "assets", "mini_gripper.urdf")

# ----------------------------- scene constants -----------------------------
DT = 0.01
CUBE_SIZE = 0.04
CUBE_MASS = 0.1
CUBE_POS = (0.5, 0.0, 0.0205)          # 0.5 mm clearance above the plane
ROBOT_POS = (0.5, 0.0, 0.10)           # base_link frame; grasp height

# Dof names and per-dof PD gains / delta bounds (keyed by joint name).
KP = {"lift_joint": 3000.0, "left_finger_joint": 800.0,
      "right_finger_joint": 800.0}
KV = {"lift_joint": 80.0, "left_finger_joint": 40.0,
      "right_finger_joint": 40.0}

# Finger travel: inner face at +-(0.035 - q_f); contact at q_f = 0.015.
FORCE_THRESHOLD = 1.5                  # N; ~= mass * g * 1.5 (appendix B)
Q_TOUCH = 0.0153                       # 0.3 mm nominal penetration (light)
Q_ONE = 0.0155                         # single-side light press
Q_FIRM = 0.018                         # 3 mm nominal penetration (firm)
LIFT = 0.10                            # 10 cm lift for confirmation

# Training-mode constants (sec.6.2 / sec.10.2; force-control design, v1.2).
# Dof layout is [lift_joint, left_finger_joint, right_finger_joint].
F_BASE = [0.0, 2.0, 2.0]               # base forces [lift, left, right] (N);
                                        # lift=0 rests on the lower stop at
                                        # grasp height, 2 N/finger gives a
                                        # ~0.4 mm nominal penetration
D_TOUCH = 0.025                        # finger-origin projection distance at
                                        # first touch: cube half (0.02) +
                                        # finger inner offset (0.005) (m)
TARGET_PEN = 0.001                     # PEN*: squeeze saturation level (m)
HORIZON = 15                           # forward rollout steps
RAMP_STEPS = 8                         # force ramp inside the rollout
LR = 500.0                             # gradient-descent learning rate (N)
FD_EPS = 0.02                          # finite-difference step (N)


class Reporter:
    def __init__(self):
        self.n_pass = self.n_fail = 0

    def check(self, name, cond, detail=""):
        tag = "PASS" if cond else "FAIL"
        if cond:
            self.n_pass += 1
        else:
            self.n_fail += 1
        print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
        return cond


def _np(x):
    """Genesis tensor -> numpy array (device-safe)."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_scene(requires_grad=False, show_viewer=False):
    """Plane + cube + mini gripper; returns (scene, robot, cube, plane, names)."""
    scene = gs.Scene(
        show_viewer=show_viewer,
        sim_options=gs.options.SimOptions(dt=DT, requires_grad=requires_grad),
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


def qvec(names, lift=0.0, left=0.0, right=0.0):
    """Dof-order target vector from named components (numpy)."""
    v = {"lift_joint": lift, "left_finger_joint": left,
         "right_finger_joint": right}
    return np.array([v[n] for n in names], float)


def run_ramp_hold(scene, robot, q_target, ramp_steps=60, hold_steps=60):
    """Ramp the PD target from the current dofs, then hold and settle."""
    q_target = np.asarray(q_target, float)
    q0 = _np(robot.get_dofs_position())
    for t in range(ramp_steps):
        a = (t + 1) / ramp_steps
        robot.control_dofs_position(q0 + a * (q_target - q0))
        scene.step()
    for _ in range(hold_steps):
        robot.control_dofs_position(q_target)
        scene.step()


# --------------------------------------------------------------------------
# Part A: inference mode (sec.10.1)
# --------------------------------------------------------------------------

def part_inference(rep, show_viewer=False):
    print("\n=== Part A: inference mode (GraspDetector, sec.10.1) ===")
    scene, robot, cube, plane, names = build_scene(
        requires_grad=False, show_viewer=show_viewer)

    link_names = [l.name for l in robot.links]
    print("Robot links:", link_names)
    print("Robot joints:", names, f"(n_dofs={robot.n_dofs})")
    rep.check("expected links present",
              {"base_link", "gripper_base", "left_finger_link",
               "right_finger_link"} <= set(link_names))
    rep.check("expected dof layout (3 prismatic joints)",
              set(names) == set(KP) and robot.n_dofs == 3)

    detector = GraspDetector(
        scene=scene, robot_entity=robot, object_entity=cube,
        left_finger_name="left_finger_link",
        right_finger_name="right_finger_link",
        support_entity=plane,
        force_threshold=FORCE_THRESHOLD,
    )

    # --- case 1: fully open, away from the object ---
    print("\n--- Case 1: gripper open, lifted away from the object ---")
    scene.reset()
    run_ramp_hold(scene, robot, qvec(names, lift=0.12),
                  ramp_steps=50, hold_steps=40)
    ok, info = detector.detect(confirm_lift=False, return_debug_info=True)
    rep.check("case1 detect -> False", not ok)
    rep.check("case1 both sides no contact",
              not info["touching_left"] and not info["touching_right"],
              f"L={info['force_left']:.3f}N R={info['force_right']:.3f}N")

    # --- case 2: only the left finger touching ---
    print("\n--- Case 2: only the left finger in contact ---")
    scene.reset()
    run_ramp_hold(scene, robot, qvec(names, left=Q_ONE),
                  ramp_steps=40, hold_steps=30)
    ok, info = detector.detect(confirm_lift=False, return_debug_info=True)
    rep.check("case2 detect -> False", not ok)
    rep.check("case2 single-side contact (left only)",
              info["touching_left"] and not info["touching_right"],
              f"L={info['force_left']:.3f}N R={info['force_right']:.3f}N")

    # --- case 3: bilateral light touch, force below threshold ---
    print("\n--- Case 3: bilateral light touch (force < threshold) ---")
    scene.reset()
    run_ramp_hold(scene, robot, qvec(names, left=Q_TOUCH, right=Q_TOUCH),
                  ramp_steps=60, hold_steps=60)
    ok, info = detector.detect(confirm_lift=False, return_debug_info=True)
    rep.check("case3 detect -> False", not ok)
    rep.check("case3 bilateral contact present",
              info["touching_left"] and info["touching_right"])
    rep.check("case3 forces below threshold",
              info["force_left"] < FORCE_THRESHOLD
              and info["force_right"] < FORCE_THRESHOLD,
              f"L={info['force_left']:.3f}N R={info['force_right']:.3f}N "
              f"thresh={FORCE_THRESHOLD}N")

    # --- cases 4-7: one continuous grasp / lift / slip-off sequence ---
    print("\n--- Case 4: firm bilateral contact, closing phase ---")
    scene.reset()
    run_ramp_hold(scene, robot, qvec(names, left=Q_FIRM, right=Q_FIRM),
                  ramp_steps=60, hold_steps=80)
    ok4, info4 = detector.detect(confirm_lift=False, return_debug_info=True)
    rep.check("case4 detect(confirm_lift=False) -> True", ok4,
              f"L={info4['force_left']:.3f}N R={info4['force_right']:.3f}N")
    rep.check("case4 force threshold passed", info4["force_threshold_passed"])

    print("\n--- Case 5: not lifted yet, confirm_lift=True ---")
    ok5, info5 = detector.detect(return_debug_info=True)
    rep.check("case5 detect(confirm_lift=True) -> False", not ok5)
    rep.check("case5 object still on support", info5["on_support"])

    print("\n--- Case 6: 10 cm lift, sustained grasp (ring buffer) ---")
    run_ramp_hold(scene, robot, qvec(names, lift=LIFT, left=Q_FIRM,
                                     right=Q_FIRM),
                  ramp_steps=100, hold_steps=80)
    buf = deque(maxlen=5)  # sec.10.3: 5-frame window suppresses jitter
    for _ in range(8):
        robot.control_dofs_position(qvec(names, lift=LIFT, left=Q_FIRM,
                                         right=Q_FIRM))
        scene.step()
        buf.append(detector.detect())
    rep.check("case6 sustained over 5-frame window", all(buf),
              f"window={list(buf)}")
    ok6, info6 = detector.detect(return_debug_info=True)
    rep.check("case6 detect -> True after lift", ok6,
              f"L={info6['force_left']:.3f}N R={info6['force_right']:.3f}N")
    rep.check("case6 object off the support", not info6["on_support"])
    z6 = float(cube.get_pos()[2])
    rep.check("case6 cube lifted above 5 cm", z6 > 0.05, f"z={z6:.3f}m")

    print("\n--- Case 7: object slips off (left finger opens) ---")
    run_ramp_hold(scene, robot, qvec(names, lift=LIFT, left=0.0,
                                     right=Q_FIRM),
                  ramp_steps=40, hold_steps=60)
    ok7, info7 = detector.detect(return_debug_info=True)
    z7 = float(cube.get_pos()[2])
    rep.check("case7 detect -> False", not ok7,
              f"touchL={info7['touching_left']} z={z7:.3f}m")
    rep.check("case7 left contact lost", not info7["touching_left"])
    rep.check("case7 cube dropped below 6 cm", z7 < 0.06, f"z={z7:.3f}m")


# --------------------------------------------------------------------------
# Part B: training mode (sec.10.2)
# --------------------------------------------------------------------------

def part_training(rep, iters=10):
    print("\n=== Part B: training mode (DiffGraspLoss, sec.10.2) ===")
    scene, robot, cube, _plane, names = build_scene(
        requires_grad=True, show_viewer=False)
    rep.check("scene.requires_grad is True", scene.requires_grad)

    diff = DiffGraspLoss(
        scene, robot, cube,
        "left_finger_link", "right_finger_link",
        contact_distance=D_TOUCH,
        target_penetration=TARGET_PEN,
        symmetry_weight=0.5,
    )

    # Sceneless gs.Tensor leaves: the taped force keeps the gs.Tensor type
    # (via __torch_function__), which control_dofs_force requires for
    # input-gradient replay; scene=None avoids re-entering scene._backward.
    f_base = gs.tensor(F_BASE)
    delta_f = make_force_parameter([0.0, 0.0, 0.0])
    i_left = names.index("left_finger_joint")

    def rollout():
        """One diffsim forward pass; returns the differentiable loss."""
        scene.reset()  # back to the registered initial state, replay forward
        for t in range(HORIZON):
            # Force ramp for a gentle close; stays a gs.Tensor on the tape.
            robot.control_dofs_force(ramp_force(f_base, delta_f, t,
                                                RAMP_STEPS))
            scene.step()
        return diff.compute_loss()

    # --- B1: loss is finite, positive and attached to the graph ---
    loss = rollout()
    rep.check("loss finite, positive, requires_grad",
              bool(torch.isfinite(loss)) and float(loss) > 0.0
              and loss.requires_grad,
              f"loss={float(loss):.6f}")

    # --- B2: scene.backward populates a nonzero gradient ---
    scene.backward(loss)
    grad = delta_f.grad
    rep.check("delta_f.grad is not None", grad is not None)
    gnorm = 0.0 if grad is None else float(grad.detach().abs().sum())
    rep.check("delta_f.grad nonzero (sec.10.2)", gnorm > 0.0,
              f"|grad|_1={gnorm:.3e}")

    # --- B3: gradient structure: lift inert, fingers want more squeeze ---
    if grad is not None and gnorm > 0.0:
        g = grad.detach()
        rep.check("lift component negligible",
                  float(g[0].abs()) < 1e-3 * float(g.abs().max()),
                  f"|g_lift|={float(g[0].abs()):.2e}")
        rep.check("finger components negative (squeeze more)",
                  float(g[1]) < 0.0 and float(g[2]) < 0.0,
                  f"g_left={float(g[1]):.3e} g_right={float(g[2]):.3e}")

    # --- B4: finite-difference cross-check (sign + magnitude) ---
    # Note: the Genesis 1.3.3 contact adjoint is approximate, so FD and
    # analytic gradients agree in sign and order of magnitude, not exactly.
    def loss_at(delta_vals):
        with torch.no_grad():
            delta_f.copy_(torch.tensor(delta_vals, dtype=delta_f.dtype))
        return float(rollout())

    with torch.no_grad():
        delta_f.copy_(torch.zeros_like(delta_f))
    fd = (loss_at([0.0, FD_EPS, 0.0]) - loss_at([0.0, -FD_EPS, 0.0])) \
        / (2 * FD_EPS)
    analytic = float(grad[i_left]) if grad is not None else 0.0
    ratio = fd / analytic if abs(analytic) > 0 else float("inf")
    rep.check("FD vs analytic: same sign, same order of magnitude",
              fd * analytic > 0.0 and 0.2 < ratio < 5.0,
              f"FD={fd:.3e} analytic={analytic:.3e} ratio={ratio:.2f}")

    # --- B5: gradient descent reduces the loss (sec.6.2 skeleton) ---
    with torch.no_grad():
        delta_f.copy_(torch.zeros_like(delta_f))
    losses = []
    for it in range(iters):
        delta_f.zero_grad()
        loss = rollout()
        # scene.backward: snapshot -> torch.autograd.backward -> sim
        # backward (input-grad replay) -> restore snapshot.
        scene.backward(loss)
        with torch.no_grad():
            delta_f -= LR * delta_f.grad
        losses.append(float(loss))
        print(f"  iter {it + 1}/{iters}: loss={losses[-1]:.6f} "
              f"delta_f={_np(delta_f).round(4).tolist()}")
    rep.check("gradient descent reduces the loss",
              losses[-1] < losses[0],
              f"{losses[0]:.6f} -> {losses[-1]:.6f}")
    rep.check("finger forces stay symmetric",
              abs(float(delta_f[1]) - float(delta_f[2])) < 0.5,
              f"left={float(delta_f[1]):.3f} right={float(delta_f[2]):.3f} N")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Stage 2 validation of the grasp_detector package")
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--precision", choices=["32", "64"], default="64",
                    help="64 recommended for contact-force gradients "
                         "(stage2.md sec.3)")
    ap.add_argument("--iters", type=int, default=10,
                    help="gradient-descent iterations in Part B")
    ap.add_argument("--viewer", action="store_true",
                    help="show the Part A scene in a viewer")
    args = ap.parse_args()

    gs.init(backend=getattr(gs, args.backend), precision=args.precision,
            seed=0)

    rep = Reporter()
    print("Stage 2 check - grasp_detector package (doc/stage2.md v1.2)")
    print(f"backend={args.backend}, precision={args.precision}, dt={DT}")

    # Quick sanity of the tensor utils (sec.5.2).
    fm = force_magnitude(torch.tensor([[0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]))
    rep.check("utils.force_magnitude (n,3)->(n,)",
              _np(fm).shape == (2,) and np.allclose(_np(fm), [3.0, 4.0]))

    part_inference(rep, show_viewer=args.viewer)
    part_training(rep, iters=args.iters)

    print(f"\n===== Summary: {rep.n_pass} passed, {rep.n_fail} failed =====")
    sys.exit(0 if rep.n_fail == 0 else 1)


if __name__ == "__main__":
    main()
