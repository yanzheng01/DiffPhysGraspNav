#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 demo: analytic grasp base-pose computation + probabilistic modeling.

Implements the pipeline defined in stage1.md for technical validation:
  1. Inputs: T_WO (object pose in planning frame {W}), p_c, d, a_hat,
     optional contact pair (p1, p2) given in {O}.
  2. Frame unification: transform p_c / a / (p1, p2) into {W}.
  3. Position: P_base = p_c^W - d * a^W.
  4. Orientation: Z_e = a^W; X_e via contact-pair projection (priority)
     or reference-vector Gram-Schmidt fallback (anti-singularity).
  5. T_base assembly (4x4, expressed in {W}).
  6. Disturbance model: 5-dim small Gaussian x Roll distribution
     (uniform in fallback mode / small Gaussian in prior mode).
  7. Sampling in the right-perturbation tangent space: T = T_base @ exp(xi^).
  8. Light validity pre-check: approach-path vs AABB obstacles.
  9. Quadrants backend (Genesis differentiable kernels, optional): the batch
     perturbation runs as a parallel f64 kernel on CPU/CUDA, and a
     stage-2-style pose-deviation loss is differentiated with qd.ad.Tape
     (reverse mode). Both are cross-validated against the numpy reference
     (central finite differences for the gradient).
 10. Pose-distribution visualization: TCP cloud colored by |roll|, approach
     arrows and finger-pair clouds against the target object (proxy box);
     2x2 figure (3D fallback/prior + XY/XZ projections).

Conventions (per stage1.md):
  - All computation in the planning frame {W}.
  - Tangent vector order: xi = [v (3), omega (3)]  (translation first).
    NOTE: Sophus/Pinocchio use (omega, v) — reorder when interfacing.
  - Right (body-frame) perturbation; Roll = omega_z (about Z_e).
  - Units: meter / radian.
  - Quadrants kernels: single top-level parallel loop per kernel; nested
    runtime loops are unsupported under reverse-mode AD, so the loss
    accumulates atomically. Locals are defined before `if` and overwritten
    inside branches (AST does not merge branch-only definitions).

Usage:
  mamba activate diffphy
  python scripts/stage1_demo.py [--n-samples N] [--seed S] [--no-plot]
                                [--arch {auto,cpu,cuda}] [--no-qd]

Verified env: diffphy (Python 3.12, numpy 2.5, matplotlib 3.11,
quadrants 1.3 with CPU and CUDA backends).
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

try:
    import quadrants as qd
    HAVE_QD = True
except Exception:
    qd = None
    HAVE_QD = False

EPS = 1e-12
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# SO(3) / SE(3) math (tangent order: xi = [v, omega], translation first)
# --------------------------------------------------------------------------

def hat3(w):
    """3x3 skew-symmetric matrix of w."""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def so3_exp(w):
    """Rotation matrix from axis-angle vector w."""
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        # Taylor: I + hat(w) + 0.5 hat(w)^2
        W = hat3(w)
        return np.eye(3) + W + 0.5 * W @ W
    W = hat3(w)
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / theta**2
    return np.eye(3) + A * W + B * (W @ W)


def so3_log(R):
    """Axis-angle vector of R (principal branch, theta in [0, pi])."""
    c = np.clip(0.5 * (np.trace(R) - 1.0), -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if theta < np.pi - 1e-7:
        return theta / (2.0 * np.sin(theta)) * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    # Near pi: axis from the diagonal (largest element for stability)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    axis = np.zeros(3)
    axis[i] = np.sqrt(max(0.0, R[i, i] + 1.0 - R[j, j] - R[k, k])) * 0.5
    axis[j] = (R[j, i] + R[i, j]) / (4.0 * axis[i] + EPS)
    axis[k] = (R[k, i] + R[i, k]) / (4.0 * axis[i] + EPS)
    if np.array_equal(so3_exp(theta * axis), R) or True:
        return theta * axis


def se3_exp(xi):
    """SE(3) exponential of xi = [v, omega] (stage1.md 4.1 convention).

    Returns 4x4 T = [[R, t], [0, 1]] with t = V(omega) v (left Jacobian
    couples translation and rotation — see stage1.md 4.2 coupling note).
    """
    v, w = xi[:3], xi[3:]
    theta = np.linalg.norm(w)
    R = so3_exp(w)
    if theta < 1e-10:
        W = hat3(w)
        t = v + 0.5 * W @ v + (1.0 / 6.0) * (W @ W) @ v
    else:
        W = hat3(w)
        B = (1.0 - np.cos(theta)) / theta**2
        C = (theta - np.sin(theta)) / theta**3
        V = np.eye(3) + B * W + C * (W @ W)
        t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_log(T):
    """Principal-branch log: xi = [v, omega] with log(T) = xi^ (||omega|| < pi)."""
    R, t = T[:3, :3], T[:3, 3]
    w = so3_log(R)
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        v = t
    else:
        W = hat3(w)
        c, s = np.cos(theta), np.sin(theta)
        V_inv = (np.eye(3) - 0.5 * W +
                 (1.0 / theta**2 - (1.0 + c) / (2.0 * theta * s)) * (W @ W))
        v = V_inv @ t
    return np.concatenate([v, w])


# --------------------------------------------------------------------------
# Stage 1 core (stage1.md sections 2-4)
# --------------------------------------------------------------------------

def transform_to_W(T_WO, p):
    """Transform a point p (in {O}) into {W}."""
    return T_WO[:3, :3] @ p + T_WO[:3, 3]


def build_rotation(a_W, p1_W=None, p2_W=None, eps_sing=1e-6, eps_prior=1e-3):
    """Build R_base = [X_e | Y_e | Z_e] (columns) in {W} (stage1.md 3.2).

    Priority: contact-pair geometric prior (p1 -> p2 defines +X_e),
    fallback: reference-vector Gram-Schmidt with anti-singularity switch.
    Returns (R, mode, info).
    """
    Z = a_W / np.linalg.norm(a_W)
    mode, info = "fallback", {}
    X = None
    if p1_W is not None and p2_W is not None:
        dp = np.asarray(p2_W) - np.asarray(p1_W)
        proj = dp - (dp @ Z) * Z
        nrm = np.linalg.norm(proj)
        if nrm >= eps_prior:
            X = proj / nrm
            mode = "prior"
            info["prior_residual"] = float(np.linalg.norm(dp) - nrm)
        else:
            info["prior_degenerate"] = True
    if X is None:
        r = np.array([0.0, 0.0, 1.0])
        if abs(r @ Z) > 1.0 - eps_sing:
            r = np.array([1.0, 0.0, 0.0])
            info["ref_switched"] = True
        X = r - (r @ Z) * Z
        X = X / np.linalg.norm(X)
    Y = np.cross(Z, X)
    R = np.column_stack([X, Y, Z])
    return R, mode, info


def build_base_pose(T_WO, p_c, d, a, p1=None, p2=None):
    """Full stage-1 analytic pipeline (stage1.md sections 2-3).

    Returns (T_base, info). T_base is the EEF pose T^W_E.
    """
    p_c_W = transform_to_W(T_WO, np.asarray(p_c, float))
    a_W = T_WO[:3, :3] @ (np.asarray(a, float) / np.linalg.norm(a))
    p1_W = transform_to_W(T_WO, np.asarray(p1, float)) if p1 is not None else None
    p2_W = transform_to_W(T_WO, np.asarray(p2, float)) if p2 is not None else None
    P_base = p_c_W - d * a_W
    R, mode, rot_info = build_rotation(a_W, p1_W, p2_W)
    T_base = np.eye(4)
    T_base[:3, :3] = R
    T_base[:3, 3] = P_base
    info = dict(p_c_W=p_c_W, a_W=a_W, mode=mode, rot_info=rot_info)
    return T_base, info


def sample_xi(n, sigma_p, sigma_ab, roll_mode, sigma_gamma=None,
              symmetric=False, rng=None):
    """Draw tangent-space disturbances xi = [v, omega] (stage1.md 4.2).

    roll_mode: 'uniform' (fallback) or 'gaussian' (contact-prior).
    symmetric: gripper 180-deg symmetry -> Roll in [0, pi).
    """
    rng = rng or np.random.default_rng()
    xi = np.zeros((n, 6))
    xi[:, 0:3] = rng.normal(0.0, sigma_p, size=(n, 3))
    xi[:, 3] = rng.normal(0.0, sigma_ab, size=n)
    xi[:, 4] = rng.normal(0.0, sigma_ab, size=n)
    if roll_mode == "uniform":
        lo = 0.0 if symmetric else -np.pi
        xi[:, 5] = rng.uniform(lo, np.pi, size=n)
    elif roll_mode == "gaussian":
        if sigma_gamma is None:
            raise ValueError("sigma_gamma required for gaussian roll mode")
        xi[:, 5] = rng.normal(0.0, sigma_gamma, size=n)
    else:
        raise ValueError(f"unknown roll_mode: {roll_mode}")
    return xi


def perturb(T_base, xi):
    """Right perturbation T = T_base @ exp(xi^) (vectorized over samples)."""
    Ts = np.zeros((len(xi), 4, 4))
    for i in range(len(xi)):
        Ts[i] = T_base @ se3_exp(xi[i])
    return Ts


# --------------------------------------------------------------------------
# Validity pre-check (stage1.md 3.4)
# --------------------------------------------------------------------------

def seg_aabb_intersect(p0, p1, box):
    """Segment p0->p1 vs AABB box=(lo, hi). Slab method, t in [0, 1]."""
    lo, hi = box
    d = np.asarray(p1) - np.asarray(p0)
    tmin, tmax = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
        else:
            t1, t2 = (lo[i] - p0[i]) / d[i], (hi[i] - p0[i]) / d[i]
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def precheck(T_base, p_c_W, env_boxes=(), ws_radius=None):
    """Light validity pre-check. Returns (ok, reason)."""
    P_base = T_base[:3, 3]
    for k, box in enumerate(env_boxes):
        if seg_aabb_intersect(P_base, p_c_W, box):
            return False, f"approach path collides with obstacle #{k}"
    if ws_radius is not None and np.linalg.norm(P_base) > ws_radius:
        return False, f"P_base outside workspace (|P|={np.linalg.norm(P_base):.3f} m)"
    return True, "ok"


# --------------------------------------------------------------------------
# Quadrants differentiable backend (Genesis kernels, optional)
# --------------------------------------------------------------------------

if HAVE_QD:

    def init_quadrants(pref="auto"):
        """Initialize the quadrants runtime in f64. Returns the arch name."""
        if pref != "cpu":
            try:
                qd.init(arch=qd.cuda, default_fp=qd.f64)
                return "cuda"
            except Exception as exc:
                if pref == "cuda":
                    raise
                print(f"    [quadrants] CUDA unavailable ({type(exc).__name__}); "
                      f"falling back to CPU")
        qd.init(arch=qd.cpu, default_fp=qd.f64)
        return "cpu"

    @qd.kernel
    def _perturb_kernel(n: qd.i32, xi: qd.types.NDArray[qd.f64, 2],
                        Rb: qd.types.NDArray[qd.f64, 2],
                        pb: qd.types.NDArray[qd.f64, 1],
                        Ts: qd.types.NDArray[qd.f64, 3]) -> None:
        """Ts[i, :3, :4] <- top 3 rows of T_base @ exp(xi_i^); row 4 = [0,0,0,1].

        Unrolled scalar Rodrigues with a Taylor branch below th = 1e-8:
          R(w)  = cos(th) I + A hat(w) + B w w^T
          V(w)v = (1 - C th^2) v + B (w x v) + C (w.v) w
          A = sin/th, B = (1 - cos)/th^2, C = (th - sin)/th^3
        """
        for i in range(n):
            vx = xi[i, 0]; vy = xi[i, 1]; vz = xi[i, 2]
            wx = xi[i, 3]; wy = xi[i, 4]; wz = xi[i, 5]
            th2 = wx * wx + wy * wy + wz * wz
            th = qd.sqrt(th2)
            cs = 1.0; A = 1.0; B = 0.5; C = 1.0 / 6.0
            if th >= 1e-8:
                cs = qd.cos(th)
                A = qd.sin(th) / th
                B = (1.0 - cs) / th2
                C = (th - qd.sin(th)) / (th2 * th)
            # R_xi = Rodrigues(omega)
            r00 = cs + B * wx * wx
            r01 = -A * wz + B * wx * wy
            r02 = A * wy + B * wx * wz
            r10 = A * wz + B * wy * wx
            r11 = cs + B * wy * wy
            r12 = -A * wx + B * wy * wz
            r20 = -A * wy + B * wz * wx
            r21 = A * wx + B * wz * wy
            r22 = cs + B * wz * wz
            # V(omega) v
            wv = wx * vx + wy * vy + wz * vz
            kf = 1.0 - C * th2
            tx = kf * vx + B * (wy * vz - wz * vy) + C * wv * wx
            ty = kf * vy + B * (wz * vx - wx * vz) + C * wv * wy
            tz = kf * vz + B * (wx * vy - wy * vx) + C * wv * wz
            # Compose with T_base: R_i = Rb @ R_xi, t_i = Rb @ t_xi + pb
            b00 = Rb[0, 0]; b01 = Rb[0, 1]; b02 = Rb[0, 2]
            b10 = Rb[1, 0]; b11 = Rb[1, 1]; b12 = Rb[1, 2]
            b20 = Rb[2, 0]; b21 = Rb[2, 1]; b22 = Rb[2, 2]
            Ts[i, 0, 0] = b00 * r00 + b01 * r10 + b02 * r20
            Ts[i, 0, 1] = b00 * r01 + b01 * r11 + b02 * r21
            Ts[i, 0, 2] = b00 * r02 + b01 * r12 + b02 * r22
            Ts[i, 1, 0] = b10 * r00 + b11 * r10 + b12 * r20
            Ts[i, 1, 1] = b10 * r01 + b11 * r11 + b12 * r21
            Ts[i, 1, 2] = b10 * r02 + b11 * r12 + b12 * r22
            Ts[i, 2, 0] = b20 * r00 + b21 * r10 + b22 * r20
            Ts[i, 2, 1] = b20 * r01 + b21 * r11 + b22 * r21
            Ts[i, 2, 2] = b20 * r02 + b21 * r12 + b22 * r22
            Ts[i, 0, 3] = b00 * tx + b01 * ty + b02 * tz + pb[0]
            Ts[i, 1, 3] = b10 * tx + b11 * ty + b12 * tz + pb[1]
            Ts[i, 2, 3] = b20 * tx + b21 * ty + b22 * tz + pb[2]

    @qd.kernel
    def _pose_loss_kernel(n: qd.i32, scale: qd.f64,
                          xi: qd.types.NDArray[qd.f64, 2],
                          Rb: qd.types.NDArray[qd.f64, 2],
                          pb: qd.types.NDArray[qd.f64, 1],
                          cx: qd.f64, cy: qd.f64, cz: qd.f64, w_rot: qd.f64,
                          loss: qd.types.NDArray[qd.f64, 1]) -> None:
        """Accumulate L = mean_i [ ||t_i - c||^2 + w_rot * ||R_i - Rb||_F^2 ].

        t_i, R_i from T_base @ exp(xi_i^). The rotation term uses the
        Frobenius invariance ||Rb R_xi - Rb||_F^2 = ||R_xi - I||_F^2
        = 6 - 2 tr(R_xi) with tr(R_xi) = 3 cos(th) + B th^2.
        Reverse-mode AD constraints (quadrants): a single top-level parallel
        loop only (nested runtime loops are rejected in backwards mode), and
        the loss accumulates via atomic `+=`.
        """
        for i in range(n):
            vx = xi[i, 0]; vy = xi[i, 1]; vz = xi[i, 2]
            wx = xi[i, 3]; wy = xi[i, 4]; wz = xi[i, 5]
            th2 = wx * wx + wy * wy + wz * wz
            th = qd.sqrt(th2)
            cs = 1.0; B = 0.5; C = 1.0 / 6.0
            if th >= 1e-8:
                cs = qd.cos(th)
                B = (1.0 - cs) / th2
                C = (th - qd.sin(th)) / (th2 * th)
            tr = 3.0 * cs + B * th2  # tr(R_xi)
            wv = wx * vx + wy * vy + wz * vz
            kf = 1.0 - C * th2
            tx = kf * vx + B * (wy * vz - wz * vy) + C * wv * wx
            ty = kf * vy + B * (wz * vx - wx * vz) + C * wv * wy
            tz = kf * vz + B * (wx * vy - wy * vx) + C * wv * wz
            b00 = Rb[0, 0]; b01 = Rb[0, 1]; b02 = Rb[0, 2]
            b10 = Rb[1, 0]; b11 = Rb[1, 1]; b12 = Rb[1, 2]
            b20 = Rb[2, 0]; b21 = Rb[2, 1]; b22 = Rb[2, 2]
            dx = (b00 * tx + b01 * ty + b02 * tz + pb[0]) - cx
            dy = (b10 * tx + b11 * ty + b12 * tz + pb[1]) - cy
            dz = (b20 * tx + b21 * ty + b22 * tz + pb[2]) - cz
            loss[0] += (dx * dx + dy * dy + dz * dz
                        + w_rot * (6.0 - 2.0 * tr)) * scale

    class QdBackend:
        """Quadrants backend for the stage-1 batch sampling pipeline.

        Holds T_base on device; perturb() runs the batched se3_exp kernel,
        pose_loss_and_grad() adds a Tape-based reverse-mode gradient
        (stage-2 differentiable readiness).
        """

        def __init__(self, T_base):
            self.Rb = qd.ndarray(qd.f64, (3, 3))
            self.Rb.from_numpy(np.ascontiguousarray(T_base[:3, :3], dtype=np.float64))
            self.pb = qd.ndarray(qd.f64, (3,))
            self.pb.from_numpy(np.ascontiguousarray(T_base[:3, 3], dtype=np.float64))

        def perturb(self, xi):
            """T_i = T_base @ exp(xi_i^) for all samples (parallel kernel)."""
            n = len(xi)
            xi_q = qd.ndarray(qd.f64, (n, 6))
            xi_q.from_numpy(np.ascontiguousarray(xi, dtype=np.float64))
            Ts_q = qd.ndarray(qd.f64, (n, 3, 4))
            _perturb_kernel(n, xi_q, self.Rb, self.pb, Ts_q)
            Ts = np.zeros((n, 4, 4))
            Ts[:, :3, :4] = Ts_q.to_numpy()
            Ts[:, 3, 3] = 1.0
            return Ts

        def pose_loss_and_grad(self, xi, c, w_rot):
            """Return (L, dL/dxi) with L = mean_i [ ||t_i - c||^2
            + w_rot * ||R_i - R_base||_F^2 ] (reverse-mode autodiff)."""
            n = len(xi)
            xi_q = qd.ndarray(qd.f64, (n, 6), needs_grad=True)
            xi_q.from_numpy(np.ascontiguousarray(xi, dtype=np.float64))
            loss = qd.ndarray(qd.f64, (1,), needs_grad=True)
            with qd.ad.Tape(loss=loss):
                _pose_loss_kernel(n, 1.0 / n, xi_q, self.Rb, self.pb,
                                  float(c[0]), float(c[1]), float(c[2]),
                                  float(w_rot), loss)
            return float(loss.to_numpy()[0]), xi_q.grad.to_numpy()


def pose_loss_np(T_base, xi, c, w_rot):
    """numpy reference of the pose-deviation loss (for FD cross-check)."""
    Ts = perturb(T_base, xi)
    d = Ts[:, :3, 3] - c
    Rm = Ts[:, :3, :3] - T_base[:3, :3]
    rot = np.einsum("nij,nij->n", Rm, Rm)
    return float(np.mean(np.sum(d * d, axis=1) + w_rot * rot))


# --------------------------------------------------------------------------
# Check helper
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Demo scenarios
# --------------------------------------------------------------------------

def demo_scene():
    """Common scene: object standing on a table (all in {W})."""
    # Object pose in {W}: on the table, yawed 35 deg, slightly tilted
    T_WO = np.eye(4)
    T_WO[:3, :3] = so3_exp(np.array([0.0, 0.0, np.deg2rad(35.0)])) @ \
        so3_exp(np.array([np.deg2rad(8.0), 0.0, 0.0]))
    T_WO[:3, 3] = [0.45, 0.05, 0.15]
    # Table (AABB) and a stray obstacle near the approach corridor
    table = (np.array([0.20, -0.30, -0.10]), np.array([0.70, 0.30, 0.05]))
    obstacle = (np.array([0.28, -0.10, 0.00]), np.array([0.42, 0.10, 0.30]))
    return T_WO, table, obstacle


def demo_exp_log_roundtrip(rep, rng):
    """Verify se3_log(T_base^-1 @ T_base @ exp(xi^)) == xi (||omega|| < pi)."""
    print("\n=== Demo 1: exp/log round-trip on SE(3) ===")
    T_base = np.eye(4)
    T_base[:3, :3] = so3_exp(np.array([0.1, -0.2, 0.3]))
    T_base[:3, 3] = [0.4, 0.1, 0.3]
    n, max_err = 200, 0.0
    for _ in range(n):
        xi = np.zeros(6)
        xi[:3] = rng.normal(0.0, 0.05, 3)
        theta = rng.uniform(0.0, np.pi - 0.05)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        xi[3:] = theta * axis
        T = T_base @ se3_exp(xi)
        xi_rec = se3_log(np.linalg.inv(T_base) @ T)
        max_err = max(max_err, float(np.abs(xi_rec - xi).max()))
    rep.check("exp/log round-trip", max_err < 1e-8, f"max |xi_log - xi| = {max_err:.2e}")


def demo_base_pose(rep):
    """Nominal tilted approach, top-down singularity, contact prior."""
    print("\n=== Demo 2: base pose construction (stage1.md sec.3) ===")
    T_WO, _, _ = demo_scene()

    # (a) Tilted approach, no contact pair -> default reference vector
    T_a, info_a = build_base_pose(T_WO, p_c=[0.0, 0.0, 0.05], d=0.15,
                                  a=[0.15, 0.05, -1.0])
    R = T_a[:3, :3]
    rep.check("(a) tilted: R orthonormal",
              np.allclose(R.T @ R, np.eye(3), atol=1e-10),
              f"mode={info_a['mode']}, ref_switched={info_a['rot_info'].get('ref_switched', False)}")
    rep.check("(a) tilted: det(R)=+1", abs(np.linalg.det(R) - 1.0) < 1e-10)
    rep.check("(a) tilted: Z_e == a^W",
              np.allclose(R[:, 2], info_a["a_W"], atol=1e-12))
    rep.check("(a) tilted: right-handed Y=ZxX",
              np.allclose(R[:, 1], np.cross(R[:, 2], R[:, 0]), atol=1e-12))
    P_exp = info_a["p_c_W"] - 0.15 * info_a["a_W"]
    rep.check("(a) tilted: P_base = p_c^W - d*a^W",
              np.allclose(T_a[:3, 3], P_exp, atol=1e-12))

    # (b) Exact top-down approach in {W} -> |r.Z| = 1 triggers reference switch
    a_world_down = T_WO[:3, :3].T @ np.array([0.0, 0.0, -1.0])  # a^W = [0,0,-1]
    T_b, info_b = build_base_pose(T_WO, p_c=[0.0, 0.0, 0.05], d=0.15,
                                  a=a_world_down)
    R_b = T_b[:3, :3]
    ok = (info_b["rot_info"].get("ref_switched", False)
          and np.allclose(R_b[:, 2], [0, 0, -1], atol=1e-12)
          and np.allclose(R_b[:, 0], [1, 0, 0], atol=1e-6)
          and np.allclose(R_b.T @ R_b, np.eye(3), atol=1e-10))
    rep.check("(b) top-down singularity: fallback axis switch, valid R", ok,
              f"X_e={np.round(R_b[:, 0], 6)}")

    # (c) Contact pair -> geometric prior for X_e (projection of p2-p1)
    p1_O, p2_O = [-0.025, 0.01, 0.05], [0.025, -0.01, 0.05]
    T_c, info_c = build_base_pose(T_WO, p_c=[0.0, 0.0, 0.05], d=0.15,
                                  a=[0.15, 0.05, -1.0], p1=p1_O, p2=p2_O)
    dp_W = (transform_to_W(T_WO, np.array(p2_O))
            - transform_to_W(T_WO, np.array(p1_O)))
    Z_c = info_c["a_W"]
    proj = dp_W - (dp_W @ Z_c) * Z_c  # stage1.md 3.2 projection
    X_e = T_c[:3, 0]
    rep.check("(c) contact prior: mode == prior", info_c["mode"] == "prior")
    rep.check("(c) contact prior: X_e == normalized projection of p2-p1",
              np.allclose(X_e, proj / np.linalg.norm(proj), atol=1e-10),
              f"cos(X_e, dp) = {X_e @ dp_W / np.linalg.norm(dp_W):.6f}")

    # (d) Degenerate contact pair (collinear with approach) -> fallback
    T_d, info_d = build_base_pose(T_WO, p_c=[0.0, 0.0, 0.05], d=0.15,
                                  a=[0.15, 0.05, -1.0],
                                  p1=[0.0, 0.0, 0.0], p2=[0.03, 0.01, -0.2])
    rep.check("(d) degenerate pair: falls back to reference vector",
              info_d["mode"] == "fallback"
              and info_d["rot_info"].get("prior_degenerate", False))

    # (e) p_c consistency check (stage1.md 3.2)
    mid_W = 0.5 * (transform_to_W(T_WO, np.array(p1_O))
                   + transform_to_W(T_WO, np.array(p2_O)))
    dist = float(np.linalg.norm(info_c["p_c_W"] - mid_W))
    print(f"    (e) p_c vs contact-pair midpoint distance: {dist*1000:.2f} mm "
          f"(should be within sensing error)")


def demo_sampling(rep, rng, n, make_plot):
    """Sampling in tangent space; validate uniform/gaussian roll and coupling."""
    print("\n=== Demo 3: disturbance sampling (stage1.md sec.4) ===")
    T_WO, _, _ = demo_scene()
    T_base, info = build_base_pose(T_WO, p_c=[0.0, 0.0, 0.05], d=0.15,
                                   a=[0.15, 0.05, -1.0])
    sigma_p, sigma_ab, sigma_gamma = 0.002, 0.02, 0.08

    # --- Fallback mode: uniform roll on [-pi, pi) ---
    xi_fb = sample_xi(n, sigma_p, sigma_ab, "uniform", rng=rng)
    Ts_fb = perturb(T_base, xi_fb)
    Rs = Ts_fb[:, :3, :3]
    ortho_err = np.abs(np.einsum("nij,nik->njk", Rs, Rs) - np.eye(3)).max()
    det_err = np.abs(np.linalg.det(Rs) - 1.0).max()
    rep.check("fallback: sampled rotations valid (SO(3))",
              ortho_err < 1e-9 and det_err < 1e-9,
              f"ortho err {ortho_err:.1e}, det err {det_err:.1e}")

    # Recover xi via log over ALL samples (end-to-end check)
    # Two-tier tolerance: arccos loses ~8 digits near the cut locus
    # (||omega|| -> pi), so a wider band is checked separately.
    T_inv = np.linalg.inv(T_base)
    xi_rec = np.array([se3_log(T_inv @ T) for T in Ts_fb])
    w_norm = np.linalg.norm(xi_fb[:, 3:], axis=1)
    err_far = np.abs(xi_rec[w_norm < np.pi - 1e-3] - xi_fb[w_norm < np.pi - 1e-3]).max()
    err_all = np.abs(xi_rec[w_norm < np.pi] - xi_fb[w_norm < np.pi]).max()
    rep.check("fallback: log recovers xi (||omega|| < pi - 1e-3)", err_far < 1e-8,
              f"max err {err_far:.2e}; near cut locus: {err_all:.2e} (arccos precision)")

    # Roll uniformity: 12-bin frequencies over all samples
    roll_fb = xi_rec[:, 5]
    hist, _ = np.histogram(roll_fb, bins=12, range=(-np.pi, np.pi))
    freq = hist / hist.sum()
    p_bin = 1.0 / 12.0
    tol = max(0.02, 6.0 * np.sqrt(p_bin * (1.0 - p_bin) / n))  # stat floor
    rep.check("fallback: Roll ~ Uniform[-pi, pi)",
              freq.max() - freq.min() < tol,
              f"bin freq in [{freq.min():.4f}, {freq.max():.4f}], tol {tol:.4f}")

    # Symmetric variant: Roll in [0, pi)
    xi_sym = sample_xi(n, sigma_p, sigma_ab, "uniform", symmetric=True, rng=rng)
    rep.check("fallback+180deg symmetry: Roll in [0, pi)",
              xi_sym[:, 5].min() >= 0.0 and xi_sym[:, 5].max() < np.pi)

    # --- Prior mode: small Gaussian roll ---
    xi_pr = sample_xi(n, sigma_p, sigma_ab, "gaussian", sigma_gamma=sigma_gamma, rng=rng)
    Ts_pr = perturb(T_base, xi_pr)
    roll_pr = np.array([se3_log(T_inv @ T)[5] for T in Ts_pr])
    std_ratio = roll_pr.std() / sigma_gamma
    rep.check("prior: Roll std matches sigma_gamma", abs(std_ratio - 1.0) < 0.05,
              f"std = {roll_pr.std():.4f} rad (target {sigma_gamma}), ratio {std_ratio:.3f}")

    # --- Translation-rotation coupling (stage1.md 4.2 note) ---
    # V(omega) = int_0^1 exp(s*omega^) ds is an average of rotations, so
    # ||V|| <= 1 always: lateral (X_e-Y_e plane) translation disturbance is
    # ROTATED by ~||omega||/2 and SHRUNK by sinc(||omega||/2) in [2/pi, 1].
    # For uniform roll the theoretical lateral std ratio is
    # sqrt(E[sinc^2]) ~= 0.88 (never amplified).
    def body_dev(Ts):
        d = np.array([T[:3, 3] - T_base[:3, 3] for T in Ts])
        return np.column_stack([d @ T_base[:3, 0], d @ T_base[:3, 1], d @ T_base[:3, 2]])

    sub = slice(None, None, max(1, n // 4000))
    dev_fb = body_dev(Ts_fb[sub])
    dev_pr = body_dev(Ts_pr[sub])
    std_fb, std_pr = dev_fb.std(axis=0), dev_pr.std(axis=0)
    print("    translation std per EEF axis (mm):")
    print(f"      fallback (uniform roll): X={std_fb[0]*1e3:.2f}  "
          f"Y={std_fb[1]*1e3:.2f}  Z={std_fb[2]*1e3:.2f}   (sigma_p = {sigma_p*1e3:.1f})")
    print(f"      prior    (gaussian roll): X={std_pr[0]*1e3:.2f}  "
          f"Y={std_pr[1]*1e3:.2f}  Z={std_pr[2]*1e3:.2f}")
    lat_ratio = std_fb[:2] / sigma_p
    rep.check("coupling: fallback lateral std shrunk (theory ~0.88, never >1)",
              0.80 < lat_ratio.min() and lat_ratio.max() < 0.96,
              f"ratio X={lat_ratio[0]:.3f}, Y={lat_ratio[1]:.3f}, isotropic "
              f"|X-Y| = {abs(lat_ratio[0]-lat_ratio[1]):.3f}")
    rep.check("coupling: prior mode stays ~sigma_p on all axes",
              np.all(np.abs(std_pr / sigma_p - 1.0) < 0.1))
    rep.check("coupling: Z-axis unaffected in both modes",
              abs(std_fb[2] / sigma_p - 1) < 0.05 and abs(std_pr[2] / sigma_p - 1) < 0.05)

    if make_plot and HAVE_PLT:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        axs[0, 0].hist(roll_fb, bins=48, range=(-np.pi, np.pi), color="tab:blue")
        axs[0, 0].set_title("Fallback mode: Roll histogram (uniform)")
        axs[0, 0].set_xlabel("Roll [rad]")
        axs[0, 1].hist(roll_pr, bins=48, color="tab:green")
        axs[0, 1].set_title(f"Prior mode: Roll histogram (sigma={sigma_gamma} rad)")
        axs[0, 1].set_xlabel("Roll [rad]")
        lat = np.linalg.norm(dev_fb[:, :2], axis=1)
        roll_sub = xi_fb[sub][:, 5]
        axs[1, 0].scatter(np.abs(roll_sub), lat * 1e3, s=2, alpha=0.3, color="tab:red")
        axs[1, 0].set_title("Coupling: |Roll| vs lateral translation error")
        axs[1, 0].set_xlabel("|Roll| [rad]")
        axs[1, 0].set_ylabel("lateral error [mm]")
        x = np.arange(3)
        axs[1, 1].bar(x - 0.15, std_fb * 1e3, width=0.3, label="fallback", color="tab:blue")
        axs[1, 1].bar(x + 0.15, std_pr * 1e3, width=0.3, label="prior", color="tab:green")
        axs[1, 1].axhline(sigma_p * 1e3, ls="--", c="k", lw=1, label="sigma_p")
        axs[1, 1].set_xticks(x, ["X_e", "Y_e", "Z_e"])
        axs[1, 1].set_ylabel("translation std [mm]")
        axs[1, 1].set_title("Translation spread per EEF axis")
        axs[1, 1].legend()
        for ax in axs.flat:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(HERE, "stage1_demo_output.png")
        fig.savefig(out, dpi=120)
        print(f"    plot saved: {out}")

    # Data consumed by the pose-distribution visualization (demo 6)
    return dict(T_base=T_base, info=info, d=0.15,
                Ts_fb=Ts_fb, xi_fb=xi_fb, Ts_pr=Ts_pr, xi_pr=xi_pr)


def demo_precheck(rep):
    """Validity pre-check: table sweep and obstacle corridor (stage1.md 3.4)."""
    print("\n=== Demo 4: validity pre-check (stage1.md sec.3.4) ===")
    T_WO, table, obstacle = demo_scene()
    env = [table, obstacle]

    # (a) Top-down grasp of object top: path stays above table -> pass
    T_a, info_a = build_base_pose(T_WO, p_c=[0.0, 0.0, 0.12], d=0.15,
                                  a=[0.0, 0.0, -1.0])
    ok_a, why_a = precheck(T_a, info_a["p_c_W"], env_boxes=env)
    rep.check("(a) top-down: approach path clear", ok_a, why_a)

    # (b) Horizontal grasp near table surface: retract path sweeps the table
    T_b, info_b = build_base_pose(T_WO, p_c=[0.0, 0.0, -0.11], d=0.15,
                                  a=[0.0, 1.0, 0.0])
    ok_b, why_b = precheck(T_b, info_b["p_c_W"], env_boxes=env)
    rep.check("(b) low horizontal: collision detected (rejected)", not ok_b, why_b)

    # (c) Workspace check
    T_c, info_c = build_base_pose(T_WO, p_c=[0.0, 0.0, 0.12], d=0.15,
                                  a=[0.0, 0.0, -1.0])
    ok_c, why_c = precheck(T_c, info_c["p_c_W"], env_boxes=env, ws_radius=0.20)
    rep.check("(c) workspace radius check triggers", not ok_c, why_c)


def demo_quadrants(rep, rng, n, arch_pref):
    """Quadrants-accelerated backend: batch se3_exp kernel + Tape autodiff."""
    print("\n=== Demo 5: quadrants differentiable backend ===")
    arch = init_quadrants(arch_pref)
    print(f"    backend = {arch} (f64)")
    T_WO, _, _ = demo_scene()
    T_base, info = build_base_pose(T_WO, p_c=[0.0, 0.0, 0.05], d=0.15,
                                   a=[0.15, 0.05, -1.0])
    be = QdBackend(T_base)

    # (a) Batch right-perturbation kernel vs numpy reference (full batch)
    xi = sample_xi(n, 0.002, 0.02, "uniform", rng=rng)
    Ts_qd = be.perturb(xi)  # first call also triggers JIT compilation
    Ts_np = perturb(T_base, xi)
    err = float(np.abs(Ts_qd - Ts_np).max())
    rep.check("(a) batch se3_exp kernel == numpy reference", err < 1e-9,
              f"max |dT| = {err:.2e} over {n} samples")

    # (b) Kernel-side SO(3) validity (independent of the reference path)
    Rs = Ts_qd[:, :3, :3]
    ortho = float(np.abs(np.einsum("nij,nik->njk", Rs, Rs) - np.eye(3)).max())
    det = float(np.abs(np.linalg.det(Rs) - 1.0).max())
    rep.check("(b) kernel rotations in SO(3)", ortho < 1e-12 and det < 1e-12,
              f"ortho err {ortho:.1e}, det err {det:.1e}")

    # (c) Stage-2 readiness: reverse-mode gradient of a pose-deviation loss
    #     L = mean_i [ ||t_i - p_c^W||^2 + w_rot * ||R_i - R_base||_F^2 ]
    n_ad, n_fd, w_rot = 128, 16, 1e-3
    xi_ad = sample_xi(n_ad, 0.002, 0.02, "uniform", rng=rng)
    c = info["p_c_W"]
    loss_val, grad = be.pose_loss_and_grad(xi_ad, c, w_rot)
    loss_ref = pose_loss_np(T_base, xi_ad, c, w_rot)
    h = 1e-6
    idx = rng.choice(n_ad, size=n_fd, replace=False)
    g_fd = np.zeros((n_fd, 6))
    for k in range(n_fd):
        for j in range(6):
            xp = xi_ad.copy(); xp[idx[k], j] += h
            xm = xi_ad.copy(); xm[idx[k], j] -= h
            g_fd[k, j] = (pose_loss_np(T_base, xp, c, w_rot)
                          - pose_loss_np(T_base, xm, c, w_rot)) / (2.0 * h)
    rel = float(np.linalg.norm(grad[idx] - g_fd) / np.linalg.norm(g_fd))
    rep.check("(c) Tape autodiff dL/dxi == central FD", rel < 1e-6,
              f"loss diff {abs(loss_val - loss_ref):.1e}, "
              f"grad rel err {rel:.1e} ({n_fd} samples x 6 comps)")

    # (d) Throughput (info only): numpy loop vs quadrants kernel, end-to-end
    n_t = max(n, 20000)
    xi_t = sample_xi(n_t, 0.002, 0.02, "uniform", rng=rng)
    t0 = time.perf_counter()
    perturb(T_base, xi_t)
    t_np = time.perf_counter() - t0
    t0 = time.perf_counter()
    be.perturb(xi_t)
    t_qd = time.perf_counter() - t0
    print(f"    (d) perturb {n_t} samples: numpy loop {t_np*1e3:7.1f} ms | "
          f"quadrants[{arch}] {t_qd*1e3:7.1f} ms (incl. host copies) | "
          f"speedup x{t_np / t_qd:.1f}")


# --------------------------------------------------------------------------
# Pose-distribution visualization (matplotlib 3D + orthographic projections)
# --------------------------------------------------------------------------

_BOX_EDGES = ((0, 2), (2, 6), (6, 4), (4, 0),
              (1, 3), (3, 7), (7, 5), (5, 1),
              (0, 1), (2, 3), (4, 5), (6, 7))

OBJ_SIZE = (0.06, 0.06, 0.10)  # proxy object size [m], centered at {O}


def box_corners(T, size):
    """Corners (3, 8) of an axis-aligned box of `size`, centered at pose T."""
    sx, sy, sz = np.asarray(size, float) / 2.0
    local = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy)
                      for z in (-sz, sz)], dtype=float).T
    return T[:3, :3] @ local + T[:3, 3:4]


def draw_box(ax, T, size, color="k", lw=1.0, alpha=1.0, label=None):
    """Wireframe box at pose T (object proxy, table / obstacle AABBs)."""
    c = box_corners(T, size)
    for k, (i, j) in enumerate(_BOX_EDGES):
        ax.plot(*c[:, [i, j]], color=color, lw=lw, alpha=alpha,
                label=label if k == 0 else None)


def draw_triad(ax, T, scale=0.05, lw=2.0, alpha=1.0):
    """RGB axes (X/Y/Z) of pose T, each of length `scale`."""
    o = T[:3, 3]
    for j, col in enumerate(("r", "g", "b")):
        p = np.column_stack([o, o + scale * T[:3, j]])
        ax.plot(*p, color=col, lw=lw, alpha=alpha)


def aabb_pose(box):
    """(lo, hi) AABB -> (centered pose T, size)."""
    lo, hi = np.asarray(box, float)
    T = np.eye(4)
    T[:3, 3] = 0.5 * (lo + hi)
    return T, hi - lo


def demo_visualization(data, out_png):
    """Demo 6: sampled pose distribution vs target object.

    2x2 figure: top row = 3D scenes (fallback / prior, fixed camera), bottom
    row = orthographic projections (XY top / XZ front, both modes overlaid).
    Visual sanity targets: TCP cloud shrinks laterally (~0.88 sigma_p under
    uniform roll), finger-pair clouds straddle the object, prior mode is
    visibly tighter than fallback.
    """
    print("\n=== Demo 6: pose distribution visualization ===")
    T_WO, table, obstacle = demo_scene()
    T_base, info, d = data["T_base"], data["info"], data["d"]
    p_c_W = info["p_c_W"]
    h_finger = 0.03  # half finger opening [m]

    modes = (("fallback: uniform roll", data["Ts_fb"], data["xi_fb"], "tab:blue", 0.20),
             ("prior: gaussian roll", data["Ts_pr"], data["xi_pr"], "tab:green", 0.50))

    fig = plt.figure(figsize=(15, 12))
    ax3d = (fig.add_subplot(2, 2, 1, projection="3d"),
            fig.add_subplot(2, 2, 2, projection="3d"))
    ax_xy = fig.add_subplot(2, 2, 3)
    ax_xz = fig.add_subplot(2, 2, 4)

    for ax, (name, Ts, xi, mcol, _) in zip(ax3d, modes):
        t = Ts[:, :3, 3]
        # TCP cloud, colored by |roll| (coupling: larger |roll| -> wider spread)
        sc = ax.scatter(t[:, 0], t[:, 1], t[:, 2], c=np.abs(xi[:, 5]),
                        cmap="viridis", vmin=0.0, vmax=np.pi, s=1.5, alpha=0.3)
        fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, label="|roll| [rad]")
        # Approach arrows (subsampled): TCP -> grasp center along +Z_e
        step_a = max(1, len(Ts) // 150)
        ia = np.arange(0, len(Ts), step_a)
        ax.quiver(t[ia, 0], t[ia, 1], t[ia, 2],
                  Ts[ia, 0, 2], Ts[ia, 1, 2], Ts[ia, 2, 2],
                  length=d, color=mcol, alpha=0.5, lw=0.8, arrow_length_ratio=0.15)
        # Finger-pair cloud (subsampled): g +/- h_finger * X_e, g = t + d * Z_e
        step_f = max(1, len(Ts) // 2000)
        s = Ts[::step_f]
        g = s[:, :3, 3] + d * s[:, :3, 2]
        f1 = g - h_finger * s[:, :3, 0]
        f2 = g + h_finger * s[:, :3, 0]
        ax.scatter(f1[:, 0], f1[:, 1], f1[:, 2], s=2, color="tab:purple",
                   alpha=0.4, label="finger 1 points (sub)")
        ax.scatter(f2[:, 0], f2[:, 1], f2[:, 2], s=2, color="tab:orange",
                   alpha=0.4, label="finger 2 points (sub)")
        # Scene furniture
        draw_box(ax, T_WO, OBJ_SIZE, color="tab:orange", lw=2.0, alpha=0.9,
                 label="object (proxy)")
        for box, col, lab in ((table, "gray", "table"), (obstacle, "tab:red", "obstacle")):
            Tb, sb = aabb_pose(box)
            draw_box(ax, Tb, sb, color=col, lw=1.0, alpha=0.4, label=lab)
        draw_triad(ax, T_base, scale=0.06)
        ax.scatter(*p_c_W, marker="*", s=130, color="k", label="$p_c$")
        ax.set_title(f"{name} — {len(Ts)} samples (TCP cloud by |roll|)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=18, azim=-60)
        ax.legend(loc="upper left", fontsize=8)

    # Shared limits (data-driven, identical extents in both 3D views)
    allpts = np.vstack([data["Ts_fb"][:, :3, 3], data["Ts_pr"][:, :3, 3],
                        box_corners(T_WO, OBJ_SIZE).T, p_c_W])
    lo, hi = allpts.min(0) - 0.06, allpts.max(0) + 0.06
    for ax in ax3d:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(tuple(hi - lo))

    # Orthographic projections (both modes overlaid)
    for ax, (i, j), title in ((ax_xy, (0, 1), "Top view XY (both modes)"),
                              (ax_xz, (0, 2), "Front view XZ (both modes)")):
        for name, Ts, _, mcol, a2d in modes:
            t = Ts[::max(1, len(Ts) // 8000)][:, :3, 3]
            ax.scatter(t[:, i], t[:, j], s=1.5, alpha=a2d, color=mcol,
                       label=f"{name} (sub)")
        c = box_corners(T_WO, OBJ_SIZE)
        ax.plot(c[i, [0, 2, 6, 4, 0]], c[j, [0, 2, 6, 4, 0]], color="tab:orange", lw=2)
        ax.scatter(p_c_W[i], p_c_W[j], marker="*", s=110, color="k")
        ax.set_xlim(lo[i], hi[i])
        ax.set_ylim(lo[j], hi[j])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{'xyz'[i]} [m]")
        ax.set_ylabel(f"{'xyz'[j]} [m]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Stage 1 pose distribution vs target object (frame {W})", fontsize=13)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    pose distribution plot saved: {out_png}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 1 technical validation demo")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--arch", choices=["auto", "cpu", "cuda"], default="auto",
                    help="quadrants backend (auto: cuda if available, else cpu)")
    ap.add_argument("--no-qd", action="store_true",
                    help="skip the quadrants backend demo")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rep = Reporter()
    use_qd = HAVE_QD and not args.no_qd
    print("Stage 1 demo — grasp base pose + probabilistic model (stage1.md)")
    print(f"n_samples={args.n_samples}, seed={args.seed}, "
          f"plot={'off' if args.no_plot or not HAVE_PLT else 'on'}, "
          f"quadrants={'on' if use_qd else 'off'}")

    demo_exp_log_roundtrip(rep, rng)
    demo_base_pose(rep)
    data = demo_sampling(rep, rng, args.n_samples, make_plot=not args.no_plot)
    demo_precheck(rep)
    if use_qd:
        demo_quadrants(rep, rng, args.n_samples, args.arch)
    else:
        print("\n=== Demo 5: quadrants backend skipped "
              f"({'not installed' if not HAVE_QD else '--no-qd'}) ===")
    if not args.no_plot and HAVE_PLT:
        demo_visualization(data, os.path.join(HERE, "stage1_pose_viz.png"))
    else:
        print("\n=== Demo 6: pose distribution visualization skipped (no plot) ===")

    print(f"\n===== Summary: {rep.n_pass} passed, {rep.n_fail} failed =====")
    sys.exit(0 if rep.n_fail == 0 else 1)


if __name__ == "__main__":
    main()
