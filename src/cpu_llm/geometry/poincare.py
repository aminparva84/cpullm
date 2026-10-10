"""Poincaré ball geometry and Ganea-style entailment cones.

Inclusion is directional. Symmetric Poincaré distance is implemented for the
retrieval baseline; it is not a subclass test.

Per pair the cone test is O(d) arithmetic, not O(1). It does not by itself
prove that a taxonomy edge exists.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_K = 0.1
_EPS = 1e-6


def project_to_ball(vector: np.ndarray, max_norm: float = 1.0 - 1e-5) -> np.ndarray:
    point = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(point))
    if norm < max_norm:
        return point
    if norm == 0.0:
        return point
    return point * (max_norm / norm)


def half_aperture(parent: np.ndarray, k: float = DEFAULT_K) -> float | None:
    """Return θ_u, or None when the apex is too close to the origin, outside the ball, or the sine argument is not positive.

    θ_u = arcsin(min(1, K * (1 - ||u||^2) / ||u||))

    The formula divides by ||u||, so the origin is not a valid apex. When the
    raw sine argument is above 1, the aperture saturates at π/2. A non-positive
    argument yields no cone.
    """
    if k <= 0.0:
        raise ValueError("K must be positive")
    norm_u = float(np.linalg.norm(np.asarray(parent, dtype=np.float64)))
    if norm_u < _EPS or norm_u >= 1.0:
        return None
    sin_arg = k * ((1.0 - norm_u * norm_u) / norm_u)
    if sin_arg <= 0.0:
        return None
    return float(math.asin(min(1.0, sin_arg)))


def poincare_distance(left: np.ndarray, right: np.ndarray) -> float:
    u = np.asarray(left, dtype=np.float64)
    v = np.asarray(right, dtype=np.float64)
    norm_u_sq = float(np.dot(u, u))
    norm_v_sq = float(np.dot(v, v))
    if norm_u_sq >= 1.0 or norm_v_sq >= 1.0:
        return math.inf
    denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq)
    if denom <= 1e-15:
        return math.inf
    diff = u - v
    alpha = 1.0 + 2.0 * float(np.dot(diff, diff)) / denom
    if alpha <= 1.0:
        return 0.0
    return float(math.acosh(alpha))


def is_in_entailment_cone(
    parent_u: np.ndarray,
    child_v: np.ndarray,
    k: float = DEFAULT_K,
) -> bool:
    """True when child_v lies in the entailment cone rooted at parent_u.

    Requires ||u|| < ||v|| and angle(u, v - u) <= θ_u, with both points inside
    the open unit ball. Equality of the two points is not taxonomic inclusion.
    """
    parent = np.asarray(parent_u, dtype=np.float64)
    child = np.asarray(child_v, dtype=np.float64)
    theta = half_aperture(parent, k)
    if theta is None:
        return False
    norm_u = float(np.linalg.norm(parent))
    norm_v = float(np.linalg.norm(child))
    if norm_v >= 1.0 or norm_u >= norm_v:
        return False
    diff = child - parent
    norm_diff = float(np.linalg.norm(diff))
    if norm_diff < _EPS:
        return False
    cos_angle = float(np.dot(parent, diff) / (norm_u * norm_diff))
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle = math.acos(cos_angle)
    return bool(angle <= theta)


def batch_in_entailment_cone(
    parent_u: np.ndarray,
    children: np.ndarray,
    k: float = DEFAULT_K,
) -> np.ndarray:
    """Vectorized cone test. Row i is true when children[i] is inside the cone of parent_u."""
    parent = np.asarray(parent_u, dtype=np.float64)
    points = np.asarray(children, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("children must have shape (n, d)")
    hits = np.zeros(points.shape[0], dtype=bool)
    theta = half_aperture(parent, k)
    if theta is None:
        return hits
    norm_u = float(np.linalg.norm(parent))
    norm_v = np.linalg.norm(points, axis=1)
    diff = points - parent
    norm_diff = np.linalg.norm(diff, axis=1)
    cos_angle = (diff @ parent) / (norm_u * np.maximum(norm_diff, 1e-12))
    np.clip(cos_angle, -1.0, 1.0, out=cos_angle)
    angle = np.arccos(cos_angle)
    hits[:] = (
        (norm_v < 1.0)
        & (norm_u < norm_v)
        & (norm_diff >= _EPS)
        & (angle <= theta)
    )
    return hits


def batch_poincare_distance(origin: np.ndarray, points: np.ndarray) -> np.ndarray:
    u = np.asarray(origin, dtype=np.float64)
    vs = np.asarray(points, dtype=np.float64)
    norm_u_sq = float(np.dot(u, u))
    distances = np.full(vs.shape[0], math.inf, dtype=np.float64)
    if norm_u_sq >= 1.0:
        return distances
    norm_v_sq = np.sum(vs * vs, axis=1)
    inside = norm_v_sq < 1.0
    if not np.any(inside):
        return distances
    diff = vs - u
    diff_sq = np.sum(diff * diff, axis=1)
    denom = (1.0 - norm_u_sq) * (1.0 - norm_v_sq)
    alpha = 1.0 + 2.0 * diff_sq / np.maximum(denom, 1e-15)
    good = inside & (denom > 1e-15) & (alpha > 1.0)
    distances[good] = np.arccosh(alpha[good])
    distances[inside & (alpha <= 1.0)] = 0.0
    return distances
