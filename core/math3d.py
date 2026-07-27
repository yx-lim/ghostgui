"""Small Qt-free math helpers shared by all robotics code.

Quaternion values use MuJoCo's ``[w, x, y, z]`` order and angles are radians.
"""

from __future__ import annotations

import math

import numpy as np


QUATERNION_ORDER = ("w", "x", "y", "z")
ANGLE_UNIT = "radians"


def normalize_quaternion(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,):
        raise ValueError(
            f"quaternion must contain four wxyz values, got {quaternion.shape}"
        )
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion contains a non-finite value")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion norm is too close to zero")
    return quaternion / norm


def rpy_to_quaternion(roll, pitch, yaw):
    roll, pitch, yaw = float(roll), float(pitch), float(yaw)
    if not np.all(np.isfinite((roll, pitch, yaw))):
        raise ValueError("roll, pitch, and yaw must be finite radians")
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return normalize_quaternion(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]))


def quaternion_to_rpy(quaternion):
    w, x, y, z = normalize_quaternion(quaternion)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = (
        math.copysign(math.pi / 2.0, sinp)
        if abs(sinp) >= 1.0
        else math.asin(sinp)
    )
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def quaternion_slerp(start, end, fraction):
    start = normalize_quaternion(start)
    end = normalize_quaternion(end)
    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(start + fraction * (end - start))
    theta = math.acos(dot)
    return (
        math.sin((1.0 - fraction) * theta) * start
        + math.sin(fraction * theta) * end
    ) / math.sin(theta)


def quaternion_angle(start, end):
    start = normalize_quaternion(start)
    end = normalize_quaternion(end)
    return 2.0 * math.acos(
        float(np.clip(abs(np.dot(start, end)), -1.0, 1.0))
    )
