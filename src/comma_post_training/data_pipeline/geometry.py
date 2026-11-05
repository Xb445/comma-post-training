"""Explicit right-handed frame conversions and relative SE(3) motion."""

import numpy as np


def validate_poses(poses: np.ndarray) -> None:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("poses must be finite [N,4,4] transforms T_world_from_vehicle")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("pose homogeneous rows must be [0,0,0,1]")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-4):
        raise ValueError("pose rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("pose rotations must have determinant +1")


def relative_pose(origin: np.ndarray, destination: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    rotation = origin[:3, :3].T
    result[:3, :3] = rotation @ destination[:3, :3]
    result[:3, 3] = rotation @ (destination[:3, 3] - origin[:3, 3])
    return result


def motion_vector(origin: np.ndarray, destination: np.ndarray) -> np.ndarray:
    relative = relative_pose(origin, destination)
    rotation = relative[:3, :3]
    # R = Rz(yaw) Ry(pitch) Rx(roll), radians; near gimbal lock is rejected.
    horizontal = np.hypot(rotation[0, 0], rotation[1, 0])
    if horizontal < 1e-6:
        raise ValueError("relative pose is too close to Euler gimbal lock")
    angles = [
        np.arctan2(rotation[2, 1], rotation[2, 2]),
        np.arctan2(-rotation[2, 0], horizontal),
        np.arctan2(rotation[1, 0], rotation[0, 0]),
    ]
    return np.asarray([*relative[:3, 3], *angles], dtype=np.float32)


def flu_to_frd_poses(poses: np.ndarray) -> np.ndarray:
    """Change both world and vehicle bases from forward-left-up to forward-right-down."""
    validate_poses(poses)
    change = np.diag([1.0, -1.0, -1.0, 1.0])
    return change @ poses @ change
