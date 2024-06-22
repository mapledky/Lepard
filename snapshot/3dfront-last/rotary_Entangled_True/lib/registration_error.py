import numpy as np
from typing import Tuple

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_matrix(rot, trn):
    rot = np.array(rot.cpu().numpy())
    trn = np.array(trn.cpu().numpy())
    gt_transform = np.concatenate((rot.reshape(3 , 3), trn.reshape(3 , 1)), axis=1)
    gt_transform = np.concatenate((gt_transform, np.array([[0, 0, 0, 1]])), axis=0)
    return gt_transform

def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)

def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation

def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte