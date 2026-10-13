"""Eye-in-hand calibration math using OpenCV and SO-101 FK transforms.

Transform names follow ``destination_T_source``. For example,
``base_T_gripper`` maps a point from the gripper frame into the base frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from camera_calibration import (
    CameraCalibration,
    find_chessboard_corners,
    make_object_points,
)


HAND_EYE_METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def transform_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return transform_from_rt(rotation.T, -rotation.T @ translation)


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def rotation_to_rpy_degrees(rotation: np.ndarray) -> np.ndarray:
    """Return fixed-axis XYZ roll, pitch, yaw angles in degrees."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-8:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def mean_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    summed = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    left, _, right = np.linalg.svd(summed)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


@dataclass(frozen=True)
class TargetObservation:
    camera_T_target: np.ndarray
    corners: np.ndarray
    reprojection_error_px: float


@dataclass(frozen=True)
class HandEyeSample:
    name: str
    image: str
    joint_degrees: tuple[float, ...]
    base_T_gripper: np.ndarray
    camera_T_target: np.ndarray
    pnp_reprojection_error_px: float

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "image": self.image,
            "joint_degrees": list(self.joint_degrees),
            "base_T_gripper": self.base_T_gripper.tolist(),
            "camera_T_target": self.camera_T_target.tolist(),
            "pnp_reprojection_error_px": self.pnp_reprojection_error_px,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> HandEyeSample:
        return cls(
            name=str(payload["name"]),
            image=str(payload["image"]),
            joint_degrees=tuple(float(value) for value in payload["joint_degrees"]),
            base_T_gripper=np.asarray(payload["base_T_gripper"], dtype=np.float64),
            camera_T_target=np.asarray(payload["camera_T_target"], dtype=np.float64),
            pnp_reprojection_error_px=float(payload["pnp_reprojection_error_px"]),
        )


@dataclass(frozen=True)
class HandEyeResult:
    method: str
    gripper_T_camera: np.ndarray
    base_T_target: np.ndarray
    board_origin_rmse_mm: np.ndarray
    board_origin_max_abs_mm: np.ndarray
    board_rotation_rmse_deg: float
    sample_count: int

    @property
    def camera_translation_in_gripper_mm(self) -> np.ndarray:
        return self.gripper_T_camera[:3, 3] * 1000.0

    @property
    def camera_rpy_in_gripper_deg(self) -> np.ndarray:
        return rotation_to_rpy_degrees(self.gripper_T_camera[:3, :3])

    def to_json(self) -> dict[str, object]:
        return {
            "method": self.method,
            "sample_count": self.sample_count,
            "transform_convention": "destination_T_source",
            "gripper_T_camera": self.gripper_T_camera.tolist(),
            "camera_translation_in_gripper_mm": self.camera_translation_in_gripper_mm.tolist(),
            "camera_rpy_xyz_in_gripper_deg": self.camera_rpy_in_gripper_deg.tolist(),
            "estimated_base_T_target": self.base_T_target.tolist(),
            "estimated_board_origin_in_base_mm": (
                self.base_T_target[:3, 3] * 1000.0
            ).tolist(),
            "consistency": {
                "board_origin_rmse_xyz_mm": self.board_origin_rmse_mm.tolist(),
                "board_origin_max_abs_xyz_mm": self.board_origin_max_abs_mm.tolist(),
                "board_rotation_rmse_deg": self.board_rotation_rmse_deg,
            },
        }


def estimate_target_pose(
    frame: np.ndarray,
    calibration: CameraCalibration,
) -> TargetObservation | None:
    """Estimate ``camera_T_target`` from a raw chessboard image."""
    corners = find_chessboard_corners(frame, calibration.pattern_size)
    if corners is None:
        return None

    # FK uses metres, so PnP object points must also use metres.
    object_points = make_object_points(
        calibration.pattern_size, calibration.square_size_mm / 1000.0
    )
    success, rotation_vector, translation = cv2.solvePnP(
        object_points,
        corners,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_points,
        corners,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
        rotation_vector,
        translation,
    )
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        calibration.camera_matrix,
        calibration.distortion_coefficients,
    )
    residual = corners.reshape(-1, 2) - projected.reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return TargetObservation(
        camera_T_target=transform_from_rt(rotation, translation),
        corners=corners,
        reprojection_error_px=error,
    )


def save_dataset(
    path: Path,
    samples: Sequence[HandEyeSample],
    *,
    camera_calibration_path: Path,
    urdf_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "configuration": "eye_in_hand",
        "transform_convention": "destination_T_source",
        "units": {"translation": "metre", "joint_angle": "degree"},
        "camera_calibration": str(camera_calibration_path.resolve()),
        "urdf": str(urdf_path.resolve()),
        "samples": [sample.to_json() for sample in samples],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_dataset(path: Path) -> tuple[dict[str, object], list[HandEyeSample]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("configuration") != "eye_in_hand":
        raise ValueError(f"Unsupported hand-eye dataset: {path}")
    samples = [HandEyeSample.from_json(item) for item in payload["samples"]]
    return payload, samples


def solve_hand_eye(samples: Sequence[HandEyeSample], method: str = "park") -> HandEyeResult:
    """Solve ``gripper_T_camera`` and evaluate fixed-target consistency."""
    if method not in HAND_EYE_METHODS:
        raise ValueError(f"Unknown hand-eye method: {method}")
    if len(samples) < 3:
        raise ValueError("At least 3 samples are required")

    gripper_rotations = [sample.base_T_gripper[:3, :3] for sample in samples]
    gripper_translations = [sample.base_T_gripper[:3, 3:4] for sample in samples]
    target_rotations = [sample.camera_T_target[:3, :3] for sample in samples]
    target_translations = [sample.camera_T_target[:3, 3:4] for sample in samples]
    camera_rotation, camera_translation = cv2.calibrateHandEye(
        gripper_rotations,
        gripper_translations,
        target_rotations,
        target_translations,
        method=HAND_EYE_METHODS[method],
    )
    gripper_T_camera = transform_from_rt(camera_rotation, camera_translation)
    if not np.all(np.isfinite(gripper_T_camera)):
        raise RuntimeError(f"OpenCV {method} returned non-finite values")

    target_transforms = [
        sample.base_T_gripper @ gripper_T_camera @ sample.camera_T_target
        for sample in samples
    ]
    origins = np.asarray([transform[:3, 3] for transform in target_transforms])
    origin_mean = np.mean(origins, axis=0)
    origin_residual_mm = (origins - origin_mean) * 1000.0
    origin_rmse = np.sqrt(np.mean(origin_residual_mm * origin_residual_mm, axis=0))
    origin_max_abs = np.max(np.abs(origin_residual_mm), axis=0)

    target_rotation = mean_rotation([transform[:3, :3] for transform in target_transforms])
    rotation_errors = [
        rotation_angle_degrees(target_rotation.T @ transform[:3, :3])
        for transform in target_transforms
    ]
    base_T_target = transform_from_rt(target_rotation, origin_mean)
    return HandEyeResult(
        method=method,
        gripper_T_camera=gripper_T_camera,
        base_T_target=base_T_target,
        board_origin_rmse_mm=origin_rmse,
        board_origin_max_abs_mm=origin_max_abs,
        board_rotation_rmse_deg=float(np.sqrt(np.mean(np.square(rotation_errors)))),
        sample_count=len(samples),
    )


def solve_all_methods(samples: Sequence[HandEyeSample]) -> dict[str, HandEyeResult]:
    results: dict[str, HandEyeResult] = {}
    for name in HAND_EYE_METHODS:
        try:
            results[name] = solve_hand_eye(samples, name)
        except (cv2.error, RuntimeError):
            continue
    return results
