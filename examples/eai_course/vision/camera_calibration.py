"""OpenCV chessboard camera calibration helpers.

The algorithm follows OpenCV's official Python calibration sample and tutorial:
https://github.com/opencv/opencv/blob/4.x/samples/python/calibrate.py
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsic camera parameters and lens distortion coefficients."""

    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rms: float
    mean_reprojection_error_px: float
    pattern_size: tuple[int, int]
    square_size_mm: float
    per_view_errors_px: tuple[float, ...] = ()
    source_images: tuple[str, ...] = ()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "model": "opencv_pinhole",
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "pattern": {
                "type": "chessboard",
                "inner_corners": {
                    "columns": self.pattern_size[0],
                    "rows": self.pattern_size[1],
                },
                "square_size_mm": self.square_size_mm,
            },
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.reshape(-1).tolist(),
            "rms": self.rms,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
            "per_view_errors_px": list(self.per_view_errors_px),
            "source_images": list(self.source_images),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CameraCalibration:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("model") != "opencv_pinhole":
            raise ValueError(f"Unsupported camera calibration format: {path}")
        size = payload["image_size"]
        corners = payload["pattern"]["inner_corners"]
        return cls(
            image_size=(int(size["width"]), int(size["height"])),
            camera_matrix=np.asarray(payload["camera_matrix"], dtype=np.float64),
            distortion_coefficients=np.asarray(
                payload["distortion_coefficients"], dtype=np.float64
            ).reshape(1, -1),
            rms=float(payload["rms"]),
            mean_reprojection_error_px=float(payload["mean_reprojection_error_px"]),
            pattern_size=(int(corners["columns"]), int(corners["rows"])),
            square_size_mm=float(payload["pattern"]["square_size_mm"]),
            per_view_errors_px=tuple(float(x) for x in payload.get("per_view_errors_px", [])),
            source_images=tuple(str(x) for x in payload.get("source_images", [])),
        )

    def undistort(self, frame: np.ndarray, *, alpha: float = 0.0) -> np.ndarray:
        height, width = frame.shape[:2]
        if (width, height) != self.image_size:
            raise ValueError(
                f"Frame size {width}x{height} does not match calibration "
                f"{self.image_size[0]}x{self.image_size[1]}"
            )
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.distortion_coefficients,
            self.image_size,
            alpha,
            self.image_size,
        )
        return cv2.undistort(
            frame,
            self.camera_matrix,
            self.distortion_coefficients,
            None,
            new_matrix,
        )


def make_object_points(
    pattern_size: tuple[int, int], square_size_mm: float
) -> np.ndarray:
    """Return planar chessboard points in OpenCV's column-first order."""
    columns, rows = pattern_size
    if columns < 2 or rows < 2:
        raise ValueError("Chessboard must have at least 2x2 inner corners")
    if square_size_mm <= 0:
        raise ValueError("Square size must be positive")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    points[:, :2] *= square_size_mm
    return points


def find_chessboard_corners(
    image: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    """Find and refine chessboard inner corners using the official OpenCV flow."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def calibrate_from_points(
    object_points: Sequence[np.ndarray],
    image_points: Sequence[np.ndarray],
    image_size: tuple[int, int],
    *,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    source_images: Sequence[str] = (),
) -> CameraCalibration:
    """Run ``cv2.calibrateCamera`` and calculate per-view pixel errors."""
    if len(object_points) != len(image_points):
        raise ValueError("Object-point and image-point view counts differ")
    if len(object_points) < 3:
        raise ValueError("At least 3 valid views are required")

    rms, matrix, distortion, rotation_vectors, translation_vectors = cv2.calibrateCamera(
        list(object_points), list(image_points), image_size, None, None
    )

    errors: list[float] = []
    for object_view, image_view, rotation, translation in zip(
        object_points,
        image_points,
        rotation_vectors,
        translation_vectors,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(
            object_view, rotation, translation, matrix, distortion
        )
        residual = image_view.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))

    return CameraCalibration(
        image_size=image_size,
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        rms=float(rms),
        mean_reprojection_error_px=float(np.mean(errors)),
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
        per_view_errors_px=tuple(errors),
        source_images=tuple(source_images),
    )


def calibrate_images(
    image_paths: Sequence[Path],
    *,
    pattern_size: tuple[int, int],
    square_size_mm: float,
    minimum_views: int = 10,
    debug_dir: Path | None = None,
) -> tuple[CameraCalibration, list[Path]]:
    """Detect corners in image files and solve camera intrinsics."""
    template = make_object_points(pattern_size, square_size_mm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[Path] = []
    image_size: tuple[int, int] | None = None

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"SKIP unreadable image: {path}")
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(
                f"All images must have one resolution; {path} is "
                f"{current_size[0]}x{current_size[1]}, expected "
                f"{image_size[0]}x{image_size[1]}"
            )

        corners = find_chessboard_corners(image, pattern_size)
        if corners is None:
            print(f"SKIP pattern not found: {path.name}")
            continue
        object_points.append(template.copy())
        image_points.append(corners)
        accepted.append(path)
        print(f"OK   {path.name}")

        if debug_dir is not None:
            annotated = image.copy()
            cv2.drawChessboardCorners(annotated, pattern_size, corners, True)
            cv2.imwrite(str(debug_dir / f"{path.stem}_corners.png"), annotated)

    if image_size is None:
        raise ValueError("No readable calibration images were found")
    if len(accepted) < minimum_views:
        raise ValueError(
            f"Only {len(accepted)} valid views; capture at least {minimum_views}"
        )

    calibration = calibrate_from_points(
        object_points,
        image_points,
        image_size,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
        source_images=[path.name for path in accepted],
    )
    return calibration, accepted
