#!/usr/bin/env python

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_calibration import (
    CameraCalibration,
    calibrate_from_points,
    find_chessboard_corners,
    make_object_points,
)


class CameraCalibrationTest(unittest.TestCase):
    def test_object_points_use_columns_then_rows(self) -> None:
        points = make_object_points((3, 2), 25.0)
        np.testing.assert_allclose(
            points[:, :2],
            [[0, 0], [25, 0], [50, 0], [0, 25], [25, 25], [50, 25]],
        )

    def test_finds_corners_in_generated_chessboard(self) -> None:
        square = 40
        margin = 30
        board = np.full((7 * square + 2 * margin, 9 * square + 2 * margin), 255, np.uint8)
        for row in range(7):
            for column in range(9):
                if (row + column) % 2 == 0:
                    top_left = (margin + column * square, margin + row * square)
                    bottom_right = (
                        margin + (column + 1) * square,
                        margin + (row + 1) * square,
                    )
                    cv2.rectangle(board, top_left, bottom_right, 0, thickness=-1)

        corners = find_chessboard_corners(board, (8, 6))

        self.assertIsNotNone(corners)
        assert corners is not None
        self.assertEqual(corners.shape, (48, 1, 2))

    def test_recovers_synthetic_intrinsics(self) -> None:
        image_size = (640, 480)
        expected_matrix = np.array(
            [[610.0, 0.0, 318.0], [0.0, 608.0, 242.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        expected_distortion = np.array([[-0.12, 0.04, 0.001, -0.001, 0.0]])
        object_template = make_object_points((8, 6), 25.0)
        object_points = []
        image_points = []

        poses = [
            ((0.00, 0.00, 0.00), (-100.0, -65.0, 650.0)),
            ((0.12, -0.08, 0.04), (-80.0, -45.0, 590.0)),
            ((-0.10, 0.16, -0.05), (-120.0, -70.0, 720.0)),
            ((0.18, 0.08, 0.10), (-60.0, -85.0, 680.0)),
            ((-0.16, -0.12, 0.07), (-135.0, -30.0, 620.0)),
            ((0.06, 0.20, -0.12), (-75.0, -95.0, 760.0)),
            ((-0.20, 0.05, 0.14), (-115.0, -55.0, 560.0)),
            ((0.14, -0.18, -0.08), (-50.0, -60.0, 700.0)),
            ((-0.08, -0.22, 0.03), (-145.0, -80.0, 670.0)),
            ((0.22, 0.12, -0.14), (-90.0, -40.0, 740.0)),
        ]
        for rotation, translation in poses:
            projected, _ = cv2.projectPoints(
                object_template,
                np.asarray(rotation, dtype=np.float64),
                np.asarray(translation, dtype=np.float64),
                expected_matrix,
                expected_distortion,
            )
            object_points.append(object_template.copy())
            image_points.append(projected.astype(np.float32))

        result = calibrate_from_points(
            object_points,
            image_points,
            image_size,
            pattern_size=(8, 6),
            square_size_mm=25.0,
        )

        self.assertLess(result.rms, 0.001)
        np.testing.assert_allclose(result.camera_matrix, expected_matrix, atol=0.1)
        np.testing.assert_allclose(
            result.distortion_coefficients.reshape(-1)[:4],
            expected_distortion.reshape(-1)[:4],
            atol=0.002,
        )

    def test_json_round_trip_and_resolution_check(self) -> None:
        calibration = CameraCalibration(
            image_size=(640, 480),
            camera_matrix=np.array(
                [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
            ),
            distortion_coefficients=np.zeros((1, 5)),
            rms=0.2,
            mean_reprojection_error_px=0.18,
            pattern_size=(8, 6),
            square_size_mm=25.0,
            per_view_errors_px=(0.15, 0.21),
            source_images=("calib_001.png", "calib_002.png"),
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "camera.json"
            calibration.save(path)
            loaded = CameraCalibration.load(path)

        np.testing.assert_allclose(loaded.camera_matrix, calibration.camera_matrix)
        self.assertEqual(loaded.image_size, (640, 480))
        with self.assertRaisesRegex(ValueError, "does not match calibration"):
            loaded.undistort(np.zeros((240, 320, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
