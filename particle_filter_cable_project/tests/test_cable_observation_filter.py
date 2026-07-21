import unittest

import numpy as np

from cable_cuda import CudaPointCloudView
from cable_detection import (
    CableDetection2D,
    CableMaskDetector,
    sampled_cable_observation,
)

try:
    import torch
except Exception:
    torch = None


class CableObservationFilterTests(unittest.TestCase):
    @staticmethod
    def _scene():
        points = np.full((1, 10, 4), np.nan, dtype=np.float32)
        points[0, 0, :3] = (0.000, 0.0, 1.0)
        points[0, 1, :3] = (0.004, 0.0, 1.0)
        points[0, 2, :3] = (0.008, 0.0, 1.0)
        points[0, 3, :3] = (0.200, 0.0, 1.0)  # spatially isolated
        points[0, 5, :3] = (0.000, 0.0, 2.0)  # outside depth range
        points[0, 6, :3] = (0.030, 0.0, 1.0)  # poor confidence
        points[0, 7, :3] = (0.040, 0.0, 1.0)  # small component
        points[0, 8, :3] = (0.050, 0.0, 1.0)  # removed by morphology
        points[..., 3] = 1.0

        mask = np.zeros((1, 10), dtype=np.uint8)
        mask[0, :7] = 255
        component = np.zeros_like(mask)
        component[0, 7] = 255
        morphology = np.zeros_like(mask)
        morphology[0, 8] = 255
        detection = CableDetection2D(
            mask=mask,
            component_count=1,
            component_rejected_mask=component,
            morphology_rejected_mask=morphology,
        )
        confidence = np.zeros((1, 10), dtype=np.float32)
        confidence[0, 6] = 99.0
        return points, detection, confidence

    @classmethod
    def _observe(cls, point_cloud, confidence_map=None, enabled=True):
        _points, detection, default_confidence = cls._scene()
        return sampled_cable_observation(
            point_cloud,
            detection,
            depth_min=0.05,
            depth_max=1.5,
            confidence_map=(
                default_confidence if confidence_map is None else confidence_map
            ),
            max_confidence=50.0,
            max_points=100,
            spatial_filter_enabled=enabled,
            spatial_radius_m=0.009,
            spatial_min_neighbors=2,
        )

    def test_rejection_reasons_are_explicit_and_pf_independent(self):
        points, _detection, confidence = self._scene()
        observation = self._observe(points, confidence, enabled=True)

        self.assertEqual(observation.accepted_count, 3)
        self.assertEqual(observation.rejected_count, 6)
        self.assertEqual(len(observation.rejected_points_xyz), 5)
        self.assertEqual(len(observation.rejected_pixels_xy), 6)
        self.assertEqual(
            observation.rejection_counts,
            {
                "invalid_depth": 1,
                "depth_range": 1,
                "depth_confidence": 1,
                "small_component": 1,
                "mask_morphology": 1,
                "spatial_isolation": 1,
            },
        )

    def test_spatial_filter_has_an_independent_feature_gate(self):
        points, _detection, confidence = self._scene()
        enabled = self._observe(points, confidence, enabled=True)
        disabled = self._observe(points, confidence, enabled=False)

        self.assertEqual(enabled.accepted_count, 3)
        self.assertEqual(disabled.accepted_count, 4)
        self.assertEqual(enabled.rejection_counts["spatial_isolation"], 1)
        self.assertEqual(disabled.rejection_counts["spatial_isolation"], 0)
        self.assertTrue(enabled.spatial_filter_applied)
        self.assertFalse(disabled.spatial_filter_applied)

    def test_partial_occlusion_does_not_remove_locally_supported_observations(self):
        points = np.full((1, 7, 4), np.nan, dtype=np.float32)
        points[0, :, 3] = 1.0
        points[0, :3, :3] = (
            (0.000, 0.0, 1.0),
            (0.004, 0.0, 1.0),
            (0.008, 0.0, 1.0),
        )
        points[0, 4:, :3] = (
            (0.040, 0.0, 1.0),
            (0.044, 0.0, 1.0),
            (0.048, 0.0, 1.0),
        )
        mask = np.full((1, 7), 255, dtype=np.uint8)
        mask[0, 3] = 0  # a genuine missing/occluded interval
        observation = sampled_cable_observation(
            points,
            CableDetection2D(mask=mask, component_count=2),
            depth_min=0.05,
            depth_max=1.5,
            max_points=100,
            spatial_filter_enabled=True,
            spatial_radius_m=0.009,
            spatial_min_neighbors=2,
        )

        self.assertEqual(observation.accepted_count, 6)
        self.assertEqual(observation.rejected_count, 0)

    def test_depth_confidence_filter_can_be_disabled_without_an_alternative(self):
        points, detection, confidence = self._scene()
        filtered = sampled_cable_observation(
            points,
            detection,
            depth_min=0.05,
            depth_max=1.5,
            confidence_map=confidence,
            max_confidence=50.0,
            max_points=100,
            spatial_filter_enabled=False,
        )
        unfiltered = sampled_cable_observation(
            points,
            detection,
            depth_min=0.05,
            depth_max=1.5,
            confidence_map=confidence,
            max_confidence=None,
            max_points=100,
            spatial_filter_enabled=False,
        )

        self.assertEqual(filtered.accepted_count, 4)
        self.assertEqual(unfiltered.accepted_count, 5)
        self.assertEqual(filtered.rejection_counts["depth_confidence"], 1)
        self.assertEqual(unfiltered.rejection_counts["depth_confidence"], 0)

    def test_component_and_morphology_filters_execute_independently(self):
        component_mask = np.zeros((7, 9), dtype=np.uint8)
        component_mask[2, 1:5] = 255
        component_mask[5, 7] = 255
        component_on = CableMaskDetector(
            min_area=2,
            open_kernel=1,
            close_kernel=1,
        ).clean_mask(component_mask)
        component_off = CableMaskDetector(
            min_area=1,
            open_kernel=1,
            close_kernel=1,
        ).clean_mask(component_mask)

        self.assertEqual(int(component_on[0][5, 7]), 0)
        self.assertEqual(int(component_on[2][5, 7]), 255)
        self.assertEqual(int(component_off[0][5, 7]), 255)

        morphology_on = CableMaskDetector(
            min_area=1,
            open_kernel=3,
            close_kernel=1,
        ).clean_mask(component_mask)
        self.assertEqual(int(morphology_on[0][5, 7]), 0)
        self.assertEqual(int(morphology_on[3][5, 7]), 255)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_and_cpu_observation_decisions_match(self):
        points, _detection, confidence = self._scene()
        cpu = self._observe(points, confidence, enabled=True)

        points_t = torch.as_tensor(points, device="cuda").contiguous()
        confidence_t = torch.as_tensor(confidence, device="cuda").contiguous()
        view = CudaPointCloudView(
            pointer=points_t.data_ptr(),
            width=points_t.shape[1],
            height=points_t.shape[0],
            step_bytes=points_t.stride(0) * points_t.element_size(),
            confidence_pointer=confidence_t.data_ptr(),
            confidence_step_bytes=(
                confidence_t.stride(0) * confidence_t.element_size()
            ),
            owner=points_t,
            confidence_owner=confidence_t,
        )
        gpu = self._observe(view, confidence, enabled=True)

        self.assertEqual(gpu.rejection_counts, cpu.rejection_counts)
        np.testing.assert_allclose(gpu.accepted_points_xyz, cpu.accepted_points_xyz)
        np.testing.assert_allclose(gpu.rejected_points_xyz, cpu.rejected_points_xyz)
        np.testing.assert_array_equal(gpu.rejected_pixels_xy, cpu.rejected_pixels_xy)


if __name__ == "__main__":
    unittest.main()
