import tempfile
import unittest
from pathlib import Path

from pf_ablation import (
    ExperimentRecorder,
    FEATURE_BY_KEY,
    leave_one_out_state,
    minimal_baseline_state,
    read_ablation_jsonl,
    summarize_ablation_records,
)


class ParticleFilterAblationTests(unittest.TestCase):
    @staticmethod
    def _configured_state():
        return {key: True for key in FEATURE_BY_KEY}

    def test_leave_one_out_removes_only_feature_and_required_dependents(self):
        state = leave_one_out_state(self._configured_state(), "pf_endpoint_tangent")
        self.assertFalse(state["pf_endpoint_tangent"])
        self.assertFalse(state["pf_endpoint_tangent_ransac"])
        self.assertFalse(state["pf_endpoint_tangent_likelihood"])
        self.assertFalse(state["pf_conditioned_proposals"])
        self.assertTrue(state["pf_velocity"])
        self.assertTrue(state["crossing_proposals"])
        self.assertTrue(state["pf_dense_path_support"])

    def test_recording_summary_discards_warmup_per_revision(self):
        features = minimal_baseline_state(self._configured_state())
        with tempfile.TemporaryDirectory() as temporary:
            recorder = ExperimentRecorder(
                temporary,
                {"schema_version": 1, "initial_features": features},
            )
            for frame_index, residual in enumerate((9.0, 1.0, 3.0)):
                recorder.record(
                    frame_index,
                    {"revision": 0, "features": features},
                    {
                        "path_support_rms_m": residual,
                        "effective_sample_size": 100.0,
                        "per_cable": [
                            {"path_support_rms_m": residual},
                            {"path_support_rms_m": 2.0 * residual},
                        ],
                    },
                    {"filter": 0.010},
                )
            changed = dict(features)
            changed["pf_velocity"] = True
            for frame_index, residual in enumerate((8.0, 2.0, 4.0), start=3):
                recorder.record(
                    frame_index,
                    {"revision": 1, "features": changed},
                    {"path_support_rms_m": residual, "effective_sample_size": 120.0},
                    {"filter": 0.012},
                )
            recorder.close()
            metadata, frames = read_ablation_jsonl(Path(recorder.path))
            summary = summarize_ablation_records(metadata, frames, warmup_frames=1)

        self.assertEqual(len(summary["revisions"]), 2)
        self.assertEqual(summary["revisions"][0]["retained_frame_count"], 2)
        self.assertAlmostEqual(
            summary["revisions"][0]["metrics"]["path_support_rms_m"]["median"],
            2.0,
        )
        self.assertAlmostEqual(
            summary["revisions"][0]["metrics"]["pf2.path_support_rms_m"]["median"],
            4.0,
        )
        self.assertAlmostEqual(
            summary["revisions"][1]["metrics"]["filter_ms"]["median"],
            12.0,
        )
        self.assertEqual(
            summary["revisions"][1]["changes_from_configured"],
            {"pf_velocity": True},
        )
        self.assertAlmostEqual(
            summary["revisions"][1]["comparison_to_baseline"]["filter_ms"]["median_delta"],
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
