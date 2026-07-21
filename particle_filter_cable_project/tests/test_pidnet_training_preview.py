import unittest

import numpy as np

from pidnet_schema import label_bit
from tools.pidnet_training_gui import (
    PidNetTrainingApp,
    apply_binary_cleanup,
    compact_item_edit_state,
    prediction_channels_to_draft,
    restore_item_edit_state,
    toml_scalar,
)


class _Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class PidNetTrainingPreviewTests(unittest.TestCase):
    @staticmethod
    def _app():
        app = PidNetTrainingApp.__new__(PidNetTrainingApp)
        app.test_threshold_var = _Variable(0.95)
        app.endpoint_threshold_vars = (_Variable(0.50), _Variable(0.98))
        app.crossing_threshold_var = _Variable(0.77)
        app.cable_count_var = _Variable(2)
        return app

    def test_preview_applies_each_endpoint_threshold_to_its_own_channel(self):
        app = self._app()
        probability = np.zeros((1, 3, 4), dtype=np.float32)
        probability[0, 0, 1] = 0.60
        probability[0, 1, 2] = 0.97
        probability[0, 2, 2] = 0.99
        np.testing.assert_array_equal(
            app.endpoint_prediction_mask(probability),
            np.asarray([[True, False, True]]),
        )

    def test_toml_writer_preserves_numeric_threshold_arrays(self):
        self.assertEqual(toml_scalar((0.50, 0.98)), "[0.5, 0.98]")

    def test_binary_cleanup_returns_mask_and_component_count(self):
        raw = np.zeros((20, 30), dtype=np.uint8)
        raw[2:8, 3:9] = 255
        raw[12:18, 20:27] = 255
        cleaned, component_count = apply_binary_cleanup(
            raw,
            {"open_kernel": 0, "close_kernel": 0, "min_area_px": 5},
        )
        self.assertIsInstance(cleaned, np.ndarray)
        self.assertEqual(cleaned.shape, raw.shape)
        self.assertEqual(component_count, 2)
        self.assertGreater(int(np.count_nonzero(cleaned)), 0)

    def test_endpoint_stroke_also_keeps_the_generic_cable_layer(self):
        app = self._app()
        app.mode_var = _Variable("endpoint_1")
        item = {"mask": np.zeros((4, 5), dtype=np.uint16), "dirty": False, "negative": False}
        brush = np.zeros((4, 5), dtype=np.uint8)
        brush[2, 3] = 255
        app.apply_brush(item, brush)
        self.assertNotEqual(int(item["mask"][2, 3] & label_bit(1)), 0)
        self.assertNotEqual(int(item["mask"][2, 3] & label_bit(2)), 0)

    def test_prediction_heads_become_independent_editable_layers(self):
        channels = [np.zeros((2, 3), dtype=bool) for _ in range(4)]
        channels[0][0, 0] = True
        channels[1][0, 1] = True
        channels[2][1, 1] = True
        channels[3][1, 2] = True
        draft = prediction_channels_to_draft(channels, (2, 3))
        self.assertNotEqual(int(draft[0, 0] & label_bit(1)), 0)
        for x, semantic_label in ((1, 2), (1, 3), (2, 4)):
            y = 0 if semantic_label == 2 else 1
            self.assertNotEqual(int(draft[y, x] & label_bit(1)), 0)
            self.assertNotEqual(int(draft[y, x] & label_bit(semantic_label)), 0)

    def test_edit_state_restores_draft_provenance_and_verification(self):
        item = {
            "mask": np.zeros((3, 4), dtype=np.uint16),
            "verified": True,
            "negative": False,
            "annotation": {"origin": "manual", "human_verified_at": "now"},
        }
        item["mask"][1, 2] |= label_bit(1)
        state = compact_item_edit_state(item)
        item["mask"].fill(0)
        item["verified"] = False
        item["annotation"] = {"origin": "model_assisted"}
        restore_item_edit_state(item, state)
        self.assertTrue(item["verified"])
        self.assertEqual(item["annotation"]["origin"], "manual")
        self.assertNotEqual(int(item["mask"][1, 2] & label_bit(1)), 0)

    def test_only_a_real_pixel_edit_invalidates_human_verification(self):
        app = self._app()
        app.mode_var = _Variable("paint_1")
        app.verified_var = _Variable(True)
        item = {
            "mask": np.zeros((3, 4), dtype=np.uint16),
            "verified": True,
            "negative": False,
            "dirty": False,
            "annotation": {"origin": "model_assisted", "human_verified_at": "now"},
        }
        brush = np.zeros((3, 4), dtype=np.uint8)
        brush[1, 2] = 255
        item["mask"][1, 2] |= label_bit(1)
        app.apply_brush(item, brush)
        self.assertTrue(item["verified"])
        app.mode_var.set("erase")
        app.apply_brush(item, brush)
        self.assertFalse(item["verified"])
        self.assertFalse(app.verified_var.get())
        self.assertTrue(item["annotation"]["human_edited"])

    def test_annotate_human_verify_button_synchronizes_shared_toggle(self):
        app = self._app()
        item = {
            "session_id": "session_a",
            "verified": False,
            "dirty": False,
            "annotation": {"origin": "model_assisted"},
        }
        app.active_item = lambda: item
        app.verified_var = _Variable(False)
        app.verification_action_var = _Variable("Human Verify")
        app.session_var = _Variable("session_a")
        app.notes_var = _Variable("")
        app.status_var = _Variable("")
        app.refresh = lambda: None

        self.assertTrue(app.toggle_human_verification())
        self.assertTrue(item["verified"])
        self.assertTrue(app.verified_var.get())
        self.assertIn("Human Verified", app.verification_action_var.get())

        self.assertTrue(app.toggle_human_verification())
        self.assertFalse(item["verified"])
        self.assertFalse(app.verified_var.get())
        self.assertEqual(app.verification_action_var.get(), "Human Verify")

    def test_early_stop_status_is_not_overwritten_by_successful_process_exit(self):
        app = self._app()
        app.train_progress_var = _Variable(0.0)
        app.train_summary_var = _Variable("")
        app.training_termination = None
        app.update_training_progress_from_line(
            "epoch 025/120 loss 3.1 val_score 0.01 val_iou 0.02 val_dice 0.03"
        )
        progress_at_stop = app.train_progress_var.get()
        app.update_training_progress_from_line(
            "EARLY STOP: validation score plateaued. Stopped at epoch 25 because val_score did not exceed threshold."
        )
        # Completion markers may be separated visually from the subprocess log.
        app.update_training_progress_from_line("\nTraining finished with exit code 0.")
        self.assertEqual(app.training_termination, "early_stop")
        self.assertEqual(app.train_progress_var.get(), progress_at_stop)
        self.assertIn("epoch 25", app.train_summary_var.get())


if __name__ == "__main__":
    unittest.main()
