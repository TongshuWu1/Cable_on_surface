import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

from pidnet_dataset import dataset_snapshot, load_dataset_manifest, move_session_split
from pidnet_schema import ANNOTATION_SCHEMA_VERSION, label_bit
from tools.pidnet_training_gui import (
    compact_mask_state,
    make_frame_item,
    read_layered_mask,
    restore_mask_state,
    save_label_pair,
)
from tools.train_pidnet_cable import (
    component_match_counts,
    compute_channel_profile,
    early_stop_summary,
    evaluate,
    focal_bce_loss,
    GenericCableEndpointDataset,
    ModelEMA,
    segmentation_loss,
    validate_resume_checkpoint,
    weighted_harmonic_mean,
)


class PidNetTrainingPipelineTests(unittest.TestCase):
    def test_compact_undo_state_preserves_overlapping_layers(self):
        mask = np.zeros((17, 23), dtype=np.uint16)
        mask[2:12, 3:18] |= label_bit(1)
        mask[4:8, 6:10] |= label_bit(2)
        mask[7:13, 12:16] |= label_bit(4)
        state = compact_mask_state(mask)
        self.assertLess(state[1].nbytes, mask.nbytes)
        np.testing.assert_array_equal(restore_mask_state(state), mask)

    def test_saved_annotation_is_four_layer_and_session_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            image = np.zeros((20, 30, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), image))
            item = make_frame_item(
                image,
                path=source,
                split="train",
                dataset_dir=root,
                cable_count=2,
                session_id="session_alpha",
            )
            item["mask"][4:8, 3:15] |= label_bit(1)
            item["mask"][4:8, 3:6] |= label_bit(2)
            item["verified"] = True
            item["annotation"] = {
                "origin": "model_assisted",
                "checkpoint": {"sha256": "abc123"},
                "human_verified_at": "2026-01-01T00:00:00+00:00",
            }
            image_path, mask_path = save_label_pair(item, root)

            layered_path = root / "masks_layers" / "train" / f"{mask_path.stem}.npz"
            with np.load(layered_path, allow_pickle=False) as payload:
                self.assertEqual(int(payload["annotation_schema_version"]), ANNOTATION_SCHEMA_VERSION)
            loaded = read_layered_mask(layered_path, cable_count=2)
            np.testing.assert_array_equal(loaded, item["mask"])
            metadata = load_dataset_manifest(root)["items"][image_path.stem]
            self.assertEqual(metadata["session_id"], "session_alpha")
            self.assertTrue(metadata["verified"])
            self.assertEqual(metadata["annotation"]["origin"], "model_assisted")
            self.assertEqual(metadata["annotation"]["checkpoint"]["sha256"], "abc123")

            profile = compute_channel_profile([(image_path, mask_path)], cable_count=2)
            self.assertEqual(profile["positive_frames"], [1, 1, 0, 0])
            self.assertEqual(profile["both_endpoint_frames"], 0)
            image_tensor, mask_tensor, boundary_tensor = GenericCableEndpointDataset(
                [(image_path, mask_path)],
                image_size=(30, 20),
                augment=False,
                cable_count=2,
            )[0]
            self.assertEqual(image_tensor.dtype, torch.uint8)
            self.assertEqual(mask_tensor.dtype, torch.uint8)
            self.assertEqual(boundary_tensor.dtype, torch.uint8)

    def test_session_move_moves_every_saved_artifact_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((12, 16, 3), dtype=np.uint8)
            for index in range(2):
                source = root / f"frame_{index}.png"
                self.assertTrue(cv2.imwrite(str(source), image))
                item = make_frame_item(
                    image,
                    path=source,
                    split="train",
                    dataset_dir=root,
                    cable_count=2,
                    session_id="same_session",
                )
                item["mask"][2:5, 2:8] |= label_bit(1)
                item["verified"] = True
                save_label_pair(item, root)
            item_count, move_count = move_session_split(root, "same_session", "val")
            self.assertEqual(item_count, 2)
            self.assertEqual(move_count, 6)
            self.assertEqual(len(list((root / "images" / "val").glob("*.png"))), 2)
            self.assertFalse(any((root / "images" / "train").glob("*.png")))
            manifest = load_dataset_manifest(root)
            self.assertTrue(all(item["split"] == "val" for item in manifest["items"].values()))

    def test_same_named_external_images_receive_distinct_dataset_stems(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_source = root / "source_a" / "frame.png"
            second_source = root / "source_b" / "frame.png"
            first_source.parent.mkdir()
            second_source.parent.mkdir()
            self.assertTrue(cv2.imwrite(str(first_source), np.zeros((12, 16, 3), dtype=np.uint8)))
            self.assertTrue(cv2.imwrite(str(second_source), np.full((12, 16, 3), 255, dtype=np.uint8)))
            first = make_frame_item(cv2.imread(str(first_source)), path=first_source, dataset_dir=root, session_id="a")
            first["mask"][2:5, 2:8] |= label_bit(1)
            first["verified"] = True
            first_image, _first_mask = save_label_pair(first, root)
            second = make_frame_item(cv2.imread(str(second_source)), path=second_source, dataset_dir=root, session_id="b")
            second["mask"][2:5, 2:8] |= label_bit(1)
            second["verified"] = True
            second_image, _second_mask = save_label_pair(second, root)
            self.assertNotEqual(first_image.stem, second_image.stem)
            self.assertEqual(len(list((root / "images" / "train").glob("*.png"))), 2)
            self.assertEqual(len(load_dataset_manifest(root)["items"]), 2)

    def test_dataset_snapshot_hash_includes_selected_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            image = np.zeros((12, 16, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(source), image))
            item = make_frame_item(
                image,
                path=source,
                split="train",
                dataset_dir=root,
                cable_count=2,
                session_id="snapshot_session",
            )
            item["mask"][2:5, 2:8] |= label_bit(1)
            item["verified"] = True
            image_path, mask_path = save_label_pair(item, root)
            pairs = {"train": [(image_path, mask_path)], "val": [], "test": []}
            before = dataset_snapshot(root, pairs)

            item["notes"] = "changed research note"
            save_label_pair(item, root)
            after = dataset_snapshot(root, pairs)

            self.assertNotEqual(before["metadata_sha256"], after["metadata_sha256"])
            self.assertNotEqual(before["dataset_sha256"], after["dataset_sha256"])

    def test_each_semantic_head_has_an_independent_loss(self):
        logits = torch.zeros((1, 4, 8, 8), dtype=torch.float32, requires_grad=True)
        boundary_logits = torch.zeros((1, 1, 8, 8), dtype=torch.float32, requires_grad=True)
        target_a = torch.zeros_like(logits)
        target_b = target_a.clone()
        target_a[:, 0, 2:6, 2:6] = 1.0
        target_b.copy_(target_a)
        target_b[:, 1, 1:3, 1:3] = 1.0
        boundary = torch.zeros_like(boundary_logits)
        arguments = dict(
            channel_alpha=np.asarray((0.6, 0.9, 0.9, 0.9)),
            endpoint_weight=2.0,
            crossing_weight=1.0,
            boundary_weight=0.2,
            focal_gamma=2.0,
            dice_weight=1.0,
        )
        _loss_a, heads_a = segmentation_loss(
            {"seg": logits, "boundary": boundary_logits}, target_a, boundary, **arguments
        )
        loss_b, heads_b = segmentation_loss(
            {"seg": logits, "boundary": boundary_logits}, target_b, boundary, **arguments
        )
        self.assertAlmostEqual(heads_a["body"], heads_b["body"], places=7)
        self.assertNotAlmostEqual(heads_a["endpoint1"], heads_b["endpoint1"], places=5)
        loss_b.backward()
        self.assertGreater(float(torch.sum(torch.abs(logits.grad[:, 1]))), 0.0)

    def test_focal_loss_normalizes_extreme_class_balance(self):
        logits = torch.zeros((1, 1, 1, 1000), dtype=torch.float32)
        rare_target = torch.zeros_like(logits)
        rare_target[..., 0] = 1.0
        balanced_target = torch.zeros_like(logits)
        balanced_target[..., :500] = 1.0
        rare = focal_bce_loss(logits, rare_target, alpha=0.999, gamma=2.0)
        balanced = focal_bce_loss(logits, balanced_target, alpha=0.5, gamma=2.0)
        self.assertAlmostEqual(float(rare), float(balanced), places=6)

    def test_selection_score_tracks_progress_before_all_heads_detect_components(self):
        failed = weighted_harmonic_mean((0.0, 0.0, 0.0, 0.0), (0.35, 0.25, 0.25, 0.15))
        body_progress = weighted_harmonic_mean((0.25, 0.0, 0.0, 0.0), (0.35, 0.25, 0.25, 0.15))
        balanced = weighted_harmonic_mean((0.25, 0.25, 0.25, 0.25), (0.35, 0.25, 0.25, 0.15))
        self.assertEqual(failed, 0.0)
        self.assertGreater(body_progress, failed)
        self.assertGreater(balanced, body_progress)

    def test_early_stop_summary_explains_patience_window_and_threshold(self):
        metrics = {
            "body_iou": 0.40,
            "endpoint1_quality": 0.30,
            "endpoint2_quality": 0.20,
            "crossing_quality": 0.10,
        }
        summary = early_stop_summary(
            epoch=25,
            best_epoch=1,
            best_score=0.1234,
            current_score=0.1200,
            patience=24,
            min_delta=0.0001,
            metrics=metrics,
        )
        self.assertEqual(summary["type"], "validation_plateau")
        self.assertEqual(summary["plateau_start_epoch"], 2)
        self.assertEqual(summary["stopped_epoch"], 25)
        self.assertAlmostEqual(summary["required_score"], 0.1235)
        self.assertEqual(summary["crossing_quality"], 0.10)

    def test_component_assignment_counts_false_components_and_centroid_error(self):
        target = np.zeros((80, 120), dtype=np.uint8)
        predicted = np.zeros_like(target)
        cv2.circle(target, (20, 25), 4, 1, -1)
        cv2.circle(target, (90, 55), 4, 1, -1)
        cv2.circle(predicted, (22, 25), 4, 1, -1)
        cv2.circle(predicted, (90, 58), 4, 1, -1)
        cv2.circle(predicted, (60, 20), 4, 1, -1)
        true_positive, false_positive, false_negative, error_sum = component_match_counts(
            predicted, target, max_distance_px=8.0
        )
        self.assertEqual((true_positive, false_positive, false_negative), (2, 1, 0))
        self.assertAlmostEqual(error_sum, 5.0, places=4)

    def test_component_assignment_rejects_centered_oversized_blob(self):
        target = np.zeros((100, 100), dtype=np.uint8)
        predicted = np.ones_like(target)
        cv2.circle(target, (50, 50), 5, 1, -1)
        counts = component_match_counts(predicted, target, max_distance_px=5.0)
        self.assertEqual(counts[:3], (0, 1, 1))

    def test_component_assignment_counts_false_components_beyond_matching_cap(self):
        target = np.zeros((120, 240), dtype=np.uint8)
        predicted = np.zeros_like(target)
        cv2.circle(target, (10, 10), 2, 1, -1)
        cv2.circle(predicted, (10, 10), 2, 1, -1)
        for index in range(40):
            x = 20 + (index % 20) * 11
            y = 30 + (index // 20) * 30
            cv2.circle(predicted, (x, y), 2, 1, -1)
        true_positive, false_positive, false_negative, _error = component_match_counts(
            predicted,
            target,
            max_distance_px=4.0,
        )
        self.assertEqual((true_positive, false_positive, false_negative), (1, 40, 0))

    def test_ema_warmup_forgets_random_initialization_on_small_datasets(self):
        model = torch.nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ema = ModelEMA(model, decay=0.995)
        with torch.no_grad():
            model.weight.fill_(1.0)
        for _ in range(4):
            ema.update(model)
        self.assertEqual(ema.updates, 4)
        self.assertLess(ema.effective_decay, ema.decay)
        self.assertGreater(float(ema.model.weight.item()), 0.99)

    def test_exact_resume_rejects_a_changed_dataset(self):
        args = SimpleNamespace(
            epochs=120,
            batch_size=8,
            lr=1e-3,
            weight_decay=1e-4,
            endpoint_weight=2.0,
            crossing_weight=1.0,
            boundary_weight=0.2,
            focal_gamma=2.0,
            dice_weight=1.0,
            ema_decay=0.995,
            seed=17,
        )
        payload = {
            "dataset_sha256": "old",
            "training": {key: value for key, value in vars(args).items()},
        }
        with self.assertRaisesRegex(ValueError, "dataset does not match"):
            validate_resume_checkpoint(payload, args, "new")

    def test_locked_evaluation_uses_fixed_validation_thresholds(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, image):
                probability = torch.full(
                    (image.shape[0], 4, image.shape[2], image.shape[3]),
                    0.60,
                    dtype=image.dtype,
                    device=image.device,
                )
                return {
                    "seg": torch.logit(probability),
                    "boundary": torch.zeros(
                        (image.shape[0], 1, image.shape[2], image.shape[3]),
                        dtype=image.dtype,
                        device=image.device,
                    ),
                }

        batch = (
            torch.zeros((1, 3, 16, 16), dtype=torch.float32),
            torch.zeros((1, 4, 16, 16), dtype=torch.float32),
            torch.zeros((1, 1, 16, 16), dtype=torch.float32),
        )
        metrics = evaluate(
            ConstantModel(),
            [batch],
            torch.device("cpu"),
            threshold_grid=(0.5, 0.7),
            fixed_thresholds=(0.7, 0.7, 0.7, 0.7),
        )
        np.testing.assert_allclose(metrics["thresholds"], (0.7, 0.7, 0.7, 0.7))


if __name__ == "__main__":
    unittest.main()
