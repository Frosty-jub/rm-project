"""Checks for the controlled MPC-CNN runner and saved inference contract."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from mpc_cnn.model import MultilevelCenterLoss
from mpc_cnn.revised import (Recipe, RevisedMPC, center_loss, mmd_weights,
                             predict_checkpoint, prepare, revised_mmd, run,
                             task_parameters, transform_with_saved)
from scripts.run_revised_suite import LONG, SHORT, select_pairs, summarize, write_progress


class RevisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_task_and_recipe_validation(self):
        self.assertEqual(task_parameters(1, 2)[1], 85.06)
        self.assertEqual(task_parameters(2, 3)[1], 92.88)
        for pair in ((0, 2), (2, 4), (10, 11), (1, 1)):
            with self.assertRaises(ValueError):
                task_parameters(*pair)
        for change in ({"head": "unknown"}, {"epochs": 0}, {"batch_size": -1},
                       {"fixed_bandwidth": 0}, {"dropout": 1.0}):
            with self.assertRaises(ValueError):
                replace(Recipe(), **change).validate()

    def test_consistent_eval_mmd_of_identical_domains(self):
        torch.manual_seed(3)
        recipe = Recipe(dropout=0.0)
        model = RevisedMPC(recipe)
        model.train()
        data = torch.randn(8, 1, 16, 8)
        train_levels = model.extractor(data)
        weights = mmd_weights(model, data, data.clone(), train_levels, recipe)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        model.eval()
        levels = model.extractor(data)
        _, distances = model.weights(levels, levels)
        self.assertTrue(torch.all(distances < 1e-3))
        self.assertTrue(model.training is False)

    def test_center_sum_and_bandwidth_gradients(self):
        torch.manual_seed(4)
        centers = MultilevelCenterLoss()
        levels = (torch.randn(5, 128), torch.randn(5, 64), torch.randn(5, 32))
        labels = torch.tensor([0, 1, 2, 3, 4])
        mean = center_loss(centers, levels, labels, Recipe(center_reduction="mean"))
        summed = center_loss(centers, levels, labels, Recipe(center_reduction="sum"))
        self.assertTrue(torch.allclose(summed, mean * 5))
        source = torch.randn(5, 7, requires_grad=True)
        target = torch.randn(5, 7)
        loss = revised_mmd(source, target, Recipe(bandwidth_mode="fixed", fixed_bandwidth=4.0))
        loss.backward()
        self.assertTrue(torch.isfinite(source.grad).all())
        self.assertGreater(source.grad.abs().sum().item(), 0)

    def test_ensemble_weight_gradient_and_normalization(self):
        torch.manual_seed(8)
        xs = torch.randn(8, 1, 16, 8, requires_grad=True)
        xt = torch.randn(8, 1, 16, 8) + 1
        recipe = Recipe(dropout=0, ensemble_gradient="differentiable",
                        bandwidth_mode="fixed", fixed_bandwidth=100.0)
        model = RevisedMPC(recipe)
        model.train()
        sl = model.extractor(xs)
        weights = mmd_weights(model, xs, xt, sl, recipe)
        self.assertTrue(weights.requires_grad)
        gradient = torch.autograd.grad(weights[0], xs, retain_graph=True)[0]
        self.assertGreater(gradient.abs().sum().item(), 0)
        detached = mmd_weights(model, xs, xt, sl, replace(recipe, ensemble_gradient="detached"))
        self.assertFalse(detached.requires_grad)

        logits = torch.randn(3, 5, 6, requires_grad=True)
        weights = torch.tensor([0.9, 0.7, 0.6])
        classes = torch.tensor([0, 1, 2, 3, 4])
        probabilities = logits.softmax(-1)
        raw = (probabilities * weights[:, None, None]).sum(0)
        normalized = raw / weights.sum()
        raw_grad = torch.autograd.grad(-raw[torch.arange(5), classes].log().mean(), logits,
                                       retain_graph=True)[0]
        normalized_grad = torch.autograd.grad(-normalized[torch.arange(5), classes].log().mean(), logits)[0]
        self.assertTrue(torch.equal(raw.argmax(1), normalized.argmax(1)))
        self.assertTrue(torch.allclose(raw_grad, normalized_grad, atol=1e-7))

    def test_suite_selection_and_macro(self):
        self.assertEqual(len(select_pairs("all")), 17)
        self.assertEqual(select_pairs("representative"), ((1, 2), (1, 6), (4, 5)))
        self.assertEqual(select_pairs(pairs="1:2,1:2,4:5"), ((1, 2), (4, 5)))
        with self.assertRaises(ValueError):
            select_pairs(pairs="2:4")
        rows = [{"source": 1, "target": 2, "target_accuracy_percent": 50,
                 "paper_accuracy_percent": 85.06},
                {"source": 1, "target": 6, "target_accuracy_percent": 70,
                 "paper_accuracy_percent": 95.16},
                {"source": 4, "target": 5, "target_accuracy_percent": 90,
                 "paper_accuracy_percent": 94.71}]
        summary = summarize(rows, select_pairs("representative"), Recipe(), "exploratory")
        self.assertEqual(summary["protocols"]["long"]["macro_target_accuracy_percent"], 60)
        self.assertEqual(summary["protocols"]["short"]["macro_target_accuracy_percent"], 70)
        self.assertEqual(summary["protocols"]["long"]["task_count"], 2)
        self.assertEqual(len(LONG), len(SHORT))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            partial = write_progress(output, rows[:1], select_pairs("representative"),
                                     Recipe(), "exploratory")
            self.assertEqual(partial["requested_pair_count"], 3)
            self.assertEqual(partial["completed_count"], 1)
            self.assertEqual(partial["status"], "incomplete")
            self.assertEqual(len((output / "tasks.csv").read_text().splitlines()), 2)

    def test_preprocessing_replay_and_label_isolation(self):
        rng = np.random.default_rng(17)
        source = rng.normal(size=(12, 1, 16, 8)).astype(np.float32)
        target = (rng.normal(size=(12, 1, 16, 8)) + 0.2).astype(np.float32)
        source_labels = np.arange(12, dtype=np.int64) % 6
        target_labels = (np.arange(12, dtype=np.int64) + 1) % 6
        recipe = Recipe(seed=7, pretrain_epochs=0, epochs=1, batch_size=12,
                        log_every=1, center_multiplier=0, row_l2=True,
                        transform="signed_log", scaling="source", dropout=0.0,
                        bandwidth_mode="fixed", fixed_bandwidth=2.0)
        xs, xt, stats = prepare(source, target, recipe)
        self.assertTrue(np.allclose(xs, transform_with_saved(source, stats, recipe, "source")))
        self.assertTrue(np.allclose(xt, transform_with_saved(target, stats, recipe, "target")))
        self.assertTrue(np.allclose(np.linalg.norm(xt.reshape(12, -1), axis=1), 1))

        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            data_dir.mkdir()
            for batch in (1, 2):
                (data_dir / f"batch{batch}.dat").write_text(f"synthetic {batch}")

            def load(path):
                return (source, source_labels) if path.stem == "batch1" else (target, target_labels)

            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            with patch("mpc_cnn.revised.load_batch", side_effect=load):
                result = run(recipe, 1, 2, first, data_dir, torch.device("cpu"))
                reused = run(recipe, 1, 2, first, data_dir, torch.device("cpu"))
                self.assertEqual(result, reused)
                with patch("mpc_cnn.revised.platform.python_version", return_value="0.0"):
                    with self.assertRaises(ValueError):
                        run(recipe, 1, 2, first, data_dir, torch.device("cpu"))
                probabilities = predict_checkpoint(first / "model.pt", target)
                self.assertEqual(np.bincount(probabilities.argmax(1), minlength=6).tolist(),
                                 result["metrics"]["target_predicted_counts"])
                self.assertAlmostEqual(float(np.mean(probabilities.argmax(1) == target_labels)),
                                       result["metrics"]["target_accuracy"], places=6)
                with self.assertRaises(ValueError):
                    run(replace(recipe, head="leaky"), 1, 2, first, data_dir, torch.device("cpu"))
                invalid_pretrain_output = Path(temporary) / "invalid_pretrain"
                with self.assertRaises(ValueError):
                    run(replace(recipe, head="leaky"), 1, 2, invalid_pretrain_output,
                        data_dir, torch.device("cpu"), pretrained=first / "pretrained.pt")
                self.assertFalse(invalid_pretrain_output.exists())

            changed_labels = (target_labels + 2) % 6

            def changed(path):
                return (source, source_labels) if path.stem == "batch1" else (target, changed_labels)

            with patch("mpc_cnn.revised.load_batch", side_effect=changed):
                with self.assertRaises(ValueError):
                    run(recipe, 1, 2, first, data_dir, torch.device("cpu"))
                run(recipe, 1, 2, second, data_dir, torch.device("cpu"))
            a = torch.load(first / "model.pt", map_location="cpu", weights_only=True)["model_state"]
            b = torch.load(second / "model.pt", map_location="cpu", weights_only=True)["model_state"]
            self.assertTrue(all(torch.equal(a[key], b[key]) for key in a))

    def test_incomplete_output_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "history.jsonl").write_text("partial")
            with patch("mpc_cnn.revised.load_batch", return_value=(np.zeros((2, 1, 16, 8)), np.zeros(2, dtype=np.int64))):
                data = output / "data"
                data.mkdir()
                for batch in (1, 2):
                    (data / f"batch{batch}.dat").write_text("dummy")
                with self.assertRaises(FileExistsError):
                    run(Recipe(pretrain_epochs=0, epochs=1), 1, 2, output, data, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
