"""Focused checks for data shape, transfer isolation, and trainable losses."""

import unittest
from pathlib import Path

import numpy as np
import torch

from mpc_cnn.data import RAW_ID_TO_PAPER_GAS, load_batch, scale_from_source
from mpc_cnn.model import MPCCNN, MultilevelCenterLoss, gaussian_mmd
from mpc_cnn.train import training_weights


class PipelineTests(unittest.TestCase):
    def test_real_batch_schema_and_source_scaler(self):
        source, labels = load_batch(Path("dataset/batch1.dat"))
        target, _ = load_batch(Path("dataset/batch2.dat"))
        xs, xt, mean, std = scale_from_source(source, target)
        self.assertEqual(xs.shape, (445, 1, 16, 8))
        self.assertEqual(xt.shape, (1244, 1, 16, 8))
        self.assertEqual(set(np.unique(labels)), set(range(6)))
        self.assertTrue(np.allclose(xs.reshape(445, 128).mean(0), 0, atol=1e-5))
        self.assertTrue(np.isfinite(xt).all())
        self.assertTrue(np.all(std > 0))

    def test_raw_class_order_matches_paper_table_2_counts(self):
        paper_gas_order = ("Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene")
        expected = {1: [83, 30, 70, 98, 90, 74], 2: [100, 109, 532, 334, 164, 5]}
        gas_to_raw = {gas: raw_id - 1 for raw_id, gas in RAW_ID_TO_PAPER_GAS.items()}
        for batch_id, counts in expected.items():
            _, labels = load_batch(Path(f"dataset/batch{batch_id}.dat"))
            observed = np.bincount(labels, minlength=6)
            self.assertEqual([int(observed[gas_to_raw[gas]]) for gas in paper_gas_order], counts)

    def test_backbone_mmd_and_center_loss_have_finite_gradients(self):
        torch.manual_seed(1)
        model = MPCCNN(dropout=0.0)
        centers = MultilevelCenterLoss()
        xs = torch.randn(8, 1, 16, 8)
        xt = torch.randn(8, 1, 16, 8)
        labels = torch.arange(8) % 6
        source_levels = model.extractor(xs)
        target_levels = model.extractor(xt)
        self.assertEqual([tuple(level.shape) for level in source_levels],
                         [(8, 128), (8, 64), (8, 32)])
        weights, distances = model.weights(source_levels, target_levels)
        self.assertTrue(torch.isfinite(distances).all())
        self.assertAlmostEqual(weights.sum().item(), 1, places=5)
        probabilities = model.probabilities(source_levels, weights.detach())
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(8), atol=1e-5))
        loss = -probabilities[torch.arange(8), labels].clamp_min(1e-10).log().mean()
        loss = loss + centers(source_levels, labels) + gaussian_mmd(source_levels[0], target_levels[0])
        loss.backward()
        self.assertTrue(torch.isfinite(model.extractor.conv1[0].weight.grad).all())
        centers.update(source_levels, labels, beta=0.5)
        self.assertNotEqual(centers.center_0.abs().sum().item(), 0)

    def test_figure3_activation_layout_and_legacy_option(self):
        model = MPCCNN(dropout=0.0)
        legacy = MPCCNN(dropout=0.0, architecture="legacy")
        self.assertEqual(len(model.extractor.conv4), 1)
        self.assertEqual(len(legacy.extractor.conv4), 2)
        with torch.no_grad():
            for head in (model.classifiers[0], legacy.classifiers[0]):
                head.weight.fill_(1.0)
                head.bias.zero_()
        levels = (torch.full((1, 128), -1.0), torch.zeros(1, 64), torch.zeros(1, 32))
        self.assertTrue(torch.equal(model.logits(levels)[0], torch.zeros(1, 6)))
        self.assertTrue(torch.equal(legacy.logits(levels)[0], torch.full((1, 6), -128.0)))

    def test_unlabeled_target_does_not_overwrite_batchnorm_statistics(self):
        torch.manual_seed(2)
        model = MPCCNN(dropout=0.0)
        model.train()
        source_levels = model.extractor(torch.randn(8, 1, 16, 8))
        running_mean = model.extractor.conv1[1].running_mean.clone()
        weights = training_weights(model, source_levels, torch.randn(8, 1, 16, 8) + 10)
        self.assertTrue(model.extractor.training)
        self.assertTrue(torch.equal(running_mean, model.extractor.conv1[1].running_mean))
        self.assertAlmostEqual(weights.sum().item(), 1, places=5)


if __name__ == "__main__":
    unittest.main()
