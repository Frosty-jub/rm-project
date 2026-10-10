"""Dataset A backbone, Gaussian MMD, multilevel ensemble and center loss."""

import torch
from torch import nn
from torch.nn import functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, dropout: float = 0.5, architecture: str = "figure3"):
        super().__init__()
        if architecture not in ("figure3", "legacy"):
            raise ValueError(f"unknown architecture: {architecture}")
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 2, padding=1), nn.BatchNorm2d(8), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU())
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU())
        conv4_layers = [nn.Conv2d(32, 16, 3, padding=1)]
        if architecture == "legacy":
            conv4_layers.append(nn.LeakyReLU())
        self.conv4 = nn.Sequential(*conv4_layers)
        self.fc1 = nn.Sequential(nn.Linear(16 * 9 * 5, 128), nn.LeakyReLU(), nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.Dropout(dropout))
        self.fc3 = nn.Sequential(nn.Linear(64, 32), nn.LeakyReLU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.conv4(self.conv3(self.pool(self.conv2(self.conv1(x)))))
        first = self.fc1(x.flatten(1))
        second = self.fc2(first)
        third = self.fc3(second)
        return first, second, third


def gaussian_mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Biased empirical Gaussian-kernel MMD, returned as a distance."""
    pooled = torch.cat((source, target), dim=0)
    squared = torch.cdist(pooled, pooled).square()
    with torch.no_grad():
        positive = squared.detach()[squared.detach() > 0]
        bandwidth = positive.median().clamp_min(1e-6) if positive.numel() else squared.new_tensor(1.0)
    kernel = torch.exp(-squared / bandwidth)
    n = len(source)
    mmd_squared = kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()
    return mmd_squared.clamp_min(1e-10).sqrt()


class MPCCNN(nn.Module):
    def __init__(self, dropout: float = 0.5, architecture: str = "figure3"):
        super().__init__()
        self.architecture = architecture
        self.extractor = FeatureExtractor(dropout, architecture)
        self.classifiers = nn.ModuleList([nn.Linear(size, 6) for size in (128, 64, 32)])

    def logits(self, levels: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(head(F.relu(level) if self.architecture == "figure3" else level)
                     for head, level in zip(self.classifiers, levels))

    @staticmethod
    def weights(source_levels, target_levels):
        distances = torch.stack([gaussian_mmd(s, t) for s, t in zip(source_levels, target_levels)])
        raw = torch.sigmoid(distances).reciprocal()  # paper Eq. 6
        return raw / raw.sum(), distances  # normalize Eq. 7 to valid probabilities

    def probabilities(self, levels, weights):
        predictions = torch.stack([F.softmax(logits, dim=1) for logits in self.logits(levels)])
        return (predictions * weights[:, None, None]).sum(dim=0)


class MultilevelCenterLoss(nn.Module):
    def __init__(self):
        super().__init__()
        for index, dim in enumerate((128, 64, 32)):
            self.register_buffer(f"center_{index}", torch.zeros(6, dim))

    @property
    def centers(self):
        return (self.center_0, self.center_1, self.center_2)

    def forward(self, levels, labels):
        losses = [0.5 * (level - center[labels]).square().sum(dim=1).mean()
                  for level, center in zip(levels, self.centers)]
        return torch.stack(losses).sum()

    @torch.no_grad()
    def update(self, levels, labels, beta: float):
        """Apply the mini-batch class-center update in paper Eq. 10."""
        for level, centers in zip(levels, self.centers):
            for class_id in labels.unique():
                class_index = int(class_id)
                selected = level.detach()[labels == class_id]
                delta = (centers[class_index] - selected).sum(dim=0) / (1 + len(selected))
                centers[class_index] -= beta * delta
