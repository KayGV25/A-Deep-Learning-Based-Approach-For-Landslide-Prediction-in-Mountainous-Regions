from collections import Counter
import random

import torch
from torch.utils.data import Dataset, WeightedRandomSampler


def make_weighted_random_sampler(y_array, indices, num_classes, seed=None):
    labels = [int(y_array[idx]) for idx in indices]
    counts = Counter(labels)

    if not counts:
        raise ValueError("Cannot build a weighted sampler for an empty dataset.")

    invalid_labels = sorted(label for label in counts if label < 0 or label >= num_classes)
    if invalid_labels:
        raise ValueError(f"Labels outside [0, {num_classes - 1}]: {invalid_labels}")

    sample_weights = [1.0 / counts[label] for label in labels]
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


class MinorityClassAugmentedDataset(Dataset):
    def __init__(self, dataset, labels, augment_fn, augment_probability=1.0):
        if len(dataset) != len(labels):
            raise ValueError("labels must have the same length as dataset.")
        if not 0.0 <= augment_probability <= 1.0:
            raise ValueError("augment_probability must be between 0.0 and 1.0.")

        self.dataset = dataset
        self.augment_fn = augment_fn
        self.augment_probability = augment_probability

        counts = Counter(int(label) for label in labels)
        max_count = max(counts.values()) if counts else 0
        self.minority_classes = {
            label for label, count in counts.items()
            if count < max_count
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if (
            self.augment_fn is not None
            and int(y) in self.minority_classes
            and random.random() < self.augment_probability
        ):
            x = self.augment_fn(x)
        return x, y
