from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


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
