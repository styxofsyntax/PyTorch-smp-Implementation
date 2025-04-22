import os
import numpy as np
from PIL import Image
import torch


def compute_pos_weights(mask_dir, num_classes):
    total_positive = torch.zeros(num_classes)
    total_pixels = 0

    for fname in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, fname)
        mask = Image.open(mask_path).resize((256, 256), resample=Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)

        total_pixels += mask_np.shape[0] * mask_np.shape[1]

        for cls in range(num_classes):
            total_positive[cls] += np.sum(mask_np == cls)

    total_negative = total_pixels - total_positive
    pos_weight = total_negative / (total_positive + 1e-6)  # avoid div by zero
    return pos_weight


# Usage
num_classes = 3
mask_path = "./data/test/masks"
pos_weight = compute_pos_weights(mask_path, num_classes)
print("pos_weight tensor:", pos_weight)
