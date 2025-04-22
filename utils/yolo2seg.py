import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def yolo_polygon_to_mask(image_path, label_path, mask_path):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 3 or len(parts[1:]) % 2 != 0:
            continue  # skip malformed annotations
        class_id = int(parts[0])
        coords = parts[1:]

        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i + 1] * height)
            points.append([x, y])
        points = np.array([points], dtype=np.int32)

        # class_id+1 to keep 0 as background
        cv2.fillPoly(mask, points, color=class_id + 1)

    cv2.imwrite(str(mask_path), mask)


def process_split(split_dir):
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'
    masks_dir = split_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)

    image_files = list(images_dir.glob('*.jpg')) + \
        list(images_dir.glob('*.png'))

    for image_file in tqdm(image_files, desc=f'Processing {split_dir.name}'):
        label_file = labels_dir / (image_file.stem + '.txt')
        mask_file = masks_dir / (image_file.stem + '.png')

        if label_file.exists():
            yolo_polygon_to_mask(image_file, label_file, mask_file)


def main(root_dir):
    for split in ['train', 'valid', 'test']:
        split_dir = Path(root_dir) / split
        process_split(split_dir)


if __name__ == '__main__':
    main('./data')
