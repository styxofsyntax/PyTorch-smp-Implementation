import os
from skimage import io, color
from skimage.util import img_as_float64
import numpy as np
import cv2 as cv
from tqdm import tqdm


def process_mask(mask, area_range=(120, 1000), aspect_ratio_thresh=15.0, solidity_thresh=0.5, compactness_thresh=0.01):
    kernel_dilate = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilated = cv.dilate(mask, kernel_dilate, iterations=1)
    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if not (area_range[0] < area < area_range[1]):
            continue
        keep = True
        if len(cnt) >= 5:
            ellipse = cv.fitEllipse(cnt)
            (_, axes, _) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            if minor_axis == 0:
                continue
            ratio = major_axis / minor_axis
            if ratio > aspect_ratio_thresh:
                keep = False
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < solidity_thresh:
            keep = False
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue
        compactness = area / (perimeter ** 2)
        if compactness < compactness_thresh:
            keep = False
        if keep:
            cv.drawContours(filtered, [cnt], -1, 255, thickness=cv.FILLED)
    return filtered


def get_image_frequency(image_gray):
    laplacian = cv.Laplacian(image_gray, cv.CV_64F)
    freq_score = np.var(laplacian)
    return freq_score


def get_dynamic_cliplimit(image_gray, min_clip=0.5, max_clip=1.0, freq_thresh=0.1):
    mean_val = np.mean(image_gray)
    freq_score = get_image_frequency(image_gray)
    if freq_score > freq_thresh:
        return min_clip
    clip = max_clip - (max_clip - min_clip) * mean_val
    return clip


def apply_clahe_tophat(image_gray, clahe=True, first_blur=1, top_hat_kernal=7, second_blur=5, threshold=(30, 45)):
    img_uint8 = (image_gray * 255).astype(np.uint8)
    img_smooth = cv.GaussianBlur(img_uint8, (first_blur, first_blur), 0)
    if clahe:
        clip_limit = get_dynamic_cliplimit(img_smooth)
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
        img_clahe = clahe.apply(img_uint8)
    else:
        img_clahe = img_uint8
    img_smooth = cv.GaussianBlur(img_clahe, (second_blur, second_blur), 0)
    kernel = cv.getStructuringElement(
        cv.MORPH_RECT, (top_hat_kernal, top_hat_kernal))
    tophat = cv.morphologyEx(img_smooth, cv.MORPH_TOPHAT, kernel)
    binary_tophat = cv.inRange(tophat, threshold[0], threshold[1])
    filtered_tophat = process_mask(binary_tophat)
    return filtered_tophat


root_dir = './data'
splits = ['train', 'test', 'valid']

for split in splits:
    image_dir = os.path.join(root_dir, split, 'images')
    save_dir = os.path.join(root_dir, split, 'pre-processed')
    os.makedirs(save_dir, exist_ok=True)

    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    print(f"\nProcessing '{split}' set ({len(all_files)} images)...")
    for file in tqdm(all_files, desc=f"{split} progress"):
        image_path = os.path.join(image_dir, file)
        save_path = os.path.join(save_dir, file)

        image = io.imread(image_path)
        if image.ndim == 3:
            image_gray = color.rgb2gray(image)
            image_rgb = image
        else:
            image_gray = image
            image_rgb = np.stack([image]*3, axis=-1)

        image_gray = img_as_float64(image_gray)
        image_rgb = img_as_float64(image_rgb)

        filtered_tophat = apply_clahe_tophat(image_gray)
        filter_img = img_as_float64(filtered_tophat)
        filter_img_rgb = np.stack([filter_img] * 3, axis=-1)
        brightness_boost = 1.0 + (filter_img_rgb * 0.25)
        enhanced_img = np.clip(image_rgb * brightness_boost, 0, 1)

        filtered_tophat_2 = apply_clahe_tophat(
            image_gray, first_blur=5, top_hat_kernal=5, second_blur=5)
        filter_img = img_as_float64(filtered_tophat_2)
        filter_img_rgb = np.stack([filter_img] * 3, axis=-1)
        brightness_boost = 1.0 + (filter_img_rgb * 0.6)
        enhanced_img_2 = np.clip(enhanced_img * brightness_boost, 0, 1)

        io.imsave(save_path, (enhanced_img_2 * 255).astype(np.uint8))

print("\nAll splits processed successfully.")
