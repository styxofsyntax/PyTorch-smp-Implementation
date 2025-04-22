import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import os

# Load config & model
cfg = yaml.safe_load(open("model_config.yaml", "r"))
model_cfg = cfg["model"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.DeepLabV3Plus(**{
    "encoder_name": model_cfg["encoder_name"],
    "encoder_weights": model_cfg["encoder_weights"],
    "in_channels": model_cfg["in_channels"],
    "classes": model_cfg["num_classes"],
    "activation": model_cfg["activation"]
}).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])


def predict_image(image_path, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)
        if model_cfg["activation"] is None:
            pred = torch.sigmoid(pred)

    mask = (pred > threshold).float().cpu().numpy()[0]  # shape [C, H, W]
    return mask, img.resize((640, 640))


def visualize_prediction(image, mask):
    # Define class colors: 0 = black, 1 = blue (card), 2 = green (damage)
    color_map = {
        0: [0, 0, 0],       # background
        1: [0, 0, 255],     # card
        2: [0, 255, 0],     # damage
    }

    h, w = mask.shape[1], mask.shape[2]
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls in range(mask.shape[0]):
        rgb_mask[mask[cls] > 0.5] = color_map[cls]

    overlay = np.array(image).copy()
    for cls in [1, 2]:  # overlay card and damage
        color = np.array(color_map[cls], dtype=np.uint8)
        binary = mask[cls] > 0.5
        overlay[binary] = overlay[binary] * 0.5 + color * 0.5

    # Prepare class masks
    class_masks = [(mask[cls] * 255).astype(np.uint8)
                   for cls in range(mask.shape[0])]

    # Plot layout: 2 rows, 3 columns
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    # Row 1: original, rgb mask, overlay
    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(rgb_mask)
    axs[0, 1].set_title("Segmentation Mask (RGB)")
    axs[0, 2].imshow(overlay.astype(np.uint8))
    axs[0, 2].set_title("Overlay (Card & Damage)")

    # Row 2: per-class grayscale masks
    cls_titles = ["Background", "Card", "Damage"]
    for i in range(3):
        axs[1, i].imshow(class_masks[i], cmap="gray")
        axs[1, i].set_title(f"{cls_titles[i]} Mask")

    # Tidy layout
    for ax in axs.flatten():
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = sys.argv[1]
    mask, original_img = predict_image(image_path)

    # Save individual masks
    os.makedirs("output_masks", exist_ok=True)
    for cls in range(mask.shape[0]):
        Image.fromarray((mask[cls] * 255).astype(np.uint8)).save(
            f"output_masks/class_{cls}.png")

    # Show visualization
    visualize_prediction(original_img, mask)
