import os
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms

# =================== Config ===================
plot_name = "comparison-14"
split = "test"
image_name = "IMG_9652_JPG.rf.6d371dab96baa70677d6511eb68af1e5"
image_path_preprocess = f"data/{split}/pre-processed/{image_name}.jpg"
image_path_raw = f"data/{split}/images/{image_name}.jpg"
label_path = f"data/{split}/labels/{image_name}.txt"
model_path = "best_model.pth"  # your trained U-Net model

# Load model config
cfg = yaml.safe_load(open("model_config.yaml", "r"))
model_cfg = cfg["model"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name=model_cfg["encoder_name"],
    encoder_weights=model_cfg["encoder_weights"],
    in_channels=model_cfg["in_channels"],
    classes=model_cfg["num_classes"],
    activation=None,  # important: model output raw logits
).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# =================== Load Image ===================
original = cv2.imread(image_path_raw)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
h, w, _ = original.shape

# =================== Original Annotations ===================
polygon_image = original.copy()
overlay_poly = np.zeros_like(polygon_image, dtype=np.uint8)

with open(label_path, "r") as f:
    labels = f.readlines()

card_polygons = []
damage_polygons = []

for label in labels:
    parts = label.strip().split()
    cls = int(parts[0])
    coords = list(map(float, parts[1:]))
    points = np.array(coords, dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= w
    points[:, 1] *= h
    points = points.astype(np.int32)

    if cls == 0:
        card_polygons.append(points)
    elif cls == 1:
        damage_polygons.append(points)

for points in card_polygons:
    cv2.fillPoly(overlay_poly, [points], color=(0, 0, 255))
for points in damage_polygons:
    cv2.fillPoly(overlay_poly, [points], color=(255, 0, 0))

alpha = 0.5
blended_poly = cv2.addWeighted(
    polygon_image, 1 - alpha, overlay_poly, alpha, 0)

# =================== SMP U-Net Prediction ===================
# Load pre-processed image
preproc_image = Image.open(image_path_preprocess).convert("RGB")
input_tensor = transform(preproc_image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float().cpu().numpy()[0]  # shape [C, H, W]

# Create overlay
overlay_mask = np.zeros_like(original, dtype=np.uint8)

# Assume class 0 = background, 1 = card, 2 = damage
# Card: Blue, Damage: Green
for cls_idx, color in zip([1, 2], [(255, 0, 0), (0, 0, 255)]):
    mask = preds[cls_idx]
    overlay_mask[mask > 0.5] = color

blended_mask = cv2.addWeighted(original, 1 - alpha, overlay_mask, alpha, 0)

# =================== Plot Comparison ===================
fig, axs = plt.subplots(1, 2, figsize=(16, 16))

axs[0].imshow(blended_poly)
axs[0].set_title("Original Annotations")
axs[0].axis("off")

axs[1].imshow(blended_mask)
axs[1].set_title("U-Net Prediction on Preprocessed Image")
axs[1].axis("off")

plt.tight_layout()
# plt.savefig(f'./comparison/{plot_name}.png')
plt.show()
