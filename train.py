import numpy as np
import os
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
# from torchmetrics import IoU, F1Score, DiceScore
from torchmetrics import F1Score
from torchmetrics.segmentation import DiceScore
from torchvision import transforms
from PIL import Image

with open("model_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
train_cfg = cfg["training"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, img_size=(256, 256)):
        self.image_paths = sorted(os.listdir(images_dir))
        self.mask_paths = sorted(os.listdir(masks_dir))
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_size = img_size
        self.num_classes = model_cfg["num_classes"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(
            self.images_dir, self.image_paths[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir,  self.mask_paths[idx]))

        img = img.resize(self.img_size, resample=Image.BILINEAR)
        mask = mask.resize(self.img_size, resample=Image.NEAREST)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # one-hot encoding
        mask_np = np.array(mask, dtype=np.int64)
        one_hot_masks = []
        for cls in range(self.num_classes):
            cls_map = (mask_np == cls).astype(np.float32)
            one_hot_masks.append(torch.from_numpy(cls_map))
        mask_tensor = torch.stack(one_hot_masks, dim=0)

        return img, mask_tensor.float()


tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_ds = SegmentationDataset(
    train_cfg["train_image_path"], train_cfg["train_mask_path"], transform=tf)
val_ds = SegmentationDataset(
    train_cfg["valid_image_path"], train_cfg["valid_mask_path"], transform=tf)

train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                          shuffle=True, num_workers=train_cfg["num_workers"])
valid_loader = DataLoader(val_ds,   batch_size=train_cfg["batch_size"],
                          shuffle=False, num_workers=train_cfg["num_workers"])  # no shuffle for val :contentReference[oaicite:6]{index=6}

model = smp.Unet(
    encoder_name=model_cfg["encoder_name"],
    encoder_weights=model_cfg["encoder_weights"],
    in_channels=model_cfg["in_channels"],
    classes=model_cfg["num_classes"],
    activation=model_cfg["activation"],
).to(device)

# loss function
pos_weight = torch.tensor(
    train_cfg["pos_weight"], dtype=torch.float32).to(device)
pos_weight = pos_weight.view(-1, 1, 1)
criterion = nn.MultiLabelSoftMarginLoss(weight=pos_weight)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# metric_iou = IoU(
# num_classes=model_cfg["num_classes"], ignore_index=0).to(device)
metric_f1 = F1Score(
    task="multilabel",              # specify multilabel classification
    num_labels=3,                   # three one-hot channels
    average="macro",                # macro-average across labels
).to(device)

metric_dice = DiceScore(
    num_classes=3,                  # number of classes
    include_background=True,        # include class 0 if desired
    average="macro",                # macro-average over classes
    input_format="one-hot"          # expects one-hot tensors
).to(device)

optimizer = optim.Adam(
    model.parameters(), lr=train_cfg["learning_rate"])
scheduler = ReduceLROnPlateau(optimizer, mode="min",
                              factor=train_cfg["scheduler_factor"],
                              patience=train_cfg["scheduler_patience"])


class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


early_stopper = EarlyStopping(train_cfg["early_stopping_patience"],
                              train_cfg["early_stopping_delta"])


for epoch in range(train_cfg["num_epochs"]):
    print(f"\nEpoch {epoch+1}/{train_cfg['num_epochs']}")

    # Training
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc="  Train", leave=False)
    for imgs, masks in train_bar:
        imgs, masks = imgs.to(device), masks.to(device).float()
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix(loss=train_loss / (train_bar.n + 1))

    # Validation
    model.eval()
    val_loss = 0
    metric_f1.reset()
    metric_dice.reset()
    val_bar = tqdm(valid_loader, desc="  Valid", leave=False)
    with torch.no_grad():
        for imgs, masks in val_bar:
            imgs, masks = imgs.to(device), masks.to(device).float()
            preds = model(imgs)
            val_loss += criterion(preds, masks).item()
            metric_f1.update(preds, masks.long())
            metric_dice.update(preds, masks.long())

            val_bar.set_postfix(loss=val_loss / (val_bar.n + 1))

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(valid_loader)
    f1_score = metric_f1.compute().item()
    dice_score = metric_dice.compute().item()
    scheduler.step(avg_val)
    current_lr = scheduler.get_last_lr()[0]

    print(f"  → Train Loss: {avg_train:.4f} | "
          f"Val Loss: {avg_val:.4f} | "
          f"F1: {f1_score:.4f} | "
          f"Dice: {dice_score:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    scheduler.step(avg_val)
    early_stopper(avg_val)
    if early_stopper.early_stop:
        print("Early stopping triggered")
        break
