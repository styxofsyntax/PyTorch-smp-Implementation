import os
import torch


class EarlyStopping:
    def __init__(self, patience, delta, save_dir):
        self.patience = patience
        self.delta = delta
        self.save_dir = save_dir
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), os.path.join(
                self.save_dir, "best_model.pth"))
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
