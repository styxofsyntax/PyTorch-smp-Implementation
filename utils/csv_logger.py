import csv
import os
import pandas as pd
import matplotlib.pyplot as plt


class CSVLogger:
    def __init__(self, base_dir, fields):
        os.makedirs(base_dir, exist_ok=True)

        self.base_dir = base_dir
        self.filename = os.path.join(base_dir, "metrics.csv")
        self.fields = fields

        with open(self.filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()

    def log(self, values):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(values)

    def generate_plots(self):
        # Load data
        df = pd.read_csv(self.filename)

        # Plot 1: Loss
        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.base_dir, 'loss_curve.png'))
        plt.close()

        # Plot 2: F1 scores
        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'], df['f1_bg'], label='F1 Background')
        plt.plot(df['epoch'], df['f1_card'], label='F1 Card')
        plt.plot(df['epoch'], df['f1_damage'], label='F1 Damage')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Per-Class F1 Scores')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.base_dir, 'f1_scores.png'))
        plt.close()

        # Plot 3: Dice Score
        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'], df['dice_score'],
                 label='Dice Score', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Dice Score Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.base_dir, 'dice_score.png'))
        plt.close()

        # Plot 4: Learning Rate
        plt.figure(figsize=(8, 6))
        plt.plot(df['epoch'], df['lr'], label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.base_dir, 'learning_rate.png'))
        plt.close()
