import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

os.makedirs("assets", exist_ok=True)

# Loss Curve 
history = torch.load("checkpoints/training_history.pth", weights_only=False)
train_losses = history["train_losses"]
valid_losses = history["valid_losses"]
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', label="Train Loss")
plt.plot(epochs, valid_losses, marker='o', label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("assets/loss_curve.png", dpi=150)
plt.close()
print("Saved assets/loss_curve.png")

# Confusion Matrix + Classification Report
results = torch.load("checkpoints/test_results.pth", weights_only=False)
all_predictions = results["predictions"]
all_labels = results["labels"]
class_names = ["Empty", "Occupied"]

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=150)
plt.close()
print("Saved assets/confusion_matrix.png")

# Per-class Metrics Bar Chart 
report = classification_report(all_labels, all_predictions,
                                target_names=class_names, output_dict=True)

metrics = ["precision", "recall", "f1-score"]
x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(7, 5))
for i, metric in enumerate(metrics):
    values = [report[cls][metric] for cls in class_names]
    plt.bar(x + i * width, values, width, label=metric)

plt.xticks(x + width, class_names)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Per-Class Metrics")
plt.legend()
plt.tight_layout()
plt.savefig("assets/per_class_metrics.png", dpi=150)
plt.close()
print("Saved assets/per_class_metrics.png")

# Prediction Sample Grid
import random
from src.data.dataset import get_train_loader, get_valid_loader

all_images = results["images"]
all_predictions_arr = np.array(all_predictions)
all_labels_arr = np.array(all_labels)

total_samples = len(all_predictions_arr)
random.seed(1)
random_indices = random.sample(range(total_samples), min(25, total_samples))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

plt.figure(figsize=(15, 15))
for i, idx in enumerate(random_indices):
    plt.subplot(5, 5, i + 1)
    img = all_images[idx].numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)

    pred_idx = all_predictions_arr[idx]
    true_idx = all_labels_arr[idx]
    title_color = "green" if pred_idx == true_idx else "red"

    plt.imshow(img)
    plt.title(f"Pred: {class_names[pred_idx]}\nTrue: {class_names[true_idx]}",
              color=title_color, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("assets/prediction_samples.png", dpi=150)
plt.close()
print("Saved assets/prediction_samples.png")

# Misclassified Examples Grid 
wrong_indices = np.where(all_predictions_arr != all_labels_arr)[0]
random.seed(1)
wrong_sample = random.sample(list(wrong_indices), min(25, len(wrong_indices)))

plt.figure(figsize=(15, 15))
for i, idx in enumerate(wrong_sample):
    plt.subplot(5, 5, i + 1)
    img = all_images[idx].numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)

    pred_idx = all_predictions_arr[idx]
    true_idx = all_labels_arr[idx]

    plt.imshow(img)
    plt.title(f"Pred: {class_names[pred_idx]}\nTrue: {class_names[true_idx]}",
              color="red", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("assets/misclassified_samples.png", dpi=150)
plt.close()
print("Saved assets/misclassified_samples.png")

# Class Distribution Chart
def count_classes(loader_dataset):
    targets = loader_dataset.targets if hasattr(loader_dataset, "targets") else [loader_dataset.dataset.targets[i] for i in loader_dataset.indices]
    empty_count = sum(1 for t in targets if t == 0)
    occupied_count = sum(1 for t in targets if t == 1)
    return empty_count, occupied_count

train_loader = get_train_loader()
valid_loader = get_valid_loader()

train_counts = count_classes(train_loader.dataset)
valid_counts = count_classes(valid_loader.dataset)
test_counts = (
    sum(1 for l in all_labels if l == 0),
    sum(1 for l in all_labels if l == 1),
)

splits = ["Train", "Valid", "Test"]
empty_vals = [train_counts[0], valid_counts[0], test_counts[0]]
occupied_vals = [train_counts[1], valid_counts[1], test_counts[1]]

x = np.arange(len(splits))
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, empty_vals, width, label="Empty")
plt.bar(x + width/2, occupied_vals, width, label="Occupied")
plt.xticks(x, splits)
plt.ylabel("Number of Images")
plt.title("Class Distribution Across Splits")
plt.legend()
plt.tight_layout()
plt.savefig("assets/class_distribution.png", dpi=150)
plt.close()
print("Saved assets/class_distribution.png")
