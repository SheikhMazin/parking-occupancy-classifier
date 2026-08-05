import torch
import os
import random
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.data.dataset import get_test_loader
from src.models.classifier import get_model
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()

state_dict = torch.load(
        "checkpoints/best_model.pth",
        map_location=device,
        weights_only=True,
        )

model.load_state_dict(state_dict)
model.to(device)

model.eval()
criterion = torch.nn.CrossEntropyLoss()

all_predictions = []    # Every prediction the model makes
all_labels = []         # Every true label across the entire set
all_images = []

test_loader = get_test_loader()

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
    
    preds = torch.argmax(outputs, dim=1)

    all_images.append(images.cpu())
    all_predictions.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

all_images = torch.cat(all_images, dim=0)

acc = accuracy_score(all_labels, all_predictions)
print(f"Accuracy: {acc * 100:.2f}%")

print("\nClassification Report: ")
print(classification_report(all_labels, all_predictions, target_names=["Empty", "Occupied"]))

print("\nConfusion Matrix: ")
print(confusion_matrix(all_labels, all_predictions))

torch.save({
    "images": all_images,
    "predictions": all_predictions,
    "labels": all_labels,
}, "checkpoints/test_results.pth")
