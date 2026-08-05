import torch
import random
import numpy as np
import matplotlib.pyplot as plt

data = torch.load("checkpoints/test_results.pth", weights_only=False)
all_images = data["images"]
all_predictions = data["predictions"]
all_labels = data["labels"]

class_names = ["Empty", "Occupied"]
total_samples = len(all_predictions)
random_indices = random.sample(range(total_samples), min(25, total_samples))

plt.figure(figsize=(15, 15))
for i, idx in enumerate(random_indices):
    plt.subplot(5, 5, i + 1)
    img_tensor = all_images[idx]
    pred_idx = all_predictions[idx]
    true_idx = all_labels[idx]

    img = img_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    title_color = "green" if pred_idx == true_idx else "red"
    plt.imshow(img)
    plt.title(f"Pred: {class_names[pred_idx]}\nTrue: {class_names[true_idx]}",
              color=title_color, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.show()
