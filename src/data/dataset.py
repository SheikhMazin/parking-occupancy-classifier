import torch
import random
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

transform_pipeline = v2.Compose([
    v2.Resize(size=(128, 128), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


def get_train_loader(samples_per_class=15000, batch_size=32):
    train_dataset = ImageFolder(
        root="data/processed/train",
        transform=transform_pipeline,
    )

    empty_indices = [i for i, label in enumerate(train_dataset.targets) if label == 0]
    occupied_indices = [i for i, label in enumerate(train_dataset.targets) if label == 1]

    random.seed(42)
    sampled_empty = random.sample(empty_indices, samples_per_class)
    sampled_occupied = random.sample(occupied_indices, samples_per_class)

    balanced_indices = sampled_empty + sampled_occupied
    balanced_dataset = Subset(train_dataset, balanced_indices)

    return DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)


def get_valid_loader(batch_size=32):
    valid_dataset = ImageFolder(
        root="data/processed/valid",
        transform=transform_pipeline,
    )
    return DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


def get_test_loader(batch_size=32):
    test_dataset = ImageFolder(
        root="data/processed/test",
        transform=transform_pipeline,
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    train_loader = get_train_loader()
    valid_loader = get_valid_loader()
    test_loader = get_test_loader()

    images, labels = next(iter(train_loader))
    print(f"Train batch — images: {images.shape}, labels: {labels.shape}")
    print(f"Valid dataset size: {len(valid_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
