import torch.nn as nn
import torchvision.models as models


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    num_classes = 2
    model.fc = nn.Linear(num_features, num_classes)

    return model

if __name__ == "__main__":
    model = get_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
