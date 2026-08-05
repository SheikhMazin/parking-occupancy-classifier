import torch
import os
import torchvision.models as models
from src.data.dataset import get_train_loader, get_valid_loader 
from src.models.classifier import get_model

os.makedirs("checkpoints", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model = model.to(device)


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        )

num_epochs = 10
train_loader = get_train_loader()
valid_loader = get_valid_loader()

train_losses = []
valid_losses = []

best_valid_loss = float("inf")
for epoch in range(num_epochs):
    total_loss = 0

    # Set model to train
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}, average loss: {avg_loss:.4f}")
   
    
    total_valid_loss = 0

    # Set model to eval
    model.eval()
    for images , labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_valid_loss += loss.item()
    
    avg_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)
    print(f"Epoch {epoch}, average VALID loss: {avg_valid_loss:.4f}")

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"Saved new best model (valid loss: {avg_valid_loss:.4f})")

torch.save({
    "train_losses": train_losses,
    "valid_losses": valid_losses,
}, "checkpoints/training_history.pth")
