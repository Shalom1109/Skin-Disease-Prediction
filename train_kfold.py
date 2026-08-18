import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
import os
from sklearn.model_selection import KFold
import numpy as np
from backend.model import SkinSenseEnsemble

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
NUM_FOLDS = 5
NUM_CLASSES = 5
DATA_DIR = r"d:\Skin Disease ConvNeXt\dataset"

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_cv():
    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"--- FOLD {fold+1}/{NUM_FOLDS} ---")
        
        train_sub = Subset(dataset, train_ids)
        val_sub = Subset(dataset, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        model = SkinSenseEnsemble(num_classes=NUM_CLASSES)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        for epoch in range(EPOCHS):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            if acc > best_acc:
                best_acc = acc
            print(f"Epoch {epoch+1}, Val Accuracy: {acc:.2f}%")
        
        results.append(best_acc)
        torch.save(model.state_dict(), f"backend/model_fold_{fold+1}.pth")

    print(f"\nK-Fold Results: {results}")
    print(f"Mean Accuracy: {np.mean(results):.2f}%")

if __name__ == "__main__":
    train_cv()
