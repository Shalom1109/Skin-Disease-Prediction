import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
from backend.model import SkinSenseEnsemble

# Hyperparameters
BATCH_SIZE = 2 # Reduced for 384x384 VRAM safety
ACCUMULATION_STEPS = 8 # Effective Batch Size = 16 (Still 16)
LEARNING_RATE = 2e-5 # Finer adjustment for 384 resolution
EPOCHS = 20
NUM_CLASSES = 7
DATA_DIR = r"d:\Skin Disease ConvNeXt\dataset"
CHECKPOINT_PATH = "backend/checkpoint_latest.pth"
BEST_MODEL_PATH = "backend/skin_sense_ensemble_best.pth"

# Extreme Augmentation for >99% Accuracy (384 resolution)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(384),
    transforms.RandAugment(num_ops=2, magnitude=9), # Advanced augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train():
    # Load Dataset
    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Class Counts (akiec, bcc, bkl, df, mel, nv, vasc)
    class_counts = [327, 514, 1099, 115, 1113, 6705, 142]
    class_weights = [1.0 / c for c in class_counts]
    
    # Create sample weights for WeightedRandomSampler
    train_indices = train_dataset.indices
    sample_weights = [class_weights[dataset.samples[i][1]] for i in train_indices]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Ensemble Model
    model = SkinSenseEnsemble(num_classes=NUM_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss, Optimizer, Scaler, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    scaler = torch.amp.GradScaler('cuda') # Modern API for Mixed Precision

    # Resume Logic
    start_epoch = 0
    best_accuracy = 0.0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resuming from Epoch {start_epoch+1}")
    elif os.path.exists(BEST_MODEL_PATH):
        print(f"Found best model at {BEST_MODEL_PATH}. Loading weights...")
        try:
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            print("Weights loaded successfully.")
            # We don't know the exact epoch from weights alone, but we start from 0 by default.
            # However, if the user was on epoch 2, and best was epoch 1, they might want to start from 1.
            # For now, we stay at epoch 0 unless a full checkpoint is found.
        except Exception as e:
            print(f"Warning: Could not load weights from best model: {e}")

    # Initialize log file
    log_mode = "a" if start_epoch > 0 else "w"
    with open("train_log.txt", log_mode) as f:
        f.write(f"Training {'resumed' if start_epoch > 0 else 'started'} at Epoch {start_epoch+1}\n")
    
    print(f"Starting training on {device}...")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Auto Mixed Precision
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * ACCUMULATION_STEPS
            
            if (batch_idx + 1) % 5 == 0:
                batch_msg = f"Epoch [{epoch+1}/{EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item() * ACCUMULATION_STEPS:.4f}"
                print(batch_msg)
                with open("train_log.txt", "a") as f:
                    f.write(batch_msg + "\n")

        
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
        
        accuracy = 100 * correct / total
        log_msg = f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%"
        print(log_msg)
        
        with open("train_log.txt", "a") as f:
            f.write(log_msg + "\n")
        
        scheduler.step()
        
        # Save Best Model
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved with accuracy: {accuracy:.2f}%")
        
        # Save Latest Checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_accuracy': best_accuracy
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        
        if epoch == EPOCHS - 1:
            torch.save(model.state_dict(), "backend/skin_sense_ensemble_final.pth")

if __name__ == "__main__":
    train()
