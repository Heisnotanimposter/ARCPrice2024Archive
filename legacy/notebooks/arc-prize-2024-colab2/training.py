import torch
import torch.nn as nn
from tqdm import tqdm
import os

def run_kfold_cross_validation(data, model_fn, K=5):
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(data)):
        print(f"Fold {fold+1}/{K}")
        train_loader = DataLoader(
            Subset(data, train_indices),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            Subset(data, val_indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=10, patience=5, device='cpu', target_directory='.'):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1, model.fc.out_features)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).view(-1, model.fc.out_features)
                val_loss += criterion(outputs, targets.view(-1)).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(target_directory, 'best_model.pth'))
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save checkpoint
        checkpoint_path = os.path.join(target_directory, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")