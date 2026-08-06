# Install necessary libraries
!pip install torch torchvision

# Import libraries
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
MAX_GRID_SIZE = 30
NUM_COLORS = 11  # Colors from 0 to 9 plus padding
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

# Dataset class
class ARCDataset(Dataset):
    def __init__(self, challenges, ids):
        self.samples = []
        for id in ids:
            task = challenges[id]
            for sample in task['train']:
                input_grid = self.pad_grid(sample['input'])
                output_grid = self.pad_grid(sample['output'])
                self.samples.append((input_grid, output_grid))

    def pad_grid(self, grid):
        grid = np.array(grid)
        H, W = grid.shape
        padded_grid = np.full((MAX_GRID_SIZE, MAX_GRID_SIZE), 10, dtype=np.int64)
        padded_grid[:H, :W] = grid
        return padded_grid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_grid, output_grid = self.samples[idx]
        return torch.tensor(input_grid), torch.tensor(output_grid)

# Load datasets
with open('/path/to/arc_training_data.json', 'r') as f:
    train_challenges = json.load(f)
train_ids = list(train_challenges.keys())

# Split data
train_ids_train, train_ids_val = train_test_split(train_ids, test_size=0.2, random_state=42)
train_dataset = ARCDataset(train_challenges, train_ids_train)
val_dataset = ARCDataset(train_challenges, train_ids_val)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model components
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_size=512):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, embed_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(embed_size),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        return x

class AutoencoderEncoder(nn.Module):
    def __init__(self, embed_size=512, latent_dim=256):
        super(AutoencoderEncoder, self).__init__()
        self.fc = nn.Linear(embed_size, latent_dim)

    def forward(self, x):
        latent = self.fc(x)
        return latent

class AutoencoderDecoder(nn.Module):
    def __init__(self, latent_dim=256, embed_size=512, output_size=NUM_COLORS):
        super(AutoencoderDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, embed_size)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_size, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, output_size, kernel_size=2, stride=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv_layers(x)
        return x

# Full model
class ARCModel(nn.Module):
    def __init__(self):
        super(ARCModel, self).__init__()
        self.encoder = CNNFeatureExtractor()
        self.autoencoder_encoder = AutoencoderEncoder()
        self.autoencoder_decoder = AutoencoderDecoder()

    def forward(self, x):
        x = F.one_hot(x, num_classes=NUM_COLORS).permute(0, 3, 1, 2).float()
        x = self.encoder(x)
        x = self.autoencoder_encoder(x)
        x = x.mean(dim=1)
        x = self.autoencoder_decoder(x)
        return x

# Initialize model, loss, optimizer
model = ARCModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for input_grid, target_grid in train_loader:
        input_grid = input_grid.to(DEVICE)
        target_grid = target_grid.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input_grid)
        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, NUM_COLORS)
        target = target_grid.view(-1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_grid, target_grid in val_loader:
            input_grid = input_grid.to(DEVICE)
            target_grid = target_grid.to(DEVICE)
            outputs = model(input_grid)
            outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, NUM_COLORS)
            target = target_grid.view(-1)
            loss = criterion(outputs, target)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'arc_model.pth')

# Evaluation
model.eval()
test_ids = train_ids_val  # Use validation set for testing
test_dataset = ARCDataset(train_challenges, test_ids)
test_loader = DataLoader(test_dataset, batch_size=1)
correct = 0
total = 0
with torch.no_grad():
    for input_grid, target_grid in test_loader:
        input_grid = input_grid.to(DEVICE)
        target_grid = target_grid.to(DEVICE)
        outputs = model(input_grid)
        predicted = outputs.argmax(dim=1)
        total += target_grid.numel()
        correct += (predicted == target_grid).sum().item()
accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")