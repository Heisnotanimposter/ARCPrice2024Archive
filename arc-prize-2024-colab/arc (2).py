# Import necessary libraries
import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.model_selection import train_test_split

# Mount Google Drive (if necessary)
from google.colab import drive
drive.mount('/content/drive')

# Define color map for visualization
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

# Verify CUDA availability
print("CUDA available:", torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the maximum grid size based on the dataset analysis
MAX_GRID_SIZE = 30  # Adjust as needed

# Define utility functions
def pad_grid(grid, size):
    h, w = grid.shape
    padded_grid = np.zeros((size, size), dtype=grid.dtype)
    padded_grid[:h, :w] = grid
    return padded_grid

def grid_to_sequence(grid):
    # Flatten the grid to create a sequence
    return grid.flatten()

def sequence_to_grid(sequence, size):
    return sequence.reshape(size, size)

# Define the data loading function with error handling
def load_arc_data(challenges_path):
    try:
        with open(challenges_path, 'r') as f:
            challenges_data = json.load(f)
    except Exception as e:
        print(f"Error loading data from {challenges_path}: {e}")
        return [], []

    inputs, outputs = [], []

    for task_id, task_data in tqdm(challenges_data.items(), desc="Loading tasks"):
        # Process training examples only
        for example in task_data.get("train", []):
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])

            if input_grid.ndim != 2 or output_grid.ndim != 2:
                print(f"Invalid grid dimensions in task {task_id}")
                continue

            # Standardize grid sizes
            input_grid_padded = pad_grid(input_grid, size=MAX_GRID_SIZE)
            output_grid_padded = pad_grid(output_grid, size=MAX_GRID_SIZE)

            # Convert grids to sequences
            input_sequence = grid_to_sequence(input_grid_padded)
            output_sequence = grid_to_sequence(output_grid_padded)

            inputs.append(input_sequence)
            outputs.append(output_sequence)

    return inputs, outputs

# Paths to your data files
train1_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_training_challenges.json'

# Load and preprocess data
inputs, outputs = load_arc_data(train1_path)

# Verify data loading
print(f"Number of samples loaded: {len(inputs)}")
if len(inputs) == 0:
    print("No data was loaded. Please check your data paths and preprocessing steps.")
else:
    print(f"Sample input sequence length: {len(inputs[0])}")
    print(f"Sample output sequence length: {len(outputs[0])}")

# Define the ARCDataset class with data augmentation
class ARCDataset(Dataset):
    def __init__(self, inputs, outputs, augment=False):
        self.inputs = inputs
        self.outputs = outputs
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        output_seq = self.outputs[idx]

        if self.augment:
            # Convert sequences back to grids
            input_grid = input_seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
            output_grid = output_seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)

            # Apply random transformations
            if random.random() > 0.5:
                input_grid = np.flip(input_grid, axis=0)  # Vertical flip
                output_grid = np.flip(output_grid, axis=0)
            if random.random() > 0.5:
                input_grid = np.flip(input_grid, axis=1)  # Horizontal flip
                output_grid = np.flip(output_grid, axis=1)
            if random.random() > 0.5:
                k = random.choice([1, 2, 3])  # Rotate 90, 180, or 270 degrees
                input_grid = np.rot90(input_grid, k=k)
                output_grid = np.rot90(output_grid, k=k)

            # Convert grids back to sequences
            input_seq = input_grid.flatten()
            output_seq = output_grid.flatten()

        # Convert sequences to tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)    # [seq_length]
        output_tensor = torch.tensor(output_seq, dtype=torch.long)  # [seq_length]
        return input_tensor, output_tensor

# Define the collate function
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)   # [batch_size, seq_length]
    targets = torch.stack(targets) # [batch_size, seq_length]
    return inputs, targets

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split

train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
    inputs, outputs, test_size=0.1, random_state=42)

# Create training and validation datasets
train_dataset = ARCDataset(train_inputs, train_outputs, augment=True)
val_dataset = ARCDataset(val_inputs, val_outputs, augment=False)

# Create DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Hyperparameters
input_size = 10
hidden_size = 256
num_layers = 3
num_classes = 10
learning_rate = 0.0005
dropout_rate = 0.2
num_epochs = 50
patience = 5

# Define the model class with dropout
class xLSTMModelClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.0):
        super(xLSTMModelClassification, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        # Define a fully connected layer to map hidden states to class scores
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_length]
        batch_size = x.size(0)
        seq_length = x.size(1)

        # Embed input to one-hot vectors
        x = nn.functional.one_hot(x, num_classes=10).float()  # [batch_size, seq_length, num_classes]

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: [batch_size, seq_length, hidden_size]

        # Flatten out for fully connected layer
        out = out.reshape(-1, self.hidden_size)  # [batch_size * seq_length, hidden_size]
        out = self.fc(out)  # [batch_size * seq_length, num_classes]

        # Reshape back to [batch_size, seq_length, num_classes]
        out = out.view(batch_size, seq_length, -1)
        return out

# Instantiate the model
model = xLSTMModelClassification(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout_rate=dropout_rate
).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Define the training function with early stopping and checkpointing
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=10, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs_flat = outputs.view(-1, num_classes)
            targets_flat = targets.view(-1)

            loss = criterion(outputs_flat, targets_flat)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs_flat, 1)
            train_correct += (predicted == targets_flat).sum().item()
            train_total += targets_flat.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                outputs_flat = outputs.view(-1, num_classes)
                targets_flat = targets.view(-1)

                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()

                _, predicted = torch.max(outputs_flat, 1)
                val_correct += (predicted == targets_flat).sum().item()
                val_total += targets_flat.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, f'checkpoint_epoch_{epoch+1}.pth')

# Train the model
train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=num_epochs, patience=patience)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Define the inference function
def infer(model, input_sequence):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_length]
        output_logits = model(input_tensor)  # [1, seq_length, num_classes]
        predicted_classes = torch.argmax(output_logits, dim=-1).squeeze(0).cpu().numpy()  # [seq_length]
    return predicted_classes

# Define the visualization function
def visualize_grids(input_grid, predicted_grid, target_grid=None):
    fig, axs = plt.subplots(1, 3 if target_grid is not None else 2, figsize=(15, 5))

    axs[0].imshow(input_grid, cmap=cmap, norm=norm)
    axs[0].set_title('Input Grid')

    axs[1].imshow(predicted_grid, cmap=cmap, norm=norm)
    axs[1].set_title('Predicted Output Grid')

    if target_grid is not None:
        axs[2].imshow(target_grid, cmap=cmap, norm=norm)
        axs[2].set_title('Target Output Grid')

    plt.show()

# Select a sample from the validation dataset
sample_idx = 0  # Change index to test different samples
input_seq, target_seq = val_dataset[sample_idx]
input_seq = input_seq.numpy()    # Shape: [seq_length]
target_seq = target_seq.numpy()

# Run inference
predicted_seq = infer(model, input_seq)

# Reshape sequences back to grids
input_grid = input_seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
predicted_grid = predicted_seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
target_grid = target_seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)

# Visualize the grids
visualize_grids(input_grid, predicted_grid, target_grid)