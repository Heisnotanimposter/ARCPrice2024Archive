from google.colab import drive
drive.mount('/content/drive')

!pip install colorama
!pip install utils
!pip install xLSTM

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random

MAX_GRID_SIZE = 30
input_size = 10
hidden_size = 256
num_layers = 3
num_classes = 10
learning_rate = 0.0005
dropout_rate = 0.2
num_epochs = 50
patience = 5
batch_size = 128
target_directory = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/results/'

class xLSTMModelClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.0):
        super(xLSTMModelClassification, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = nn.functional.one_hot(x, num_classes=self.fc.out_features).float()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out.reshape(-1, self.hidden_size)).view(x.size(0), -1, self.fc.out_features)
        return out

def pad_grid(grid, size):
    h, w = grid.shape
    padded_grid = np.zeros((size, size), dtype=grid.dtype)
    padded_grid[:h, :w] = grid
    return padded_grid

def grid_to_sequence(grid):
    return grid.flatten()

def sequence_to_grid(sequence, size):
    return sequence.reshape(size, size)

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
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(output_seq, dtype=torch.long)

def collate_fn(batch):
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)

def load_arc_data(challenges_path):
    with open(challenges_path, 'r') as f:
        challenges_data = json.load(f)
    inputs, outputs = [], []
    for task_id, task_data in challenges_data.items():
        for example in task_data.get("train", []):
            input_grid = np.array(example["input"])
            output_grid = np.array(example["output"])
            input_sequence = grid_to_sequence(pad_grid(input_grid, MAX_GRID_SIZE))
            output_sequence = grid_to_sequence(pad_grid(output_grid, MAX_GRID_SIZE))
            inputs.append(input_sequence)
            outputs.append(output_sequence)
    return inputs, outputs

train_data_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_training_challenges.json'
inputs, outputs = load_arc_data(train_data_path)

train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.1, random_state=42)
train_dataset = ARCDataset(train_inputs, train_outputs, augment=True)
val_dataset = ARCDataset(val_inputs, val_outputs, augment=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=10, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).view(-1, num_classes)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).view(-1, num_classes)
                val_loss += criterion(outputs, targets.view(-1)).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

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

def infer_and_save(model, input_tasks, output_file="submission.json"):
    model.eval()
    results = {}
    for task_id, input_grid in tqdm(input_tasks.items(), desc="Running Inference"):
        input_tensor = torch.tensor(grid_to_sequence(pad_grid(input_grid, MAX_GRID_SIZE)), dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output_sequence = model(input_tensor).argmax(-1).squeeze(0).cpu().numpy()
        output_grid = sequence_to_grid(output_sequence, MAX_GRID_SIZE)
        results[task_id] = output_grid.tolist()

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Submission saved to {output_file}")
def load_test_data(test_data_path):
    """
    Loads the test data from the JSON file and returns a dictionary of input grids.

    Args:
        test_data_path (str): Path to the test data JSON file.

    Returns:
        input_tasks (dict): A dictionary where the keys are task IDs and the values are input grids.
    """
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    input_tasks = {}

    for task_id, task in test_data.items():
        # For each test example in the task
        for idx, example in enumerate(task["test"]):
            input_grid = np.array(example["input"])
            input_tasks[f"{task_id}_{idx}"] = input_grid

    return input_tasks
# Now use this function to load the test data and run inference
test_data_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_test_challenges.json'
input_tasks = load_test_data(test_data_path)
infer_and_save(model, input_tasks)

# Load test data and run inference
test_data_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_test_challenges.json'
input_tasks = load_test_data(test_data_path)
infer_and_save(model, input_tasks)

def run_kfold_cross_validation(data, model_fn, K=5):
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(data)):
        print(f"Fold {fold+1}/{K}")
        train_loader = DataLoader(Subset(data, train_indices), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(Subset(data, val_indices), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        model = model_fn().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        train_model(model, train_loader, val_loader, optimizer, criterion, scheduler)
        fold_results.append(model)

    return fold_results

writer = SummaryWriter(log_dir=os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S')))
for epoch in range(num_epochs):
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

def print_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
        reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_memory = reserved_memory - allocated_memory
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Reserved GPU Memory: {reserved_memory:.2f} GB")
        print(f"Allocated GPU Memory: {allocated_memory:.2f} GB")
        print(f"Free Reserved GPU Memory: {free_memory:.2f} GB\n")
    else:
        print("CUDA is not available.")

# Before training loop
print_gpu_memory()

for epoch in range(num_epochs):
    # Training code...

    # After each epoch
    print(f"Epoch {epoch+1} completed.")
    print_gpu_memory()

import psutil
import GPUtil
import csv
import time

def log_resource_usage(log_file_path, epoch, step):
    """
    Logs the current resource usage to a CSV file.

    Args:
        log_file_path (str): Path to the CSV log file.
        epoch (int): Current epoch number.
        step (int or str): Current step number or description.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_data = {}
    if gpus:
        gpu = gpus[0]  # Modify if using multiple GPUs
        gpu_data = {
            'gpu_load_percent': gpu.load * 100,
            'gpu_memory_util_percent': gpu.memoryUtil * 100,
            'gpu_memory_total_MB': gpu.memoryTotal,
            'gpu_memory_used_MB': gpu.memoryUsed
        }
    else:
        gpu_data = {
            'gpu_load_percent': 0,
            'gpu_memory_util_percent': 0,
            'gpu_memory_total_MB': 0,
            'gpu_memory_used_MB': 0
        }

    # Prepare log entry
    log_entry = {
        'timestamp': timestamp,
        'epoch': epoch,
        'step': step,
        'cpu_usage_percent': cpu_usage,
        'ram_usage_percent': ram_usage,
        **gpu_data
    }

    # Write to CSV file
    file_exists = os.path.isfile(log_file_path)
    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'epoch', 'step', 'cpu_usage_percent', 'ram_usage_percent',
            'gpu_load_percent', 'gpu_memory_util_percent', 'gpu_memory_total_MB', 'gpu_memory_used_MB'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

accumulation_steps = 4  # Adjust as needed
optimizer.zero_grad()

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        outputs_flat = outputs.view(-1, num_classes)
        targets_flat = targets.view(-1)
        loss = criterion(outputs_flat, targets_flat) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,  # Adjust based on your CPU cores
    pin_memory=True,
    collate_fn=collate_fn
)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            outputs_flat = outputs.view(-1, num_classes)
            targets_flat = targets.view(-1)
            loss = criterion(outputs_flat, targets_flat)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# After validation phase
del inputs, targets, outputs, loss
torch.cuda.empty_cache()

checkpoint_path = os.path.join(target_directory, f'checkpoint_epoch_{epoch+1}.pth')
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
best_val_loss = checkpoint['best_val_loss']

print(f"Resuming training from epoch {start_epoch}")

from sklearn.model_selection import KFold

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)

full_dataset = ARCDataset(inputs, outputs, augment=True)
num_samples = len(full_dataset)

for fold, (train_indices, val_indices) in enumerate(kf.split(range(num_samples))):
    print(f'Fold {fold+1}/{K}')
    
    # Create data loaders for this fold
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
model = xLSTMModelClassification(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout_rate=dropout_rate
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=num_epochs, patience=patience)

# Save the trained model
best_model_path = os.path.join(target_directory, 'best_model.pth')
torch.save(model.state_dict(), best_model_path)
print(f"Model saved to {best_model_path}")

# Load test data and run inference
test_data_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_test_challenges.json'
input_tasks = load_test_data(test_data_path)

# Load the trained model for inference
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
infer_and_save(model, input_tasks)    
    # Train the model for this fold
    train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=num_epochs, patience=patience)
    
    # Save the model for this fold
    fold_model_path = os.path.join(target_directory, f'best_model_fold_{fold+1}.pth')
    torch.save(model.state_dict(), fold_model_path)
    print(f'Model for fold {fold+1} saved to {fold_model_path}')

    import numpy as np

avg_train_loss = np.mean([np.mean(losses) for losses in fold_train_losses])
avg_val_loss = np.mean([np.mean(losses) for losses in fold_val_losses])
avg_train_accuracy = np.mean([np.mean(accs) for accs in fold_train_accuracies])
avg_val_accuracy = np.mean([np.mean(accs) for accs in fold_val_accuracies])

print(f'Average Training Loss: {avg_train_loss:.4f}')
print(f'Average Validation Loss: {avg_val_loss:.4f}')
print(f'Average Training Accuracy: {avg_train_accuracy:.4f}')
print(f'Average Validation Accuracy: {avg_val_accuracy:.4f}')

import csv

experiment_log_path = os.path.join(target_directory, 'experiment_log.csv')
write_headers = not os.path.exists(experiment_log_path)

with open(experiment_log_path, 'a', newline='') as csvfile:
    fieldnames = ['Experiment_ID', 'Date', 'Hyperparameters', 'Avg_Train_Loss', 'Avg_Val_Loss', 'Avg_Train_Acc', 'Avg_Val_Acc', 'Comments']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if write_headers:
        writer.writeheader()
    
    writer.writerow({
        'Experiment_ID': f'Exp_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Hyperparameters': f'LR={learning_rate}, HS={hidden_size}, NL={num_layers}',
        'Avg_Train_Loss': avg_train_loss,
        'Avg_Val_Loss': avg_val_loss,
        'Avg_Train_Acc': avg_train_accuracy,
        'Avg_Val_Acc': avg_val_accuracy,
        'Comments': 'Cross-validation results'
    })

import necessary libraries
define hyperparameters and paths

    # Validation code...

    # Validation code...

"""from google.colab import drive
drive.mount('/content/drive')

!pip install colorama
!pip install utils
!pip install xLSTM

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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define utility functions
def pad_grid(grid, size):
    h, w = grid.shape
    padded_grid = np.zeros((size, size), dtype=grid.dtype)
    padded_grid[:h, :w] = grid
    return padded_grid

def grid_to_sequence(grid):
    return grid.flatten()

def sequence_to_grid(sequence, size):
    return sequence.reshape(size, size)

# Define the model class
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
        x = nn.functional.one_hot(x, num_classes=self.fc.out_features).float()  # [batch_size, seq_length, num_classes]

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

# Hyperparameters
input_size = 10
hidden_size = 256
num_layers = 3
num_classes = 10
dropout_rate = 0.2

# Instantiate the model
model = xLSTMModelClassification(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_classes=num_classes,
    dropout_rate=dropout_rate
)

# Load the best model weights
state_dict = torch.load('best_model.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Load test data
def load_test_data(test_challenges_path):
    with open(test_challenges_path, 'r') as f:
        test_data = json.load(f)

    input_tasks = {}

    for task_id, task in test_data.items():
        # For each test example in the task
        for idx, example in enumerate(task["test"]):
            input_grid = np.array(example["input"])
            input_tasks[f"{task_id}_{idx}"] = input_grid

    return input_tasks

# Path to your test data file
test_data_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_test_challenges.json'

# Load test data
input_tasks = load_test_data(test_data_path)

# Define the inference function
def infer_and_save(model, input_tasks, output_file="submission.json"):
    model.eval()
    device = next(model.parameters()).device
    results = {}

    for task_id, input_grid in tqdm(input_tasks.items(), desc="Running Inference"):
        # Preprocess input grid
        input_grid_padded = pad_grid(input_grid, size=MAX_GRID_SIZE)
        input_sequence = grid_to_sequence(input_grid_padded)
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_length]

        with torch.no_grad():
            output_logits = model(input_tensor)  # [1, seq_length, num_classes]
            predicted_classes = torch.argmax(output_logits, dim=-1).squeeze(0).cpu().numpy()  # [seq_length]

        # Convert the sequence back to grid
        output_grid = sequence_to_grid(predicted_classes, size=MAX_GRID_SIZE)

        # Remove padding to match original input size
        original_height, original_width = input_grid.shape
        output_grid_cropped = output_grid[:original_height, :original_width]

        # Convert grid to list for JSON serialization
        output_grid_list = output_grid_cropped.tolist()

        # Add result to dictionary
        results[task_id] = output_grid_list

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Submission saved to {output_file}")

# Run inference and save submission
infer_and_save(model, input_tasks, output_file="submission.json")


import torch
import json
import numpy as np
from collections import deque

# Ensure proper padding and transformation
def pad_grid(grid, size):
    h, w = grid.shape
    padded_grid = np.zeros((size, size), dtype=grid.dtype)
    padded_grid[:h, :w] = grid
    return padded_grid

def grid_to_sequence(grid):
    # Flatten the grid to create a sequence
    return grid.flatten()

def sequence_to_grid(sequence, size):
    # Reshape the sequence back into grid form
    return sequence.reshape(size, size)

# Transform grid using the BFS algorithm shared earlier
def transform_grid(input_grid):
    rows = len(input_grid)
    cols = len(input_grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    output_grid = [row.copy() for row in input_grid]

    # Directions for adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Initialize queue for BFS and mark edge 'A's as visited
    queue = deque()

    # Enqueue all 'A's on the edges
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and input_grid[i][j] == 'A':
                queue.append((i, j))
                visited[i][j] = True

    # Perform BFS to find all 'A's connected to the edges
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and input_grid[nx][ny] == 'A':
                visited[nx][ny] = True
                queue.append((nx, ny))

    # Replace enclosed 'A's with 'E's in the output grid
    for i in range(rows):
        for j in range(cols):
            if input_grid[i][j] == 'A' and not visited[i][j]:
                output_grid[i][j] = 'E'

    return output_grid

# Inference function and save the output to JSON
def infer_and_save(model, input_tasks, output_file="submission.json", max_grid_size=30):
    """
    Runs inference on a list of input grids, and saves results in submission.json format.
    
    Parameters:
    - model: trained PyTorch model
    - input_tasks: dictionary with task IDs and input grids to process
    - output_file: filename to save the submission as JSON
    """
    model.eval()
    device = next(model.parameters()).device
    results = {}

    for task_id, input_grid in input_tasks.items():
        # Preprocess input grid
        input_grid_padded = pad_grid(np.array(input_grid), size=max_grid_size)
        input_grid_normalized = input_grid_padded / 9.0
        input_sequence = grid_to_sequence(input_grid_normalized)
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # Perform model inference
            output_sequence = model(input_tensor)
            output_sequence = output_sequence.squeeze(0).cpu().numpy()

        # Convert the sequence back to grid and denormalize
        output_grid = sequence_to_grid(output_sequence, size=max_grid_size)
        output_grid_denormalized = (output_grid * 9.0).round().astype(int).tolist()

        # Add result to dictionary
        results[task_id] = output_grid_denormalized

    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Submission saved to {output_file}")

# Assuming you have a list of input grids called input_grids and a trained model
input_tasks = {
    "025d127b_0": test_input,  # Replace with your input grids for actual testing
    "045e512c_0": test_input  # Add more tasks as needed
}

infer_and_save(model, input_tasks, output_file="submission.json")

# Initialize lists to store loss and accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

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
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

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
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # ... [Rest of your training loop] ...

# After training, plot the loss curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.title("Loss over Epochs")
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.title("Accuracy over Epochs")
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Get a sample from the training dataset
sample_idx = 0  # Change index to test different samples
input_seq, target_seq = train_dataset[sample_idx]
input_seq = input_seq.numpy()
target_seq = target_seq.numpy()

# Run inference
predicted_seq = infer(model, input_seq)

# Reshape sequences back to grids
input_grid = input_seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
predicted_grid = predicted_seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)
target_grid = target_seq.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE)

# Visualize the grids
visualize_grids(input_grid, predicted_grid, target_grid)

num_epochs = 100  # Increase from 50 to 100

learning_rate = 0.001  # Try increasing from 0.0005
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

hidden_size = 512  # Increase from 256

num_layers = 4  # Increase from 3

self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)

def transform_grid(input_grid):
    # Implement task-specific logic here
    # For example, fill enclosed areas as in your earlier code
    # ...
    return output_grid

    def infer_and_save(model, input_tasks, output_file="submission.json"):
    # ... [existing code] ...

    for task_id, input_grid in tqdm(input_tasks.items(), desc="Running Inference"):
        # Check if task requires specific logic
        if task_id in task_specific_logic_tasks:
            output_grid = transform_grid(input_grid)
        else:
            # Proceed with model inference
            # ... [existing inference code] ...

        # Add result to dictionary
        results[task_id] = output_grid.tolist()

# Load submission data
with open("submission.json", "r") as f:
    submission_data = json.load(f)

# Pick a task to validate
task_id = '025d127b_0'  # Replace with actual task ID

# Retrieve input and output grids
input_grid = input_tasks[task_id]
output_grid = submission_data[task_id]

# Visualize input and output grids
print(f"Task ID: {task_id}")
print("Input Grid:")
for row in input_grid:
    print(row)
print("\nPredicted Output Grid:")
for row in output_grid:
    print(row)

if random.random() > 0.5:
    input_grid = np.rot90(input_grid)
    output_grid = np.rot90(output_grid)

color_map = {i: (i + random.randint(1, 9)) % 10 for i in range(10)}
input_grid = np.vectorize(color_map.get)(input_grid)
output_grid = np.vectorize(color_map.get)(output_grid)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

import jsonschema

# Define the schema (example)
schema = {
    "type": "object",
    "patternProperties": {
        "^[a-z0-9_]+$": {  # Task ID pattern
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "integer"}
            }
        }
    }
}

# Load the submission file
with open("submission.json", "r") as f:
    submission_data = json.load(f)

# Validate the JSON structure
try:
    jsonschema.validate(instance=submission_data, schema=schema)
    print("Submission file is valid.")
except jsonschema.exceptions.ValidationError as e:
    print("Submission file is invalid:", e)

# Check the values in the output grids
for task_id, grid in submission_data.items():
    for row in grid:
        for value in row:
            if not (0 <= value <= 9):
                print(f"Invalid value {value} in task {task_id}")

if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    patience_counter = 0
    torch.save(model.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")
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

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
best_val_loss = checkpoint['best_val_loss']

from sklearn.model_selection import KFold

# Number of folds
K = 5

# Create KFold object
kf = KFold(n_splits=K, shuffle=True, random_state=42)

# Your full dataset
full_dataset = ARCDataset(data, augment=True)

# Get the total number of samples
num_samples = len(full_dataset)

class ARCDataset(Dataset):
    def __init__(self, data, indices=None, augment=False):
        self.data = data
        self.augment = augment
        if indices is not None:
            self.data = [self.data[i] for i in indices]
        # ... [rest of your code] ...


from torch.utils.data import Subset
import copy

# Lists to store results
fold_train_losses = []
fold_val_losses = []
fold_train_accuracies = []
fold_val_accuracies = []

for fold, (train_indices, val_indices) in enumerate(kf.split(range(num_samples))):
    print(f'Fold {fold+1}/{K}')
    
    # Create data loaders for this fold
    train_subset = ARCDataset(data, indices=train_indices, augment=True)
    val_subset = ARCDataset(data, indices=val_indices, augment=False)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model, optimizer, scheduler, etc.
    model = xLSTMModelClassification(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training and validation code here
        # ... [use the same training loop as before, but within the fold loop] ...
        
        # Store fold results
        fold_train_losses.append(train_losses)
        fold_val_losses.append(val_losses)
        fold_train_accuracies.append(train_accuracies)
        fold_val_accuracies.append(val_accuracies)
    
    # Optionally, save the model for each fold
    fold_model_path = os.path.join(target_directory, f'best_model_fold_{fold+1}.pth')
    torch.save(model.state_dict(), fold_model_path)
    print(f'Model for fold {fold+1} saved to {fold_model_path}')

    import numpy as np

# Calculate average losses and accuracies across folds
avg_train_loss = np.mean([np.mean(losses) for losses in fold_train_losses])
avg_val_loss = np.mean([np.mean(losses) for losses in fold_val_losses])
avg_train_accuracy = np.mean([np.mean(accs) for accs in fold_train_accuracies])
avg_val_accuracy = np.mean([np.mean(accs) for accs in fold_val_accuracies])

print(f'Average Training Loss: {avg_train_loss:.4f}')
print(f'Average Validation Loss: {avg_val_loss:.4f}')
print(f'Average Training Accuracy: {avg_train_accuracy:.4f}')
print(f'Average Validation Accuracy: {avg_val_accuracy:.4f}')

# Install TensorBoard if not already installed
!pip install tensorboard

from torch.utils.tensorboard import SummaryWriter

# Define a unique log directory, e.g., using timestamp
from datetime import datetime

log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
writer = SummaryWriter(log_dir)

for epoch in range(num_epochs):
    # ... [existing training code] ...
    
    # Log training metrics
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
    
    # Log validation metrics
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
    
    # Optionally, log learning rate
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Learning Rate', current_lr, epoch)

%load_ext tensorboard
%tensorboard --logdir logs

# For Google Colab
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
tbc = TensorBoardColab()

import csv

# Open a CSV file to append experiment logs
experiment_log_path = os.path.join(target_directory, 'experiment_log.csv')

# Check if the file exists to write headers
write_headers = not os.path.exists(experiment_log_path)

with open(experiment_log_path, 'a', newline='') as csvfile:
    fieldnames = ['Experiment_ID', 'Date', 'Hyperparameters', 'Avg_Train_Loss', 'Avg_Val_Loss', 'Avg_Train_Acc', 'Avg_Val_Acc', 'Comments']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if write_headers:
        writer.writeheader()
    
    # Log experiment details
    writer.writerow({
        'Experiment_ID': f'Exp_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        'Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Hyperparameters': f'LR={learning_rate}, HS={hidden_size}, NL={num_layers}',
        'Avg_Train_Loss': avg_train_loss,
        'Avg_Val_Loss': avg_val_loss,
        'Avg_Train_Acc': avg_train_accuracy,
        'Avg_Val_Acc': avg_val_accuracy,
        'Comments': 'Added cross-validation and increased epochs'
    })

!nvidia-smi

import torch

# Check total and available GPU memory
def print_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_memory = reserved_memory - allocated_memory
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Reserved GPU Memory: {reserved_memory:.2f} GB")
        print(f"Allocated GPU Memory: {allocated_memory:.2f} GB")
        print(f"Free GPU Memory: {free_memory:.2f} GB")
    else:
        print("CUDA is not available.")

# Call this function at desired points in your code
print_gpu_memory()

# Define accumulation steps
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    
    outputs = model(inputs)
    outputs_flat = outputs.view(-1, num_classes)
    targets_flat = targets.view(-1)
    
    loss = criterion(outputs_flat, targets_flat)
    loss = loss / accumulation_steps  # Normalize loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            outputs_flat = outputs.view(-1, num_classes)
            targets_flat = targets.view(-1)
            loss = criterion(outputs_flat, targets_flat)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# After validation phase
del inputs, targets, outputs, loss
torch.cuda.empty_cache()

target_directory = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/results/'

# Replace 'best_model.pth' with the full path
best_model_path = os.path.join(target_directory, 'best_model.pth')

# Inside your training function
torch.save(model.state_dict(), best_model_path)
print(f"Best model saved to {best_model_path}")

# For saving checkpoints
checkpoint_path = os.path.join(target_directory, f'checkpoint_epoch_{epoch+1}.pth')
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")

# Load the best model
best_model_path = os.path.join(target_directory, 'best_model.pth')
model.load_state_dict(torch.load(best_model_path, map_location=device))
"""