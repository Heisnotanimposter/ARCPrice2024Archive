from google.colab import drive
drive.mount('/content/drive')

!pip install colorama
!pip install utils
!pip install xLSTM
!pip install GPUtil
!pip install e2cnn

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random

import psutil
import GPUtil
import csv
import time

from e2cnn import gspaces
from e2cnn import nn as enn

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
r2_act = gspaces.Rot2dOnR2(N=4)
in_type = enn.FieldType(r2_act, 11 * [r2_act.trivial_repr])
out_type = enn.FieldType(r2_act, 64 * [r2_act.regular_repr])
equiv_conv = enn.R2Conv(in_type, out_type, kernel_size=3, padding=1)



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
        pad_grid = np.zeros((size, size), dtype=grid.dtype)
        pad_grid[:h, :w] = grid
        return pad_grid

    def grid_to_sequence(grid):
        return grid.flatten()

    def sequence_to_grid(sequence, size):
        return sequence.reshape(size, size)

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

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(11, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

class ARCDataset(Dataset):

    def __init__(self, inputs, outputs, augment=False):
        self.inputs = inputs
        self.outputs = outputs
        self.augment = augment

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_grid = self.inputs[idx]
        output_grid = self.outputs[idx]

        if self.augment:
            augmented_inputs, augmented_outputs = augment_with_symmetry(input_grid, output_grid)
            # Randomly select one augmentation
            rand_idx = random.randint(0, len(augmented_inputs) - 1)
            input_grid = augmented_inputs[rand_idx]
            output_grid = augmented_outputs[rand_idx]

        # Convert to tensors
        input_tensor = torch.tensor(input_grid, dtype=torch.float32)
        output_tensor = torch.tensor(output_grid, dtype=torch.long)
        return input_tensor, output_tensor

    def collate_fn(batch):
        inputs, targets = zip(*batch)
        return torch.stack(inputs), torch.stack(targets)

    def augment_with_symmetry(input_grid, output_grid):
        augmented_inputs = []
        augmented_outputs = []

        # Original
        augmented_inputs.append(input_grid)
        augmented_outputs.append(output_grid)

        # Rotations
        for k in [1, 2, 3]:
            augmented_inputs.append(np.rot90(input_grid, k))
            augmented_outputs.append(np.rot90(output_grid, k))

        # Flips
        augmented_inputs.append(np.fliplr(input_grid))
        augmented_outputs.append(np.fliplr(output_grid))

        augmented_inputs.append(np.flipud(input_grid))
        augmented_outputs.append(np.flipud(output_grid))

        # Rotations + Flips
        for k in [1, 2, 3]:
            augmented_inputs.append(np.fliplr(np.rot90(input_grid, k)))
            augmented_outputs.append(np.fliplr(np.rot90(output_grid, k)))

            augmented_inputs.append(np.flipud(np.rot90(input_grid, k)))
            augmented_outputs.append(np.flipud(np.rot90(output_grid, k)))

        return augmented_inputs, augmented_outputs


# Load and preprocess data
train_data_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_training_challenges.json'
inputs, outputs = load_arc_data(train_data_path)

# Verify data loading
print(f"Number of samples loaded: {len(inputs)}")
if len(inputs) == 0:
    print("No data was loaded. Please check your data paths and preprocessing steps.")
else:
    print(f"Sample input grid shape: {inputs[0].shape}")
    print(f"Sample output grid shape: {outputs[0].shape}")

# Split data, create datasets and dataloaders
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
    
# Load the trained weights (if available)
# Save the trained model
best_model_path = os.path.join(target_directory, 'best_model.pth')
torch.save(model.state_dict(), best_model_path)
print(f"Model saved to {best_model_path}")
def load_test_data(test_data_path):

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    input_tasks = {}

    for task_id, task in test_data.items():
        # For each test example in the task
        for idx, example in enumerate(task["test"]):
            input_grid = np.array(example["input"])
            input_tasks[f"{task_id}_{idx}"] = input_grid

    return input_tasks



# Load test data and run inference
test_data_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_test_challenges.json'
input_tasks = load_test_data(test_data_path)

# Load the trained model for inference
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.to(device)
infer_and_save(model, input_tasks)

def symmetry_loss(output):
    # Assuming output shape is [batch_size, channels, height, width]
    output_flipped = torch.flip(output, dims=[-1])  # Horizontal flip
    loss = F.mse_loss(output, output_flipped)
    return loss

total_loss = original_loss + lambda_symmetry * symmetry_loss(output_pred)
channels = [256, 512, 768]
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        return out + x  # Residual connection

train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=num_epochs, patience=patience)

# Now use this function to load the test data and run inference
#train_data_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024-colab/arc-agi_training_challenges.json'
#inputs, outputs = load_arc_data(train_data_path)
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


def log_resource_usage(log_file_path, epoch, step):
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