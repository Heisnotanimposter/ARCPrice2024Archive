# dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import json  # Ensure you import json

MAX_GRID_SIZE = 30

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

# Move collate_fn outside the class
def collate_fn(batch):
    inputs, targets = zip(*batch)
    return torch.stack(inputs), torch.stack(targets)

# Move augment_with_symmetry outside the class
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

# Other utility functions
def pad_grid(grid, size):
    h, w = grid.shape
    padded_grid = np.zeros((size, size), dtype=grid.dtype)
    padded_grid[:h, :w] = grid
    return padded_grid

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

def load_test_data(test_data_path):
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    input_tasks = {}
    for task_id, task in test_data.items():
        for idx, example in enumerate(task["test"]):
            input_grid = np.array(example["input"])
            input_tasks[f"{task_id}_{idx}"] = input_grid
    return input_tasks