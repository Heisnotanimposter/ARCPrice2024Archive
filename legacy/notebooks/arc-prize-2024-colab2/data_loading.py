import os
import json
import numpy as np

MAX_GRID_SIZE = 30

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