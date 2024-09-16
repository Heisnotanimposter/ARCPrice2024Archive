import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from faster_rcnn_model import FasterRCNNModel

# Initialize Faster R-CNN model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
faster_rcnn = FasterRCNNModel(device=device, model_type='resnet50_fpn')

# Load and preprocess data
def load_data(file_path):
    """
    Load data from a JSON file.

    Args:
    - file_path (str): Path to the JSON file.

    Returns:
    - data (dict): Loaded JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load training and evaluation data
training_data = load_data(r'C:\Users\redca\Documents\GitHub\ARCPrice2024\arc-agi_training_challenges.json')
evaluation_data = load_data(r'C:\Users\redca\Documents\GitHub\ARCPrice2024\arc-agi_evaluation_challenges.json')

# Define possible transformations as functions
def rotate_90(grid):
    """
    Rotate the grid by 90 degrees.

    Args:
    - grid (np.ndarray): Input grid.

    Returns:
    - rotated_grid (np.ndarray): Rotated grid.
    """
    return np.rot90(grid)

def flip_horizontal(grid):
    """
    Flip the grid horizontally.

    Args:
    - grid (np.ndarray): Input grid.

    Returns:
    - flipped_grid (np.ndarray): Horizontally flipped grid.
    """
    return np.fliplr(grid)

def flip_vertical(grid):
    """
    Flip the grid vertically.

    Args:
    - grid (np.ndarray): Input grid.

    Returns:
    - flipped_grid (np.ndarray): Vertically flipped grid.
    """
    return np.flipud(grid)

# List of transformation functions
transformations = [rotate_90, flip_horizontal, flip_vertical]

# Visualize a grid
def visualize_grid(grid):
    """
    Visualize a 2D grid using matplotlib.

    Args:
    - grid (np.ndarray): 2D grid to visualize.
    """
    plt.imshow(grid, cmap='viridis')
    plt.show()

# Feature extraction using Faster R-CNN
def extract_features(grid):
    """
    Extract features from a grid using Faster R-CNN.

    Args:
    - grid (np.ndarray): Input grid.

    Returns:
    - features (tuple): Extracted features from the model.
    """
    # Convert grid to RGB format
    image = np.stack((grid,) * 3, axis=-1)  # Convert grayscale to RGB by stacking

    # Convert image to a floating-point tensor and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Predict features using Faster R-CNN
    predictions = faster_rcnn.predict(image)
    features = faster_rcnn.postprocess(predictions)
    return features

# Generate possible solutions
def generate_solutions(grid, features):
    """
    Generate possible solutions for a given grid based on extracted features.

    Args:
    - grid (np.ndarray): Input grid.
    - features (tuple): Extracted features.

    Returns:
    - candidates (list of np.ndarray): List of candidate solutions.
    """
    candidates = []
    for transformation in transformations:
        transformed_grid = transformation(grid)
        candidates.append(transformed_grid)
    return candidates

# Evaluate candidate solutions
def evaluate_candidate(candidate):
    """
    Evaluate a candidate solution.

    Args:
    - candidate (np.ndarray): The candidate solution to evaluate.

    Returns:
    - score (float): A score representing the quality of the candidate solution.
    """
    # Example scoring logic: Sum of all elements
    score = np.sum(candidate)
    return score

# Score and rank solutions
def score_solutions(candidates):
    """
    Score and rank candidate solutions.

    Args:
    - candidates (list of np.ndarray): List of candidate solutions.

    Returns:
    - ranked_candidates (list of np.ndarray): Ranked list of candidates based on their scores.
    """
    scores = []
    for candidate in candidates:
        score = evaluate_candidate(candidate)
        scores.append(score)
    
    # Debug: Print scores to check their values and types
    print("Scores:", scores)  # This should output a list of scalar values
    
    # Sort by scalar score values
    ranked_candidates = [x for _, x in sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)]
    
    return ranked_candidates

# Select the best solution
def select_best_solution(ranked_candidates):
    """
    Select the best solution from the ranked candidates.

    Args:
    - ranked_candidates (list of np.ndarray): Ranked list of candidates.

    Returns:
    - best_solution (np.ndarray): The best candidate solution.
    """
    best_solution = ranked_candidates[0]  # Select the highest scoring candidate
    return best_solution

# Main loop to process tasks
def process_tasks(tasks):
    """
    Process tasks to generate and evaluate solutions.

    Args:
    - tasks (dict): Dictionary of tasks to process.
    """
    for task_id, task in tasks.items():
        input_grid = task['train'][0]['input']
        features = extract_features(input_grid)
        candidates = generate_solutions(input_grid, features)
        ranked_candidates = score_solutions(candidates)
        best_solution = select_best_solution(ranked_candidates)
        visualize_grid(best_solution)

# Run the process on training data
process_tasks(training_data)
