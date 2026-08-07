import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from faster_rcnn_model import FasterRCNNModel

# 🎯 **1. Initialize Faster R-CNN model**
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 📌 Check if GPU is available, else use CPU
faster_rcnn = FasterRCNNModel(device=device, model_type='resnet50_fpn')  # 📌 Load Faster R-CNN with ResNet-50 backbone

# 🎯 **2. Load and preprocess data**
def load_data(file_path):
    """
    Load data from a JSON file.
    
    📌 Args:
    - file_path (str): Path to the JSON file.
    
    📌 Returns:
    - data (dict): Loaded JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 📌 Load training and evaluation datasets
training_data = load_data(r'C:\Users\redca\Documents\GitHub\ARCPrice2024\arc-agi_training_challenges.json')
evaluation_data = load_data(r'C:\Users\redca\Documents\GitHub\ARCPrice2024\arc-agi_evaluation_challenges.json')

# 🎯 **3. Define transformation functions**
def rotate_90(grid):
    """📌 Rotate a 2D grid by 90 degrees."""
    return np.rot90(grid)

def flip_horizontal(grid):
    """📌 Flip a 2D grid horizontally."""
    return np.fliplr(grid)

def flip_vertical(grid):
    """📌 Flip a 2D grid vertically."""
    return np.flipud(grid)

# 📌 List of transformation functions to apply to input grids
transformations = [rotate_90, flip_horizontal, flip_vertical]

# 🎯 **4. Visualization function**
def visualize_grid(grid):
    """
    📌 Visualize a 2D grid using matplotlib.
    
    Args:
    - grid (np.ndarray): 2D grid to visualize.
    """
    plt.imshow(grid, cmap='viridis')
    plt.show()

# 🎯 **5. Feature extraction using Faster R-CNN**
def extract_features(grid):
    """
    📌 Extract features from a grid using Faster R-CNN.
    
    Args:
    - grid (np.ndarray): Input grid.
    
    Returns:
    - features (tuple): Extracted features.
    """
    image = np.stack((grid,) * 3, axis=-1)  # 📌 Convert grayscale to RGB by stacking channels
    image = image.astype(np.float32) / 255.0  # 📌 Normalize image to range [0,1]

    predictions = faster_rcnn.predict(image)  # 📌 Get model predictions
    features = faster_rcnn.postprocess(predictions)  # 📌 Extract relevant feature vectors
    return features

# 🎯 **6. Generate possible solutions using transformations**
def generate_solutions(grid, features):
    """
    📌 Generate candidate solutions by applying transformations.
    
    Args:
    - grid (np.ndarray): Input grid.
    - features (tuple): Extracted features.
    
    Returns:
    - candidates (list of np.ndarray): List of transformed grids.
    """
    candidates = [transformation(grid) for transformation in transformations]  # 📌 Apply transformations
    return candidates

# 🎯 **7. Evaluate candidate solutions**
def evaluate_candidate(candidate):
    """
    📌 Evaluate a candidate solution.
    
    Args:
    - candidate (np.ndarray): The candidate solution.
    
    Returns:
    - score (float): A numerical score representing the quality of the candidate.
    """
    score = np.sum(candidate)  # 📌 Example scoring: Sum of all elements in the grid
    return score

# 🎯 **8. Score and rank solutions**
def score_solutions(candidates):
    """
    📌 Score and rank candidate solutions.
    
    Args:
    - candidates (list of np.ndarray): List of candidate solutions.
    
    Returns:
    - ranked_candidates (list of np.ndarray): Solutions ranked from highest to lowest score.
    """
    scores = [evaluate_candidate(candidate) for candidate in candidates]  # 📌 Compute scores for each candidate
    print("Scores:", scores)  # 📌 Debugging output: Check score values

    ranked_candidates = [x for _, x in sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)]  
    # 📌 Sort candidates based on scores (highest to lowest)

    return ranked_candidates

# 🎯 **9. Select the best solution**
def select_best_solution(ranked_candidates):
    """
    📌 Select the best solution from ranked candidates.
    
    Args:
    - ranked_candidates (list of np.ndarray): Ordered list of candidate solutions.
    
    Returns:
    - best_solution (np.ndarray): The top candidate solution.
    """
    return ranked_candidates[0]  # 📌 Choose the highest-scoring candidate

# 🎯 **10. Process tasks and find solutions**
def process_tasks(tasks):
    """
    📌 Process tasks, generate and evaluate candidate solutions.
    
    Args:
    - tasks (dict): Dictionary of tasks to process.
    """
    for task_id, task in tasks.items():
        input_grid = np.array(task['train'][0]['input'])  # 📌 Convert input to NumPy array
        features = extract_features(input_grid)  # 📌 Extract features using Faster R-CNN
        candidates = generate_solutions(input_grid, features)  # 📌 Generate transformed grids
        ranked_candidates = score_solutions(candidates)  # 📌 Rank candidates based on scores
        best_solution = select_best_solution(ranked_candidates)  # 📌 Choose the best transformation
        visualize_grid(best_solution)  # 📌 Display the best solution

# 🎯 **11. Run the process on training data**
process_tasks(training_data)  # 📌 Apply the pipeline to the ARC training dataset