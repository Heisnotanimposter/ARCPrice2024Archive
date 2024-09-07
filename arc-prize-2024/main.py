# main.py
import logging
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from setup import load_dataset
from helper import json_to_image, plot_pic, plot_task
from dataset import ARCDataset
from model import VisionTransformer
from training import train_vit
from submission import set_submission, save_submission

logging.basicConfig(filename='arc_project.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(message)s')


def main():

    logging.info("Starting ARC Prize 2024 Project...")

    try:
        # Define paths - kaggle version
        train1_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_training_challenges.json'
        train2_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_training_solutions.json'
        eval1_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_evaluation_challenges.json'
        eval2_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_evaluation_solutions.json'
        test_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/arc-agi_test_challenges.json'
        sample_path = '/Users/seungwonlee/ARC Prize 2024/ARCPrice2024/arc-prize-2024/sample_submission.json'

        # Load the dataset
        logging.info("Loading dataset...")
        train_dataset = ARCDataset(train1_path)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        #train_loader, val_loader = load_dataset()

        # Initialize the model
        logging.info("Initializing the model...")
        model = VisionTransformer(input_size=100, num_classes=100)  # Ensure this matches your actual input and class sizes
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        # Train the model
        logging.info("Starting training...")
        train_vit(model, train_loader, optimizer, criterion)
        
        # Generate submission file
        logging.info("Generating submission file...")
        submission = set_submission()
        save_submission(submission, sample_path)
        logging.info("Submission file saved successfully.")
                
        print("Task Complete")
      
    except Exception as e:
        logging.error(f"Error in main process: {e}", exc_info=True)
        print(f" error: {e}")

"""
    # Load and prepare the training dataset
    print("Loading dataset...")
    train_dataset = ARCDataset(train1_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model
    print("Initializing the model...")
    model = VisionTransformer(input_size=100, num_classes=100)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Train the model
    print("Starting training...")
    train_vit(model, train_loader, optimizer, criterion)

    # Generate submission file
    print("Generating submission file...")
    submission = set_submission()

    print("All tasks completed. Submission file 'submission.json' created successfully.")
"""

if __name__ == "__main__":
    main()
