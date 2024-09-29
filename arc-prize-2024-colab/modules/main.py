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
from train import VisionTransformer, train_vit
from submission import set_submission, save_submission

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

logging.basicConfig(filename='arc_project.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(message)s')

def main():

    logging.info("Starting ARC Prize 2024 Project...")

    model = VisionTransformer(input_size=100, patch_size=10, embedding_dim=768, num_classes=100, num_heads=8, num_layers=6)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader  

    train_vit(model, dataloader, optimizer, criterion, epochs=10)

    try:
        # Define paths - colab version

        train1_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_training_challenges.json'
        
        train2_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_training_solutions.json'
        
        eval1_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_evaluation_challenges.json'
        
        eval2_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_evaluation_solutions.json'
        
        test_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_test_challenges.json'
        
        sample_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/sample_submission.json'
        
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


