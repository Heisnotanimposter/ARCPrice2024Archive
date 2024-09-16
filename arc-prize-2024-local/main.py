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

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

logging.basicConfig(filename='arc_project.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(message)s')

def initialize_model(input_size, num_classes, learning_rate=1e-4):
    model = VisionTransformer(input_size=input_size, num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    return model, optimizer, criterion

def main():
    logging.info("Starting ARC Prize 2024 Project...")
    try:
        # Define paths
        paths = {
            'train': '/path/to/train_data.json',
            'val': '/path/to/val_data.json',
            'test': '/path/to/test_data.json',
            'sample_submission': '/path/to/sample_submission.json'
        }
        # Load the dataset
        train_dataset = ARCDataset(paths['train'])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Initialize the model, optimizer, and loss function
        model, optimizer, criterion = initialize_model(input_size=100, num_classes=100)

        # Train the model
        logging.info("Starting training...")
        train_vit(model, train_loader, optimizer, criterion)

        # Generate and save submission file
        logging.info("Generating submission file...")
        submission = set_submission()
        save_submission(submission, paths['sample_submission'])
        
        logging.info("Submission file saved successfully.")
        print("Task Complete")
    except Exception as e:
        logging.error(f"Error in main process: {e}", exc_info=True)
        print(f" error: {e}")

if __name__ == "__main__":
    main()