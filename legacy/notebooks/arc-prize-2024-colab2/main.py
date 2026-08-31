import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_loading import load_arc_data, load_test_data
from dataset import ARCDataset, collate_fn
from models.xLSTM import xLSTMModelClassification
from training import train_model
from inference import infer_and_save
from utils import print_gpu_memory
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate the ARC model.')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training data JSON file.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test data JSON file.')
    parser.add_argument('--target_directory', type=str, default='.', help='Directory to save models and outputs.')
    parser.add_argument('--num_epochs', type=int, default=32, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping.')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for LSTM.')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of LSTM layers.')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess data
    inputs, outputs = load_arc_data(args.train_data_path)

    # Verify data loading
    print(f"Number of samples loaded: {len(inputs)}")
    if len(inputs) == 0:
        print("No data was loaded. Please check your data paths and preprocessing steps.")
        return
    else:
        print(f"Sample input grid shape: {inputs[0].shape}")
        print(f"Sample output grid shape: {outputs[0].shape}")

    # Split data, create datasets and dataloaders
    train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(
        inputs, outputs, test_size=0.1, random_state=42
    )
    train_dataset = ARCDataset(train_inputs, train_outputs, augment=True)
    val_dataset = ARCDataset(val_inputs, val_outputs, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ARCDataset.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Define the model
    model = xLSTMModelClassification(
        input_size=10,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=10,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Define the loss function, optimizer, and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Train the model
    train_model(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        epochs=args.num_epochs, patience=args.patience, device=device, target_directory=args.target_directory
    )

    # Save the trained model
    best_model_path = os.path.join(args.target_directory, 'best_model.pth')
    torch.save(model.state_dict(), best_model_path)
    print(f"Model saved to {best_model_path}")

    # Load the trained model for inference
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    # Load test data and run inference
    input_tasks = load_test_data(args.test_data_path)
    infer_and_save(model, input_tasks, output_file=os.path.join(args.target_directory, 'submission.json'), device=device)

if __name__ == '__main__':
    main()
