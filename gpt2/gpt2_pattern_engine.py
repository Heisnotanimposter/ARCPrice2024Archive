from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Custom Dataset class for our 2D grid task
class GridDataset(Dataset):
    def __init__(self, input_grids, output_grids):
        self.inputs = input_grids
        self.outputs = output_grids

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
                "labels": torch.tensor(self.outputs[idx], dtype=torch.long)}

# Tokenizer and Model initialization
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Sample Training Data
train_input = [
    [0, 1, 0, 0, 0, 0, 0, 1, 0], # Flattened input grid 1
    [0, 0, 0, 1, 0, 0, 0, 0, 0], # Flattened input grid 2
    # Add more flattened inputs as needed
]

train_output = [
    [2, 0, 0, 0, 0, 0, 0, 0, 0], # Flattened output grid 1
    [2, 2, 0, 0, 0, 0, 0, 0, 0], # Flattened output grid 2
    # Add more flattened outputs as needed
]

# Create Dataset
train_dataset = GridDataset(train_input, train_output)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-grid-output",
    evaluation_strategy="steps",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the GPT-2 model
trainer.train()

# Sample Training Data
train_input = [
    [0, 1, 0, 0, 0, 0, 0, 1, 0], # Flattened input grid 1
    [0, 0, 0, 1, 0, 0, 0, 0, 0], # Flattened input grid 2
    # Add more flattened inputs as needed
]

train_output = [
    [2, 0, 0, 0, 0, 0, 0, 0, 0], # Flattened output grid 1
    [2, 2, 0, 0, 0, 0, 0, 0, 0], # Flattened output grid 2
    # Add more flattened outputs as needed
]

# Create Dataset
train_dataset = GridDataset(train_input, train_output)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./gpt2-grid-output",
    evaluation_strategy="steps",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the GPT-2 model
trainer.train()