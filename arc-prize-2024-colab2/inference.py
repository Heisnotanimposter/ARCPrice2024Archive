import torch
import numpy as np
import json
from tqdm import tqdm
from data_loading import grid_to_sequence, pad_grid, sequence_to_grid, MAX_GRID_SIZE

def infer_and_save(model, input_tasks, output_file="submission.json", device='cpu'):
    model.eval()
    results = {}
    for task_id, input_grid in tqdm(input_tasks.items(), desc="Running Inference"):
        input_sequence = grid_to_sequence(pad_grid(input_grid, MAX_GRID_SIZE))
        input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output_sequence = model(input_tensor).argmax(-1).squeeze(0).cpu().numpy()
        output_grid = sequence_to_grid(output_sequence, MAX_GRID_SIZE)
        results[task_id] = output_grid.tolist()

    with open(output_file, "w") as f:
        json.dump(results, f)
    print(f"Submission saved to {output_file}")