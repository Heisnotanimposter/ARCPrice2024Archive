import torch
import torch.nn as nn
import torch.nn.functional as F

class xLSTMModelClassification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.0):
        super(xLSTMModelClassification, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = nn.functional.one_hot(x, num_classes=self.fc.out_features).float()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out.reshape(-1, self.hidden_size)).view(x.size(0), -1, self.fc.out_features)
        return out

    def pad_grid(grid, size):
        h, w = grid.shape
        pad_grid = np.zeros((size, size), dtype=grid.dtype)
        pad_grid[:h, :w] = grid
        return pad_grid

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

