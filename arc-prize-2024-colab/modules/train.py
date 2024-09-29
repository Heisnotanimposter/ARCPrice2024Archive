#train.py
import torch
import torch.nn as nn
from tqdm import tqdm

class VisionTransformer(nn.Module):
    def __init__(self, input_size=10, patch_size=10, embedding_dim=768, num_classes=100, num_heads=8, num_layers=6):
        super(VisionTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2

        self.embedding = nn.Linear(patch_size * patch_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = x.to(self.embedding.weight.dtype)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.size(0), -1, self.patch_size * self.patch_size)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        output = self.fc(x)
        return output

def train_vit(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        tepoch = tqdm(dataloader, unit="batch", desc=f"Epoch {epoch+1}/{epochs}")
        for batch in tepoch:
            inputs, targets = batch
            inputs, targets = inputs.float(), targets.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, 1, 10, 10)  # Adjust based on your target dimensions
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tepoch.set_postfix(loss=total_loss / len(dataloader))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")