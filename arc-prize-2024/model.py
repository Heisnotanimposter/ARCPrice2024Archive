# model.py
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, input_size=10, patch_size=10, embedding_dim=768, num_classes=100, num_heads=8, num_layers=6):
        super(VisionTransformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2

        # Embedding layer
        self.embedding = nn.Linear(patch_size * patch_size, embedding_dim)  # Example: for 10x10 patches, input size = 100
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # Ensure x has the same dtype as the model's layers
        x = x.to(self.embedding.weight.dtype)

        # Split input image into patches and flatten
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.size(0), -1, self.patch_size * self.patch_size)

        # Apply the embedding and transformer encoder
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Aggregate representations from all patches
        output = self.fc(x)
        return output
