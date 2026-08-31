import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from   matplotlib import colors
plt.rcParams["font.family"] = "serif"

import json
import time
from tqdm import tqdm
from IPython.display import clear_output

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'arc-prize-2024:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F67357%2F8951125%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240919%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240919T085807Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D14654d1f0a6d4bdde404e3207658a4b7c633474835bce8a1f473bbec190a8b35a28b9387fcb54f9a28890bd36fbc38794b70767ab130a16a55173f8f93848807ad95b2c8ba707b94a4eb9323fc8ef2657b6f6e9b1bae9a768a5ef6ec98bf3cef095302355e2c5313b28c7b66bd0d7e4c88eadeab53c4c50711d1b4afa922593ee2fee7397516b002b077066604a477daf8f4832ca07a76d3eb9e95bcdbd59d1e23313fd54335353a70fe5709eace9e4c1693d5e05884ffa1ab0b09ea88e1c7894a5fbcdf68f7968b0d0b39d23e64badb69465192cfa57001df82d91fb89c28392e5c88956400592f2984a944cb69868f28404818e67cf5b7260e29339892b08f,arc-cnn-solver:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F5682992%2F9370433%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240919%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240919T085807Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3Dacfd0577b3ffad96a8349d80f90702997746f02961554b93a3299462f6964c70bc8ec9bb8f977dfebf082fa2b8a9add6d45f38b8fb0b72ee7491b1c71a50317762612ee288eaa2a2e5c22ef34bb43a0127e00bb8ce354ee661d2d9a60c965938388c6bd565e2d7b0af041d11b671d4f5ee30526ff11fbcfd022fd76dc7ccbd60196bd470fb6aaa25cf38613c4e02ab0abb2aec32c4fcc9b733275822a61871dc252483bc6980dfa12a5ae668c6268d6d535a49da0c2bb82c11e177861d5240a07fc182e7ceec7e66cec2e99a61f3a6ffa49fb15de02c941966d72760255212bd90cf40bfccded26f9af03c5f2b124536762fd2d0cee201579f596311ff4c77ba'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)



try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

def pad_to_30x30(grid, pad_value=10):
    """Pads the grid to a size of 30x30 with the specified pad_value."""
    padded_grid = torch.full((30, 30), pad_value, dtype=torch.int8)
    height, width = len(grid), len(grid[0])
    padded_grid[:height, :width] = torch.tensor(grid, dtype=torch.int8)
    return padded_grid

def preprocess_images(dataset):
        one_hot_images = F.one_hot(dataset, num_classes=11)
        one_hot_images = one_hot_images.permute(0, 3, 1, 2)
        return one_hot_images.float()

def count_parameters(model):
    """Count the trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

_cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', "#FFFFFF"])
_norm = colors.Normalize(vmin=0, vmax=10)

def plot_img(input_matrix, ax, title=""):
    ax.imshow(input_matrix, cmap=_cmap, norm=_norm)
    ax.grid(True, which = 'both',color = 'lightgrey', linewidth = 0.5)
    ax.set_yticks([x-0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1 + len(input_matrix[0]))])
    if len(title) > 0: ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

plt.figure(figsize=(3, 1), dpi=100)
plt.imshow([list(range(10+1))], cmap=_cmap, norm=_norm)
plt.xticks(list(range(10+1)))
plt.yticks([])
plt.show()

with open("/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json", 'r') as f:
    train_challenges = json.load(f)
with open('/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json', 'r') as f:
    train_solutions = json.load(f)
train_ids = list(train_challenges.keys())

# that re-arc tasks (but not train set by itself) were in train
train_ids_train = [train_ids[j] for j in [342, 318, 306, 314,  94, 145, 236, 177, 358, 251, 248, 322, 294,  76, 111, 367, 243, 182, 363, 278, 109, 293, 239,  37, 235, 113, 291, 105, 155, 262,  92, 133, 283,  11, 237, 214,  75, 240, 346, 267, 186,  19, 385, 265, 203, 268,  16, 274, 365, 132, 170, 175, 362, 227,  70, 344,  14, 370,  72, 296, 246,  61, 302, 189, 118, 181, 127, 230, 135, 374, 357,  52, 192, 110,  58, 209, 151,   1, 210, 372,  35, 353, 300, 141,  80, 150, 247, 185, 315, 379,  53, 147, 121, 208, 389, 215, 169, 104, 153, 154,  50, 202, 361, 288, 329, 301,  96, 305, 241, 124,  95, 303, 343,  45, 348, 399, 368, 325,  27, 161, 279, 233, 125, 347,  59, 304, 371, 339,  28, 295, 137, 188, 200, 266, 252, 228, 312, 146, 244, 320, 259, 375,  66,  90, 359, 297, 224,  41, 392,  30,  33,  18, 160, 213, 280,  15, 101, 377,  77, 383,  60, 282, 281,  93, 345,  13,  88, 284, 270, 174,  48, 388,  85,  97, 341, 144, 330, 255, 226,  68, 171, 231,   4, 271, 116, 221, 211, 190, 366, 134, 352, 328,  49, 275,  79,  25,  34, 378, 334, 258, 326, 156, 212, 242, 166, 232, 395, 164,  64,  24,   3, 162, 130, 136, 234, 310, 307,  91, 311, 250, 398, 229, 257, 397, 273, 163, 272, 103, 114, 165, 100, 245,  36, 129, 321, 117, 225, 356, 219, 290, 102, 292, 269, 106, 168,  73, 380,  69, 384,  89,  87,   9, 159, 396,  78,  56,  65,  43, 126,  26, 360, 317,  74,  31,   2,  20, 238, 308, 119, 386, 327,  99, 128,  57, 201, 107, 148, 349,  12, 313, 123, 390,  21, 183, 309,   0, 217, 393, 143, 394,  47, 115, 376, 158, 373, 335,  46,  39, 195, 369]]
# that re-arc tasks were not in train
train_ids_test = [train_ids[j] for j in [196,  51,  82, 149, 299,  29, 331, 180, 206, 287, 178, 152, 199, 108, 276, 176,  22, 254, 172,  32,  83, 350, 387, 131,  17, 323,  38, 338,   6, 204, 193, 316, 391,  62, 249, 263, 364, 340, 286, 336,  86, 198, 289, 355, 354, 382,  40, 319, 332, 222, 140, 142, 187, 173, 157, 351,  54, 138,  71, 167, 298,  10, 285,  44, 220, 207, 223, 120, 264, 197, 139, 218,  42, 337,  98,  23,   5, 256,  63, 194, 260, 277, 324, 261, 216,   8,  81, 381, 253, 112, 122,  55,  84, 191,  67,   7, 179, 205, 184, 333]]


# Define Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)        # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # (N, query_len, embed_size)

        out = self.fc_out(out)  # (N, query_len, embed_size)
        return out

# Define Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # (1, max_len, embed_size)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

# Define Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask)

        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)

        # Add & Norm
        out = self.dropout(self.norm2(forward + x))
        return out

# Define Multi-Head Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, num_layers, max_len=100):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = self.positional_encoding(x)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# Define CNN Feature Extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_size=512):
        super(CNNFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(11, 64, kernel_size=3, padding=1),  # Assuming 11 input channels
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # Downsample by 2
            nn.Conv2d(64, embed_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(embed_size),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Shape: (batch_size, embed_size, H', W')
        x = x.view(x.size(0), x.size(1), -1)  # Flatten spatial dimensions: (batch_size, embed_size, H'*W')
        x = x.permute(0, 2, 1)  # (batch_size, H'*W', embed_size)
        return x

# Define Autoencoder Encoder
class AutoencoderEncoder(nn.Module):
    def __init__(self, embed_size=512, latent_dim=256):
        super(AutoencoderEncoder, self).__init__()
        self.fc = nn.Linear(embed_size, latent_dim)

    def forward(self, x):
        latent = self.fc(x)  # (batch_size, H'*W', latent_dim)
        return latent

# Define Autoencoder Decoder
class AutoencoderDecoder(nn.Module):
    def __init__(self, latent_dim=256, embed_size=512, output_size=11, H=7, W=7):
        super(AutoencoderDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, embed_size)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(embed_size, 64, kernel_size=2, stride=2),  # Upsample by 2
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, output_size, kernel_size=2, stride=2),  # Upsample by 2
            nn.Sigmoid()  # Assuming output is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.fc(x)  # (batch_size, embed_size)
        x = x.view(x.size(0), -1, 1, 1)  # Reshape to (batch_size, embed_size, 1, 1)
        x = self.conv_layers(x)  # (batch_size, output_size, H, W)
        return x