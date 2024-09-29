# setup.py
import os
import gc
import sys
import copy
import json
import random
import time
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from colorama import Style, Fore

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Colormap setup
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

def load_dataset(batch_size=32):
    # Define paths - adjust the paths accordingly
    train1_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_training_challenges.json'
        
    train2_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_training_solutions.json'
        
    eval1_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_evaluation_challenges.json'
        
    eval2_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_evaluation_solutions.json'
        
    test_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/arc-agi_test_challenges.json'
        
    sample_path = '/content/drive/MyDrive/2024-2/ARC/ARCPrice2024/arc-prize-2024/sample_submission.json'
    
    # Initialize dataset and dataloader
    train_dataset = ARCDataset(train1_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()
