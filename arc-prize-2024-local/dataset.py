# dataset.py

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from helper import json_to_image  # Add this line to import json_to_image

class ARCDataset(Dataset):
    def __init__(self, data_path, target_size=(10, 10)):
        with open(data_path, 'r') as f:
            self.data = list(json.load(f).values())
        self.target_size = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_image = self.resize_to_target(json_to_image(item['train'][0]['input']))
        output_image = self.resize_to_target(json_to_image(item['train'][0]['output']))
        return input_image, output_image

    def resize_to_target(self, image):
        """Resize the input image to the target size using padding if necessary."""
        target_height, target_width = self.target_size
        
        if isinstance(image, list):
            image = np.array(image)
        if isinstance(image, np.ndarray):
            image = torch.tensor(image)
        
        img_height, img_width = image.shape[-2], image.shape[-1]
        new_image = torch.zeros((1, target_height, target_width), dtype=image.dtype)
        
        padding_top = max((target_height - img_height) // 2, 0)
        padding_left = max((target_width - img_width) // 2, 0)

        start_h = padding_top
        end_h = start_h + min(img_height, target_height)
        start_w = padding_left
        end_w = start_w + min(img_width, target_width)
        
        img_start_h = 0
        img_end_h = min(img_height, target_height)
        img_start_w = 0
        img_end_w = min(img_width, target_width)

        new_image[0, start_h:end_h, start_w:end_w] = image[img_start_h:img_end_h, img_start_w:img_end_w]
        return new_image
