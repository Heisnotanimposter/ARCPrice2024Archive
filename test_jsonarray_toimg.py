import numpy as np
from PIL import Image

# Example JSON data (this would be replaced with your actual data loading logic)
json_data = {
    "grid": [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]
}

# Convert grid to an image
array = np.array(json_data["grid"], dtype=np.uint8) * 255  # Scale to 0-255
img = Image.fromarray(array, 'L')  # 'L' mode is for grayscale
img.save('output_image.png')