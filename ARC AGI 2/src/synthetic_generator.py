"""
ARC-AGI-2 Synthetic Task Generator
Inspired by 2025 Winner NVARC Strategy (Synthetic Augmentation for 4B Models)
"""

import random
import json
import numpy as np

class SyntheticARCGenerator:
    """Generates synthetic grid transformation tasks using composable primitives."""

    def __init__(self, min_size=3, max_size=30):
        self.min_size = min_size
        self.max_size = max_size

    def generate_random_grid(self, height=None, width=None, num_colors=4):
        h = height or random.randint(self.min_size, 15)
        w = width or random.randint(self.min_size, 15)
        grid = np.random.randint(0, num_colors, size=(h, w))
        return grid.tolist()

    def transform_recolor(self, grid, color_map):
        """Applies a color permutation to the grid."""
        arr = np.array(grid)
        new_arr = np.copy(arr)
        for old_c, new_c in color_map.items():
            new_arr[arr == old_c] = new_c
        return new_arr.tolist()

    def transform_rotate(self, grid, k=1):
        """Rotates the grid by 90 * k degrees."""
        return np.rot90(np.array(grid), k=k).tolist()

    def generate_task_pair(self, num_examples=3):
        """Generates a complete synthetic task with train and test pairs."""
        color_map = {1: 2, 2: 3, 3: 1}
        train_pairs = []

        for _ in range(num_examples):
            inp = self.generate_random_grid()
            out = self.transform_rotate(self.transform_recolor(inp, color_map), k=1)
            train_pairs.append({"input": inp, "output": out})

        test_input = self.generate_random_grid()
        test_output = self.transform_rotate(self.transform_recolor(test_input, color_map), k=1)

        return {
            "train": train_pairs,
            "test": [{"input": test_input, "output": test_output}]
        }

if __name__ == "__main__":
    generator = SyntheticARCGenerator()
    task = generator.generate_task_pair()
    print(f"Generated Synthetic Task with {len(task['train'])} train pairs.")
    print(json.dumps(task, indent=2)[:300] + "\n...")
