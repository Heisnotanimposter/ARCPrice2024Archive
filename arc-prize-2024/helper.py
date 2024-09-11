import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import logging

version = {
    'solver1': 3,  # Solver 1 has 3 variations
    'solver2': 2,  # Solver 2 has 2 variations
}

# Define solvers
def basic_solver_1(input_data, task):
    try:
        some_transformed_output = perform_transformation(input_data)
        logging.info(f"Solver1 Output: {some_transformed_output}")
        return [some_transformed_output]
    except Exception as ex:
        logging.error(f"Error in solver1: {ex}")
        return []

def basic_solver_2(input_data, task):
    try:
        some_other_output = perform_another_transformation(input_data)
        logging.info(f"Solver2 Output: {some_other_output}")
        return [some_other_output]
    except Exception as ex:
        logging.error(f"Error in solver2: {ex}")
        return []# Mapping solvers
    
solvers = {
    'solver1': basic_solver_1,
    'solver2': basic_solver_2,
}

# Color map and normalization for plotting
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)

def json_to_image(json_array):
    """Convert JSON array to image format (numpy array)."""
    return np.array(json_array)

def plot_pic(x):
    """Plot a single image."""
    plt.imshow(np.array(x), cmap=cmap, norm=norm)
    plt.show()

def plot_task(task):
    """Plot the train and test images for a given task."""
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n, 8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        fig_num += 1
    
    for i, t in enumerate(task["test"]):
        t_in = np.array(t["input"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        fig_num += 1
    
    plt.tight_layout()
    plt.show()

# Placeholder transformation functions
def perform_transformation(input_data):
    """Transform the input data for solver1."""
    # Your transformation logic here
    return input_data  # Replace this with the actual transformation

def perform_another_transformation(input_data):
    """Transform the input data for solver2."""
    # Your transformation logic here
    return input_data  # Replace this with the actual transformation
