import torch
import psutil
import GPUtil
from datetime import datetime
import csv
import os

def print_gpu_memory():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert bytes to GB
        reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_memory = reserved_memory - allocated_memory
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Reserved GPU Memory: {reserved_memory:.2f} GB")
        print(f"Allocated GPU Memory: {allocated_memory:.2f} GB")
        print(f"Free Reserved GPU Memory: {free_memory:.2f} GB\n")
    else:
        print("CUDA is not available.")

def log_resource_usage(log_file_path, epoch, step):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_data = {}
    if gpus:
        gpu = gpus[0]  # Modify if using multiple GPUs
        gpu_data = {
            'gpu_load_percent': gpu.load * 100,
            'gpu_memory_util_percent': gpu.memoryUtil * 100,
            'gpu_memory_total_MB': gpu.memoryTotal,
            'gpu_memory_used_MB': gpu.memoryUsed
        }
    else:
        gpu_data = {
            'gpu_load_percent': 0,
            'gpu_memory_util_percent': 0,
            'gpu_memory_total_MB': 0,
            'gpu_memory_used_MB': 0
        }

    # Prepare log entry
    log_entry = {
        'timestamp': timestamp,
        'epoch': epoch,
        'step': step,
        'cpu_usage_percent': cpu_usage,
        'ram_usage_percent': ram_usage,
        **gpu_data
    }

    # Write to CSV file
    file_exists = os.path.isfile(log_file_path)
    with open(log_file_path, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'epoch', 'step', 'cpu_usage_percent', 'ram_usage_percent',
            'gpu_load_percent', 'gpu_memory_util_percent', 'gpu_memory_total_MB', 'gpu_memory_used_MB'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)