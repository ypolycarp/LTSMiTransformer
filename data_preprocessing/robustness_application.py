# pip install torch pandas numpy

import pandas as pd
import numpy as np
import torch

def add_gaussian_noise(data, noise_ratio=0.2, sigma=0.1):
    noisy_data = data.clone()
    num_elements = data.numel()
    num_noisy = int(num_elements * noise_ratio)
    indices = torch.randperm(num_elements)[:num_noisy]
    noise = sigma * torch.randn(num_noisy).to(data.device)
    noisy_data.view(-1)[indices] += noise
    return noisy_data

def simulate_missing_data(data, missing_ratio=0.2):
    mask = torch.ones_like(data)
    num_elements = data.numel()
    num_missing = int(num_elements * missing_ratio)
    indices = torch.randperm(num_elements)[:num_missing]
    data_missing = data.clone()
    data_missing.view(-1)[indices] = 0.0
    mask.view(-1)[indices] = 0.0
    return data_missing, mask

def add_gaussian_noise_csv(file_path, noise_ratio=0.2, sigma=0.1, output_path=None):
    df = pd.read_csv(file_path)
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    noisy_tensor = add_gaussian_noise(data_tensor, noise_ratio=noise_ratio, sigma=sigma)
    noisy_df = pd.DataFrame(noisy_tensor.numpy(), columns=df.columns)
    if output_path:
        noisy_df.to_csv(output_path, index=False)
    return noisy_df

'''
    Simulates missing values in a CSV dataset and saves the modified file.
    
    Parameters:
    - filepath (str): Path to the CSV file.
    - missing_ratio (float): Proportion of entries to remove (set to 0 or NaN).
    - use_nan (bool): If True, missing values will be set to NaN. Otherwise, set to 0.0.
    - save_path (str): Output path for the manipulated CSV.
    
    Returns:
    - missing_df (pd.DataFrame): The DataFrame with simulated missing values.
'''
def simulate_missing_data_csv_zero(file_path, missing_ratio=0.2, output_path=None):
    df = pd.read_csv(file_path)
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    data_missing, _ = simulate_missing_data(data_tensor, missing_ratio)
    missing_df = pd.DataFrame(data_missing.numpy(), columns=df.columns)
    if output_path:
        missing_df.to_csv(output_path, index=False)
    return missing_df

def simulate_missing_data_csv_nan(file_path, missing_ratio=0.2, output_path=None):
    df = pd.read_csv(file_path)
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    num_elements = data_tensor.numel()
    num_missing = int(num_elements * missing_ratio)
    indices = torch.randperm(num_elements)[:num_missing]
    data_tensor.view(-1)[indices] = float('nan')
    missing_df_nan = pd.DataFrame(data_tensor.numpy(), columns=df.columns)
    if output_path:
        missing_df_nan.to_csv(output_path, index=False)
    return missing_df_nan

'''
-----------------
Usage Example:
-----------------
add_gaussian_noise_csv("cr_process.csv", output_path="noisy_data.csv")
simulate_missing_data_csv_zero("cr_process.csv", output_path="missing_data_zero.csv")
simulate_missing_data_csv_nan("cr_process.csv", output_path="missing_data_nan.csv")
'''