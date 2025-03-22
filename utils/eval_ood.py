import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb    
import urllib.request

from .utils import class_names


def evaluate_ood(model, distortion_name, severity, config):
    data_dir = "./data/ood-test"
    device = config.DEVICE

    # Load the OOD images
    images = np.load(os.path.join(data_dir, f"{distortion_name}.npy"))

    # Select the subset of images for the given severity
    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    images = images[start_index:end_index]

    # Convert to PyTorch tensors and create DataLoader
    images = torch.from_numpy(images).float() / 255.  # Normalize to [0, 1]
    images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True)

    # Normalize after converting to tensor
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    predictions = []  # Store predictions
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=f"Evaluating {distortion_name} (Severity {severity})", leave=False):
            inputs = inputs[0]
            inputs = normalize(inputs) # Apply normalization
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return predictions


def evaluate_ood_ovd(model, text_encoding, distortion_name, severity, config):
    data_dir = "./data/ood-test"
    device = config.DEVICE

    # Load the OOD images
    images = np.load(os.path.join(data_dir, f"{distortion_name}.npy"))

    # Select the subset of images for the given severity
    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    images = images[start_index:end_index]

    # Convert to PyTorch tensors and create DataLoader
    images = torch.from_numpy(images).float() / 255.  # Normalize to [0, 1]
    images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True)

    # Normalize after converting to tensor
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    predictions = []  # Store predictions
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=f"Evaluating {distortion_name} (Severity {severity})", leave=False):
            inputs = inputs[0]
            inputs = normalize(inputs) # Apply normalization
            inputs = inputs.to(device)

            image_encoding = model(inputs)
            similarity = (100.0 * image_encoding @ text_encoding.T).softmax(dim=-1)
            predicted = similarity.argmax(axis=1)
            predictions.extend(predicted.cpu().numpy())
    return predictions


def evaluate_ood_hierarchical(model, fine_models, coarse_classes, distortion_name, severity, config):
    data_dir = "./data/ood-test"
    device = config.DEVICE

    # Load the OOD images
    images = np.load(os.path.join(data_dir, f"{distortion_name}.npy"))

    # Select the subset of images for the given severity
    start_index = (severity - 1) * 10000
    end_index = severity * 10000
    images = images[start_index:end_index]

    # Convert to PyTorch tensors and create DataLoader
    images = torch.from_numpy(images).float() / 255.  # Normalize to [0, 1]
    images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True)

    # Normalize after converting to tensor
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    predictions = []  # Store predictions
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc=f"Evaluating {distortion_name} (Severity {severity})", leave=False):
            inputs = inputs[0]
            inputs = normalize(inputs)  # Apply normalization
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, classes_predicted = outputs.max(1)

            for i in range(inputs.shape[0]):
                input = inputs[i]
                input = input.to(device)
                fine_model = fine_models[coarse_classes[classes_predicted[i].item()]]
                fine_model.to(device)
                fine_model.eval()
                output = fine_model(input)
                _, predicted = output.max(1)

                predictions.extend(predicted.cpu().numpy())
    return predictions


# Check if the files are already downloaded
def files_already_downloaded(directory, num_files):
    for i in range(num_files):
        file_name = f"distortion{i:02d}.npy"
        file_path = os.path.join(directory, file_name)
        if not os.path.exists(file_path):
            return False
    return True


def evaluate_ood_test(model, config):
    data_dir = "./data/ood-test"
    device = config.DEVICE

    num_files = 19  # Number of files to download

    # Only download if files aren't already downloaded
    if not files_already_downloaded(data_dir, num_files):
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Base URL for the files
        base_url = "https://github.com/DL4DS/ood-test-files/raw/refs/heads/main/ood-test/"

        # Download files distortion00.npy to distortion18.npy
        for i in range(num_files):
            file_name = f"distortion{i:02d}.npy"
            file_url = base_url + file_name
            file_path = os.path.join(data_dir, file_name)

            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded {file_name} to {file_path}")

        print("All files downloaded successfully.")
    else:
        print("All files are already downloaded.")

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]

    all_predictions = []  # Store all predictions for the submission file

    model.eval()  # Ensure model is in evaluation mode
    for distortion in distortions:
        for severity in range(1, 6):
            predictions = evaluate_ood(model, distortion, severity, config)
            all_predictions.extend(predictions)  # Accumulate predictions
            print(f"{distortion} (Severity {severity})")

    return all_predictions


def evaluate_ood_test_ovd(model, text_encoding, config):
    data_dir = "./data/ood-test"
    device = config.DEVICE

    num_files = 19  # Number of files to download

    # Only download if files aren't already downloaded
    if not files_already_downloaded(data_dir, num_files):
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Base URL for the files
        base_url = "https://github.com/DL4DS/ood-test-files/raw/refs/heads/main/ood-test/"

        # Download files distortion00.npy to distortion18.npy
        for i in range(num_files):
            file_name = f"distortion{i:02d}.npy"
            file_url = base_url + file_name
            file_path = os.path.join(data_dir, file_name)

            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded {file_name} to {file_path}")

        print("All files downloaded successfully.")
    else:
        print("All files are already downloaded.")

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]

    all_predictions = []  # Store all predictions for the submission file

    model.eval()  # Ensure model is in evaluation mode
    for distortion in distortions:
        for severity in range(1, 6):
            predictions = evaluate_ood_ovd(model, text_encoding, distortion, severity, config)
            all_predictions.extend(predictions)  # Accumulate predictions
            print(f"{distortion} (Severity {severity})")

    return all_predictions


def evaluate_ood_test_hierarchical(model, fine_models, coarse_classes, config):
    data_dir = "./data/ood-test"
    device = config.DEVICE

    num_files = 19  # Number of files to download

    # Only download if files aren't already downloaded
    if not files_already_downloaded(data_dir, num_files):
        # Create the directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Base URL for the files
        base_url = "https://github.com/DL4DS/ood-test-files/raw/refs/heads/main/ood-test/"

        # Download files distortion00.npy to distortion18.npy
        for i in range(num_files):
            file_name = f"distortion{i:02d}.npy"
            file_url = base_url + file_name
            file_path = os.path.join(data_dir, file_name)

            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded {file_name} to {file_path}")

        print("All files downloaded successfully.")
    else:
        print("All files are already downloaded.")

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]

    all_predictions = []  # Store all predictions for the submission file

    model.eval()  # Ensure model is in evaluation mode
    for distortion in distortions:
        for severity in range(1, 6):
            predictions = evaluate_ood_hierarchical(model, fine_models, coarse_classes, distortion, severity, config)
            all_predictions.extend(predictions)  # Accumulate predictions
            print(f"{distortion} (Severity {severity})")

    return all_predictions


def create_ood_df(all_predictions):

    distortions = [f"distortion{str(i).zfill(2)}" for i in range(19)]
    
    # --- Create Submission File (OOD) ---
    # Create IDs for OOD (assuming the order is as evaluated)
    ids_ood = []
    for distortion in distortions:
        for severity in range(1, 6):
            for i in range(10000):
              ids_ood.append(f"{distortion}_{severity}_{i}")

    submission_df_ood = pd.DataFrame({'id': ids_ood, 'label': all_predictions})
    return submission_df_ood

