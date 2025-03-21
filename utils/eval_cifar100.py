import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm  # For progress bars
import wandb    

from .utils import class_names

# --- Evaluation on Clean CIFAR-100 Test Set ---
def evaluate_cifar100_test(model, testloader, device):
    """Evaluation on clean CIFAR-100 test set."""
    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(testloader, desc="Evaluating on Clean Test Set")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            predictions.extend(predicted.cpu().numpy()) # Move predictions to CPU and convert to numpy
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    clean_accuracy = 100. * correct / total
    # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    return predictions, clean_accuracy


def evaluate_cifar100_test_ovd(model, testloader, text_encoding, device):
    """Evaluation on clean CIFAR-100 test set."""
    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(testloader, desc="Evaluating on Clean Test Set")):
            inputs = inputs.to(device)
            image_encoding = model(inputs)
            similarity = (100.0 * image_encoding @ text_encoding.T).softmax(dim=-1)
            predicted = similarity.argmax(axis=1)

            predictions.extend(predicted.cpu().numpy()) # Move predictions to CPU and convert to numpy
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    clean_accuracy = 100. * correct / total
    # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    return predictions, clean_accuracy