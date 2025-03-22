import torch
from tqdm.auto import tqdm  # For progress bars
import wandb    

from .utils import class_names

# --- Evaluation on Clean CIFAR-100 Test Set ---
def evaluate_cifar100_test(model, test_loader, device):
    """Evaluation on clean CIFAR-100 test set."""
    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating on Clean Test Set")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            predictions.extend(predicted.cpu().numpy()) # Move predictions to CPU and convert to numpy
            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    clean_accuracy = 100. * correct / total
    # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    return predictions, clean_accuracy


def evaluate_cifar100_test_ovd(model, test_loader, text_encoding, device):
    """Evaluation on clean CIFAR-100 test set."""
    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating on Clean Test Set")):
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


def evaluate_cifar100_test_hierarchical(model, fine_models, test_loader, device):
    """Evaluation on clean CIFAR-100 test set."""
    model.eval()
    correct = 0
    total = 0
    predictions = []  # Store predictions for the submission file
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating on Clean Test Set")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, classes_predicted = outputs.max(1)

            for i in range(labels.shape[0]):
                input = inputs[i]
                input = input.to(device)
                fine_model = fine_models[classes_predicted[i].item()]
                output = fine_model(input)
                _, predicted = output.max(1)

                predictions.extend(predicted.cpu().numpy()) # Move predictions to CPU and convert to numpy

            total += labels.size(0)
            correct += predicted.eq(labels.to(device)).sum().item()

    clean_accuracy = 100. * correct / total
    # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")
    return predictions, clean_accuracy