import os
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split 

from src.models.resnet_cifar import ResNet152_CIFAR
from utils import *
from utils.utils import coarse_to_fine, class_names, fine_to_coarse


base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


def main():
    main_folder = "./trained_weights/hierarchical_models"
    coarse_model_folder = os.path.join(main_folder, "coarse_model")
    fine_models_folder = {}
    coarse_to_fine_idx = {}
    coarse_classes = list(coarse_to_fine.keys())
    for i, coarse_class in enumerate(coarse_classes):
        model_path = os.path.join(main_folder, f"ResNet152_CIFAR_{coarse_class}", "decompiled_model.pth")
        fine_models_folder[coarse_class] = model_path
        
        for j, fine_class in enumerate(coarse_to_fine[coarse_class]):
            fine_idx_in_all= class_names.index(fine_class)
            coarse_to_fine_idx[f"c{i}_f{j}"] = fine_idx_in_all

    model_path = os.path.join(coarse_model_folder, "decompiled_model.pth")
    model = ResNet152_CIFAR(len(coarse_classes), True)
    model.load_state_dict(torch.load(model_path))
    model.to("cuda")

    dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=base_transform
    )
    test_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )

    model.eval()
    coarse_predictions = []  
    images = []
    gt_labels = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating on Clean Test Set")):
            inputs = inputs.to("cuda")
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            coarse_predictions.extend(predicted.cpu().numpy())
            images.extend(inputs.cpu().numpy())
            gt_labels.extend(labels.numpy())

    final_pred = []
    final_labels = []
    for i, coarse_class in enumerate(coarse_classes):
        num_classes = len(coarse_to_fine[coarse_class])
        model_path = fine_models_folder[coarse_class]
        model = ResNet152_CIFAR(num_classes, True)
        model.load_state_dict(torch.load(model_path))
        model.to("cuda")
        model.eval()

        with torch.no_grad():
            for j, coarse_pred in enumerate(coarse_predictions):
                if coarse_pred == i:
                    img = torch.tensor(images[j]).unsqueeze(0).to("cuda")
                    label = gt_labels[j]
                    output = model(img)
                    _, fine_pred = output.max(1)
                    pred = class_names.index(coarse_to_fine[coarse_class][fine_pred.item()])
                    final_pred.append(pred)
                    final_labels.append(label)
    
        # time.sleep(0)
    
    del images

    final_pred = np.array(final_pred)
    final_labels = np.array(final_labels)

    accuracy = (final_pred == final_labels).mean()
    print(f"Hierarchical Classification Accuracy: {accuracy * 100:.2f}%")


    # ood predict
    data_dir = "./data/ood-test"
    num_files = 19

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

    all_predictions = []  


    model_path = os.path.join(coarse_model_folder, "decompiled_model.pth")
    model = ResNet152_CIFAR(len(coarse_classes), True)
    model.load_state_dict(torch.load(model_path))
    model.to("cuda")

    model.eval()  # Ensure model is in evaluation mode
    for distortion in distortions:
        for severity in range(1, 6):
            predictions = evaluate_ood(model, distortion, severity, coarse_classes, fine_models_folder)
            all_predictions.extend(predictions)  # Accumulate predictions
            print(f"{distortion} (Severity {severity})")


    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv(os.path.join(main_folder, "submission_ood.csv"), index=False)


def evaluate_ood(model, distortion_name, severity, coarse_classes, fine_models_folder):
    data_dir = "./data/ood-test"
    device = "cuda"

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
        batch_size=64,
        shuffle=False, 
        num_workers=4,
        pin_memory=True)

    # Normalize after converting to tensor
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    predictions = []  
    coarse_predictions = []  
    raw_images = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(dataloader, desc="Evaluating on Clean Test Set")):
            inputs = inputs[0]
            inputs = normalize(inputs)
            inputs = inputs.to("cuda")
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            coarse_predictions.extend(predicted.cpu().numpy())
            images.extend(inputs.cpu().numpy())
    
    final_pred = []
    for i, coarse_class in enumerate(coarse_classes):
        num_classes = len(coarse_to_fine[coarse_class])
        model_path = fine_models_folder[coarse_class]
        model = ResNet152_CIFAR(num_classes, True)
        model.load_state_dict(torch.load(model_path))
        model.to("cuda")
        model.eval()

        with torch.no_grad():
            for j, coarse_pred in enumerate(coarse_predictions):
                if coarse_pred == i:
                    img = torch.tensor(raw_images[j]).unsqueeze(0).to("cuda")
                    output = model(img)
                    _, fine_pred = output.max(1)
                    pred = class_names.index(coarse_to_fine[coarse_class][fine_pred.item()])
                    predictions.append(pred)
    
    return final_pred