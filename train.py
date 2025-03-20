import argparse
import datetime
import json
import os
from pandas import DataFrame
import pdb
import shutil
from sklearn.metrics import confusion_matrix
import torch
import torch.optim as optim
from tqdm.auto import tqdm
import wandb
import yaml

from utils.dataloader import build_cifar100_dataloader
from utils.utils import *
from src import *


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def train(config, model, epoch, train_loader, optimizer, criterion):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    epochs = config["EPOCHS"]
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d}/{epochs:3d} [Train]", leave=True)

    for i, (inputs, labels) in enumerate(progress_bar):
        # move inputs and labels to the target device
        inputs, labels = inputs.to(config["DEVICE"]), labels.to(config["DEVICE"])
        preds = model(inputs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(preds, 1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def validate(config, model, epoch, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    cm = None
    epochs = config["EPOCHS"]
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d}/{epochs:3d} [ Val ]", leave=True)
        for i, (inputs, labels) in enumerate(progress_bar):
            # move inputs and labels to the target device
            inputs, labels = inputs.to(config["DEVICE"]), labels.to(config["DEVICE"])
            preds = model(inputs)
            loss = criterion(preds, labels)

            running_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            if epoch+1 == config["EPOCHS"]:
                cm = confusion_matrix(all_labels, all_predictions)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc, cm


def wdnb_config(config):
    wdnb_config = {
        "model": config["MODEL"],
        "batch_size": config["BATCH_SIZE"],
        "learning_rate": config["LR"],
        "epochs": config["EPOCHS"],
        "num_workers": config["NUM_WORKERS"],
        "device": config["DEVICE"],
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": config["SEED"],
    }
    return wdnb_config


def create_runs_folder(runs_name):
    """
    Create a folder inside "trained_weights"
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_folder = os.path.join("trained_weights", f"{runs_name}_{timestamp}")
    os.makedirs(runs_folder, exist_ok=True)
    return runs_folder


def main(config):
    model = load_model(config["MODEL"])
    model.to(config["DEVICE"])
    print("\nModel summary:")
    print(f"{model}\n")

    train_loader, val_loader = build_cifar100_dataloader(config, mode='train')
    criterion = load_criterion(config["LOSS"])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["LR"]),
        eps=1e-8,

    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["LR_STEP_SIZE"],
        gamma=0.5
    )
    early_stopping = EarlyStopping(patience=5)

    # pdb.set_trace()
    # create model training folder to save model's weights, logs, ...
    log_folder = create_runs_folder(config["MODEL"])
    log_path = os.path.join(log_folder, "log.txt")
    shutil.copy(config["config_file"], log_folder)
    print(f"Training Folder: {log_folder}")

    # init wandb
    wandb.init(
        project="-sp25-ds542-challenge",
        config=wdnb_config(config),
        name=log_folder,
    )
    wandb.watch(model)

    best_val_acc = 0.0
    epochs = config["EPOCHS"]
    with open(log_path, "w") as log_file:
        for epoch in range(config["EPOCHS"]):
            train_loss, train_acc = train(
                config, model, epoch, train_loader, optimizer, criterion,
            )
            val_loss, val_acc, cm = validate(
                config, model, epoch, val_loader, criterion
            )
            scheduler.step()
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"]
            })
            log_file.write(
                f"Epoch {epoch+1:3d}/{epochs:3d} [Train]: Loss={train_loss}, Accuracy={train_acc}\n"
                f"Epoch {epoch+1:3d}/{epochs:3d} [ Val ]: Loss={val_loss}, Accuracy={val_acc}\n"
            )
            log_file.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(log_folder, "best_model.pth"))
                wandb.save(os.path.join(log_folder, "best_model.pth"))
            if early_stopping(val_loss):
                break

    wandb.finish()
    cm_df = DataFrame(
        cm,
        index=[f"True_{i}" for i in range(cm.shape[0])],
        columns=[f"Pred_{i}" for i in range(cm.shape[1])]
    )
    cm_path = os.path.join(log_folder, "confusion_matrix.csv")
    cm_df.to_csv(cm_path, index=True)
    print(f"Trained weight have been saved in {log_folder}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        print(args.config)
        config = yaml.safe_load(f)
    config["config_file"] = args.config

    main(config)