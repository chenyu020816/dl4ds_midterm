from datetime import datetime
import importlib
import os
import shutil

import wandb
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from torch import optim
import yaml
from tqdm import tqdm
from wandb.integration.torch.wandb_torch import torch

from utils.dataloader import build_transform
from utils.utils import dict2obj
from utils import *
import src


class ClassificationModel:
    def __init__(self, config_path, runs_folder=None):
        self.config_path = config_path
        self.config = self.load_config()

        self.runs_folder = self._create_runs_folder() if runs_folder is None else runs_folder
        self.log_path = os.path.join(self.runs_folder, "log.txt")
        self.wdnb_config = self._create_wdnb_config()

        self.model = self.load_model(self.config.MODEL, self.config.NUM_CLASSES)
        self.criterion = self.load_criterion()


    def load_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        config = dict2obj(config)
        return config


    def load_model(self, model_name, num_classes):
        try:
            model_class = getattr(importlib.import_module("src"), model_name)
            if self.config.MODEL.startswith("Conv"):
                return model_class(num_classes, self.config.PRETRAIN, self.config.STOCH_DEPTH_PROB)
            else:
                return model_class(num_classes, self.config.PRETRAIN)
        except AttributeError:
            raise ValueError(f"'{self.config.MODEL}' not defined.")


    def load_criterion(self):
        try:
            model_class = getattr(importlib.import_module("src"), self.config.LOSS)
            return model_class
        except AttributeError:
            raise ValueError(f"'{self.config.LOSS}' not defined.")


    def _create_runs_folder(self):
        """
        Create a folder inside "trained_weights"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_folder = os.path.join("trained_weights", f"{self.config.MODEL}_{timestamp}")
        os.makedirs(runs_folder, exist_ok=True)
        shutil.copy(self.config_path, runs_folder)
        return runs_folder


    def _create_wdnb_config(self):
        wdnb_config = {
            "model": self.config.MODEL,
            "batch_size": self.config.BATCH_SIZE,
            "learning_rate": self.config.LR,
            "epochs": self.config.EPOCHS,
            "num_workers": self.config.NUM_WORKERS,
            "device": self.config.DEVICE,
            "data_dir": "./data",
            "ood_dir": "./data/ood-test",
            "wandb_project": "sp25-ds542-challenge",
            "seed": self.config.SEED,
        }
        return wdnb_config


    @staticmethod
    def initialize_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def _model_train(self, model, train_loader, optimizer, epoch, progress_bar=True):
        model.train()

        running_loss, correct, total = 0.0, 0, 0
        epochs = self.config.EPOCHS
        labels_a, labels_b, lam = None, None, None
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1:3d}/{epochs:3d} [Train]", leave=progress_bar)

        for i, (inputs, labels) in enumerate(progress_bar):
            # move inputs and labels to the target device
            inputs, labels = inputs.to(self.config.DEVICE), labels.to(self.config.DEVICE)
            if self.config.TRANSFORM.MIXUP:
                inputs, labels_a, labels_b, lam = src.mixup_data(inputs, labels)

            preds = model(inputs)

            if self.config.TRANSFORM.MIXUP:
                loss = src.mixup_criterion(self.criterion, preds, labels_a, labels_b, lam)
            else:
                loss = self.criterion(preds, labels)

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


    def _model_validate(self, model, val_loader, epoch, progress_bar=True):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        epochs = self.config.EPOCHS
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1:3d}/{epochs:3d} [ Val ]", leave=progress_bar)
            for i, (inputs, labels) in enumerate(progress_bar):
                # move inputs and labels to the target device
                inputs, labels = inputs.to(self.config.DEVICE), labels.to(self.config.DEVICE)
                preds = model(inputs)
                loss = self.criterion(preds, labels)

                running_loss += loss.item()
                _, predicted = torch.max(preds, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

        cm = confusion_matrix(all_labels, all_predictions)
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc, cm


    def train(self):
        # if not pretained model, initialize model's weights
        if not self.config.PRETRAIN:
            self.model.apply(self.initialize_weights)
        self.model.to(self.config.DEVICE)
        # print("\nModel summary:")
        # print(f"{self.model}\n")
        train_transform = build_transform(self.config.TRANSFORM)
        train_loader, val_loader = build_cifar100_dataloader(
            self.config,
            self.config.DATA_PATH,
            mode='train',
            transform=train_transform,
        )
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.LR),
            eps=1e-8,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.EPOCHS,
            eta_min=1e-6
        )
        early_stopping = src.EarlyStopping(patience=3)

        wandb.init(
            project="-sp25-ds542-challenge",
            config=self.wdnb_config,
            name=self.runs_folder,
        )
        wandb.watch(self.model)

        best_val_acc = 0.0

        with open(self.log_path, "w") as log_file:
            for epoch in range(self.config.EPOCHS):
                train_loss, train_acc = self._model_train(
                    self.model, train_loader, optimizer, epoch
                )
                val_loss, val_acc, cm = self._model_validate(
                    self.model, val_loader, epoch
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
                    f"Epoch {epoch + 1:3d}/{self.config.EPOCHS:3d} [Train]: Loss={train_loss}, Accuracy={train_acc}\n"
                    f"Epoch {epoch + 1:3d}/{self.config.EPOCHS:3d} [ Val ]: Loss={val_loss}, Accuracy={val_acc}\n"
                )
                log_file.flush()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), os.path.join(self.runs_folder, "best_model.pth"))
                    wandb.save(os.path.join(self.runs_folder, "best_model.pth"))
                if early_stopping(val_loss):
                    break

        wandb.finish()
        cm_df = DataFrame(
            cm,
            index=[f"True_{i}" for i in range(cm.shape[0])],
            columns=[f"Pred_{i}" for i in range(cm.shape[1])]
        )
        cm_path = os.path.join(self.runs_folder, "confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=True)
        analysis_cm(cm_path, 30)
        print(f"Trained weight have been saved in {self.runs_folder}")

        return


    def eval(self, ood_pred=False):
        model_path = os.path.join(self.runs_folder, "best_model.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.config.DEVICE)

        log_path = os.path.join(self.runs_folder, "log.txt")
        test_loader = build_cifar100_dataloader(self.config, self.config.DATA_PATH, mode='test')
        _, clean_accuracy = evaluate_cifar100_test(self.model, test_loader, self.config.DEVICE)
        print(f"Test Accuracy: {clean_accuracy}")

        with open(log_path, "a") as log_file:
            log_file.write(f"Test Accuracy: {clean_accuracy}\n")
            log_file.flush()

        if ood_pred:
            all_predictions = eval_ood.evaluate_ood_test(self.model, self.config)
            submission_df_ood = eval_ood.create_ood_df(all_predictions)
            submission_df_ood.to_csv(os.path.join(self.runs_folder, "submission_ood.csv"), index=False)

        return
