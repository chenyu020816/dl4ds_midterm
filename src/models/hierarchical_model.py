from pandas import DataFrame

import src
from utils.utils import coarse_to_fine
from .model import ClassificationModel

from utils import *


class HierarchicalClassificationModel(ClassificationModel):
    def __init__(self, config, text_encoding_path, runs_folder=None):
        super().__init__(config, runs_folder)
        self.coarse_classes = coarse_to_fine.keys()
        self.coarse_to_fine = coarse_to_fine
        self.model = self.load_model(len(self.coarse_classes))
        self.fine_models = self.load_fine_models()
        self.cate_criterion = self.load_criterion()

        self._create_fine_runs_folder()

    def load_fine_models(self):
        model_dict = {}
        for coarse_class in self.coarse_classes:
            num_classes = len(self.coarse_to_fine[coarse_class])
            model_dict[coarse_class] = self.load_model(num_classes)

        return model_dict


    def _create_fine_runs_folder(self):
        for coarse_class in self.coarse_classes:
            runs_folder = os.path.join(self.runs_folder, coarse_class)
            os.makedirs(runs_folder, exist_ok=True)
        return


    def train(self):
        self.model.to(self.config.DEVICE)
        train_loader, val_loader = build_cifar100_dataloader(self.config, self.config.COARSE_DATA_PATH, mode='train')
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
        cm_path = os.path.join(self.runs_folder, "coarse_confusion_matrix.csv")
        cm_df.to_csv(cm_path, index=True)
        analysis_cm(cm_path, 30)

        fine_data_loaders = {
            coarse_class: build_cifar100_dataloader(
                self.config, os.path.join(self.config.FINE_DATA_PATH, coarse_class), mode='train'
            ) for coarse_class in self.coarse_classes
        }
        for coarse_class in self.coarse_classes:
            model = self.fine_models[coarse_class].to(self.config.DEVICE)
            early_stopping.counter = 0
            early_stopping.best_loss = float('inf')
            train_loader, val_loader = fine_data_loaders[coarse_class]
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

            best_val_acc = 0.0
            log_path = os.path.join(self.runs_folder, coarse_class, "log.txt")
            with open(log_path, "w") as log_file:
                for epoch in range(self.config.EPOCHS):
                    train_loss, train_acc = self._model_train(
                        model, train_loader, optimizer, epoch, False
                    )
                    scheduler.step()
                    val_loss, val_acc, cm = self._model_validate(
                        model, val_loader, epoch, False
                    )

                    scheduler.step()
                    log_file.write(
                        f"Epoch {epoch + 1:3d}/{self.config.EPOCHS:3d} [Train]: Loss={train_loss}, Accuracy={train_acc}\n"
                        f"Epoch {epoch + 1:3d}/{self.config.EPOCHS:3d} [ Val ]: Loss={val_loss}, Accuracy={val_acc}\n"
                    )
                    log_file.flush()

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(
                            model.state_dict(),
                            os.path.join(self.runs_folder, coarse_class, "best_model.pth")
                        )
                    if early_stopping(val_loss):
                        break

            cm_df = DataFrame(
                cm,
                index=[f"True_{i}" for i in range(cm.shape[0])],
                columns=[f"Pred_{i}" for i in range(cm.shape[1])]
            )
            cm_path = os.path.join(self.runs_folder, coarse_class, "coarse_confusion_matrix.csv")
            cm_df.to_csv(cm_path, index=True)

        print(f"Trained weight have been saved in {self.runs_folder}")

        return


    def eval(self, ood_pred=False):
        model_path = os.path.join(self.runs_folder, "best_model.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.config.DEVICE)
        self._load_fine_models_weights()
        test_loader = build_cifar100_dataloader(self.config, self.config.DATA_PATH, mode='test')

        log_path = os.path.join(self.runs_folder, "log.txt")
        _, clean_accuracy = evaluate_cifar100_test_hierarchical(
            self.model, self.fine_models, test_loader, self.config.DEVICE
        )
        print(f"Test Accuracy: {clean_accuracy}")

        with open(log_path, "a") as log_file:
            log_file.write(f"Test Accuracy: {clean_accuracy}\n")
            log_file.flush()

        if ood_pred:
            all_predictions = eval_ood.evaluate_ood_test(self.model, self.config)
            submission_df_ood = eval_ood.create_ood_df(all_predictions)
            submission_df_ood.to_csv(os.path.join(self.runs_folder, "submission_ood.csv"), index=False)


    def _load_fine_models_weights(self):
        for coarse_class in self.coarse_classes:
            model_path = os.path.join(self.runs_folder, coarse_class, "best_model.pth")
            self.fine_models[coarse_class].load_state_dict(torch.load(model_path))

        return

