import torch
from tqdm import tqdm

from .model import ClassificationModel
from .image_encoder import ImageEncoder
from utils import *


class OVDClassificationModel(ClassificationModel):
    def __init__(self, config, text_encoding_path, runs_folder=None):
        super().__init__(config, runs_folder)
        assert self.config.LOSS == "CL", f"LOSS function should be Contrastive Loss"
        self.text_encoding = torch.load(text_encoding_path).to(self.config.DEVICE)
        self.text_encoding = self.text_encoding.float()
        self.image_encoder = ImageEncoder(self.model)

        
    def _model_train(self, train_loader, optimizer, epoch):
        self.model.train()

        running_loss, correct, total = 0.0, 0, 0
        epochs = self.config.EPOCHS
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1:3d}/{epochs:3d} [Train]", leave=True)

        for i, (inputs, labels) in enumerate(progress_bar):
            # move inputs and labels to the target device
            inputs, labels = inputs.to(self.config.DEVICE), labels.to(self.config.DEVICE)

            image_encoding = self.model(inputs)

            loss = self.criterion(image_encoding, self.text_encoding, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            similarity = (100.0 * image_encoding @ self.text_encoding.T).softmax(dim=-1)
            predicted = similarity.argmax(axis=1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc
    

    def _model_validate(self, val_loader, epoch):
        self.model.eval()

        running_loss, correct, total = 0.0, 0, 0
        all_labels = []
        all_predictions = []
        epochs = self.config.EPOCHS

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1:3d}/{epochs:3d} [ Val ]", leave=True)
            for i, (inputs, labels) in enumerate(progress_bar):
                # move inputs and labels to the target device
                inputs, labels = inputs.to(self.config.DEVICE), labels.to(self.config.DEVICE)
                image_encoding = self.model(inputs)
                loss = self.criterion(image_encoding, self.text_encoding, labels)

                running_loss += loss.item()
                similarity = (100.0 * image_encoding @ self.text_encoding.T).softmax(dim=-1)
                predicted = similarity.argmax(axis=1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

        cm = confusion_matrix(all_labels, all_predictions)
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc, cm


    def eval(self, ood_pred=False):
        model_path = os.path.join(self.runs_folder, "best_model.pth")
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.config.DEVICE)

        log_path = os.path.join(self.runs_folder, "log.txt")
        test_loader = build_cifar100_dataloader(self.config, self.config.DATA_PATH, mode='test')
        _, clean_accuracy = evaluate_cifar100_test_ovd(self.model, test_loader, self.text_encoding, self.config.DEVICE)
        print(f"Test Accuracy: {clean_accuracy}")

        with open(log_path, "a") as log_file:
            log_file.write(f"Test Accuracy: {clean_accuracy}\n")
            log_file.flush()

        if ood_pred:
            all_predictions = eval_ood.evaluate_ood_test_ovd(self.model, self.text_encoding, self.config)
            submission_df_ood = eval_ood.create_ood_df(all_predictions)
            submission_df_ood.to_csv(os.path.join(self.runs_folder, "submission_ood.csv"), index=False)

        return