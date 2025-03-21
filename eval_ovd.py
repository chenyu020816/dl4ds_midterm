import argparse
import os
import torch
import yaml

from utils import eval_ood, eval_cifar100
from utils import build_cifar100_dataloader
from utils.utils import *


def main(config, runs_folder, ood_pred):
    model = load_model(config["MODEL"],  config["EMB_DIM"], config["PRETRAIN"])
    model = ImageEncoder(model)
    model_path = os.path.join(runs_folder, "best_model.pth")
    model.load_state_dict(torch.load(model_path))
    model.to(config["DEVICE"])
    text_encoding = torch.load("cifar100_text_embeddings.pt").to(config["DEVICE"])

    log_folder = runs_folder
    log_path = os.path.join(log_folder, "log.txt")
    test_loader = build_cifar100_dataloader(config, mode='test')
    _, clean_accuracy = eval_cifar100.evaluate_cifar100_test_ovd(model, test_loader, text_encoding, config["DEVICE"])
    print(f"Test Accuracy: {clean_accuracy}")

    with open(log_path, "a") as log_file:
        log_file.write(f"Test Accuracy: {clean_accuracy}\n")
        log_file.flush()

    if ood_pred:
        all_predictions = eval_ood.evaluate_ood_test_ovd(model, text_encoding, config)
        submission_df_ood = eval_ood.create_ood_df(all_predictions)
        submission_df_ood.to_csv(os.path.join(runs_folder, "submission_ood.csv"), index=False)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--runs_folder', type=str, default='./trained_weights')
    parser.add_argument('--ood_pred', action='store_true')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        print(args.config)
        config = yaml.safe_load(f)
    main(config, args.runs_folder, args.ood_pred)
