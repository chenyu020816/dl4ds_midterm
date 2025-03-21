import argparse
import os

import src

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs_folder', type=str, default='')
    parser.add_argument(
        '--model_type',
        type=str,
        default='classification',
        choices=['classification', 'ovd_classification', 'hierarchical_classification'],
    )
    args = parser.parse_args()

    config_path = os.path.join(args.runs_folder, 'config.yaml')
    if args.model_type == 'classification':
        model = src.ClassificationModel(config_path, runs_folder=args.runs_folder)
        model.eval()
    elif args.model_type == 'ovd_classification':
        model = src.OVDClassificationModel(
            config_path,
            "cifar100_text_embeddings.pt",
            runs_folder=args.runs_folder
        )
        model.eval()
    elif args.model_type == 'hierarchical_classification':
        pass
    else:
        ValueError(f"Invalid model type: {args.model_type}")


if __name__ == '__main__':
    main()
