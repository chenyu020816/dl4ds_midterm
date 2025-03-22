import argparse

import src

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument(
        '--model_type',
        type=str,
        default='ovd_classification',
        choices=[
            'classification',
            'ovd_classification',
            'hierarchical_classification',
            'hierarchical_ovd_classification'
        ],
    )
    args = parser.parse_args()

    if args.model_type == 'classification':
        model = src.ClassificationModel(args.config)
        model.train()
    elif args.model_type == 'ovd_classification':
        model = src.OVDClassificationModel(args.config, "cifar100_text_embeddings.pt")
        model.train()
    elif args.model_type == 'hierarchical_classification':
        model = src.HierarchicalClassificationModel(args.config)
        model.train()
    elif args.model_type == 'hierarchical_ovd_classification':
        pass
    else:
        ValueError(f"Invalid model type: {args.model_type}")


if __name__ == '__main__':
    main()
