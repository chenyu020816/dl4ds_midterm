import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument(
        '--model_type',
        type=str,
        default='classification',
        choices=['classification', 'ovd_classification', 'hierarchical_classification'],
    )
    args = parser.parse_args()

    if args.model_type == 'classification':
        from src import ClassificationModel
        model = ClassificationModel(args.config)
        model.train()
    elif args.model_type == 'ovd_classification':
        pass
    elif args.model_type == 'hierarchical_classification':
        pass
    else:
        ValueError(f"Invalid model type: {args.model_type}")


if __name__ == '__main__':
    main()
