# DS542 Deep Learning for Data Science -- Spring 2025 Midterm Challenge

## Setup

```bash
# build docker image
docker build -t sp25mt .
# start docker container with cuda
docker run --gpus all -t -i -v $PWD:/workspace -p 8888:8888 sp25mt /bin/bash
```

## Traing Data

#### Download training data
This will download the training images inside data folder "cifar100_data/train"
```bash
python utils/download_images.py 
```
#### Synthetic Data Generation
```bash
python utils/data_augmentation.py --image_folder PATH/TO/TRAINING_DATA/ --aug_folder PATH/TO/SAVE/SYNTHETIC_DATA --aug_size FactorOfSyntheticData
```

## Training
Modify the settings in config/config.yaml, or create a new config file

#### Available models
See ./src/models/__init__.py

#### Start training
```bash
python train.py --config PATH/TO/CONFIG/FILE
```
This will return the model's folder path inside trained_weights, the model's folder include the model's weights, log, confusion matrix.

#### Model Type Argument
```bash
python train.py --config PATH/TO/CONFIG/FILE --model_type classification/ovd_classification/hierarchical_classification
```
- classification: original classification
- ovd_classification: train the model using Contrastive Learning stragey. (Loss function should be set to "CL")
- hierarchical_classification: hierarchical learning **(still debugging)**


## Evaluation
Evaluate on the test data, and add --ood_pred to predict on out-of-distribution data.

```bash
python eval.py --runs_folder PATH/TO/MODEL/FOLDER --ood_pred --model_type classification/ovd_classification/hierarchical_classification
```