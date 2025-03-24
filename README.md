# DS542 Deep Learning for Data Science -- Spring 2025 Midterm Challenge

## Setup

```bash
docker build -t sp25mt .
docker run --gpus all -t -i -v $PWD:/workspace -p 8888:8888 sp25mt /bin/bash
```

## Download Images Data
This will download the images inside data folder
```bash
python utils/download_images.py 
```

## Training
Modify the settings in config/config.yaml, or create a new config file

train.py: Original classification model
train_hier.py: Hierarchical classification model
train_ovd.py: Open-Vocabulary Detection model
```bash
python train.py --config PATH/TO/CONFIG/FILE
```
This will return the model's folder path inside trained_weights, the model's folder include the model's weights, log, val data confusion matrix

## Evaluation
Evaluate on the test data, and predict on the ood data if add --ood_pred

```bash
python eval.py --runs_folder PATH/TO/MODEL/FOLDER --ood_pred
```