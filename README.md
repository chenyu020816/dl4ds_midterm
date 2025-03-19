# DS542 Deep Learning for Data Science -- Spring 2025 Midterm Challenge

## Setup

```bash
docker build -t sp25mt .
docker run --gpus all -t -i -v $PWD:/workspace -p 8888:8888 sp25mt /bin/bash
```

## Download Data

```bash
python utils/download_images.py
```
