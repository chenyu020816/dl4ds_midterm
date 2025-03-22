from ultralytics import YOLO

model = YOLO("yolo11s-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="../dl4ds_midterm/data/cifar100_data_yolo", epochs=100, imgsz=32, batch=128)
