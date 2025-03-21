from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="path/to/dataset", epochs=100, imgsz=32)