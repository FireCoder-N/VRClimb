from ultralytics import YOLO
# import os

# Load a model
model = YOLO("./yolov8l-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
if __name__ == '__main__':
    results = model.train(data="data.yaml", epochs=50, rect=True, workers=1)

