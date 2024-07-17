from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt

model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data="./data", epochs=100, imgsz=640)
