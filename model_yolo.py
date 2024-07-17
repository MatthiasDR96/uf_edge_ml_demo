# Imports
import os
from PIL import Image
from ultralytics import YOLO
from ultralytics import settings

class Model():

	def __init__(self):

		# Data dir
		self.data_dir = 'C://Users//matth//OneDrive - KU Leuven//Python_Projects//app//datasets//dataset.yaml'

		# Params
		self.num_epochs = 5
		self.image_size = 640

		# Model
		self.model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights
		#self.model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt")  # build from YAML and transfer weights
		#self.model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

		# Update a setting
		settings.update({"mlflow": True})

		# Reset settings to default values
		settings.reset()

		# Start server
		#os.system('mlflow server --backend-store-uri runs/mlflow')

	def model_inference(self, frame):

		# Predict with the model
		results = self.model(frame, stream=True)  # predict on an image

		# Plot results
		for r in results:
			im = r.plot()  # plot a BGR numpy array of predictions

		return im  

	def model_training(self):

		# Train the model
		self.model.train(data=self.data_dir, epochs=self.num_epochs, imgsz=self.image_size, save_dir=self.data_dir)

		# Load a pretrained YOLOv8n model
		self.model = YOLO('./runs/classify/train/weights/best.pt')

		# Validate the model
		metrics = self.model.val()

		return float(metrics*100)