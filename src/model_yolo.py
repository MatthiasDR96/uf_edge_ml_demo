# Imports
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

class Model():

	def __init__(self):

		# Params
		self.epochs = 20
		self.batch_size = 1
		self.workers = 2
		self.image_size = 256
		self.type = "Detection"
		self.conf = 0.5
		self.iou = 0.5

		# Select model
		self.model_change()

	def model_change(self):

		# Select model
		if self.type == "Detection":
			if os.path.isfile('./models/detect/train/weights/best.pt'):
				self.model = YOLO('./models/detect/train/weights/best.pt')
			else:
				self.model = YOLO('yolov8n.pt')
		elif self.type == "Classification":
			if os.path.isfile('./models/classify/train/weights/best.pt'):
				self.model = YOLO('./models/classify/train/weights/best.pt')
			else:
				self.model = YOLO('yolov8n-cls.pt') 
		elif self.type == "Segmentation":
			if os.path.isfile('./models/segment/train/weights/best.pt'):
				self.model = YOLO('./models/segment/train/weights/best.pt')
			else:
				self.model = YOLO('yolov8n-seg.pt')

	def model_inference(self, frame):

		# Predict with the model
		results = self.model(frame, stream=True, verbose=False, conf=self.conf, iou=self.iou)  # predict on an image

		# Plot results for classification
		for result in results:
			im = result.plot()  # plot a BGR numpy array of predictions

		return im  

	def model_training(self):

		# Get model type
		model_type = self.type

		# Set params based on model type
		if model_type == "Detection":
			if os.path.exists('./models/detect'): shutil.rmtree('./models/detect')
			model = YOLO('yolov8n.pt')
			data_dir = str(Path('./data/dataset.yaml').resolve())
			save_dir = './models/detect'
			project_dir = './models/detect'
		elif model_type == "Classification":
			if os.path.exists('./models/classify'): shutil.rmtree('./models/classify')
			model =  YOLO('yolov8n-cls.pt')
			data_dir = './data'
			save_dir = './models/classify'
			project_dir = './models/classify'
		elif model_type == "Segmentation":
			if os.path.exists('./models/segment'): shutil.rmtree('./models/segment')
			model =  YOLO('yolov8n-seg.pt')
			data_dir = str(Path('./data/dataset.yaml').resolve())
			save_dir = './models/segment'
			project_dir = './models/segment'

		# Remove MLflow runs
		if os.path.exists('./mlruns'): shutil.rmtree('./mlruns')

		# Finetune model
		model.train(data=data_dir, epochs=self.epochs, batch=self.batch_size, workers=self.workers, imgsz=self.image_size, project=project_dir, cache=False, val=False, save_dir=save_dir)

		# Load a finetuned YOLOv8n model
		if model_type == "Detection":
			self.model = YOLO('./models/detect/train/weights/best.pt')
		elif model_type == "Classification":
			self.model = YOLO('./models/classify/train/weights/best.pt')
		elif model_type == "Segmentation":
			self.model = YOLO('./models/segment/train/weights/best.pt')

		# Validate the model
		#metrics = self.model.val()

		# Output results
		#if model_type == "Detection":
			#output = metrics.maps[1]
		#elif model_type == "Classification":
			#output = metrics.top1
		#elif model_type == "Segmentation":
			#output = metrics.maps[1]
		output = 100

		return float(output)