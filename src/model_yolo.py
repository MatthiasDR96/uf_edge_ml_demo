# Imports
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from ultralytics import settings

class Model():

	def __init__(self):

		# Params
		self.epochs = 20
		self.type = "Detection"

		# Select model
		self.model_change()

		# Update a setting
		#settings.update({"mlflow": False})

		# Reset settings to default values
		#settings.reset()

		# Start server
		#os.system('mlflow server --backend-store-uri runs/mlflow')

	def model_change(self):

		# Select model
		if self.type == "Detection":
			if os.path.isfile('./runs/detect/train/weights/best.pt'):
				self.model = YOLO('./runs/detect/train/weights/best.pt')
			else:
				self.model = YOLO("yolov8n.yaml").load("yolov8n.pt")  
		elif self.type == "Classification":
			if os.path.isfile('./runs/classify/train/weights/best.pt'):
				self.model = YOLO('./runs/classify/train/weights/best.pt')
			else:
				self.model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt")  
		elif self.type == "Segmentation":
			if os.path.isfile('./runs/segment/train/weights/best.pt'):
				self.model = YOLO('./runs/segment/train/weights/best.pt')
			else:
				self.model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt") 

	def model_inference(self, frame):

		# Predict with the model
		results = self.model(frame, stream=True, verbose=False, conf=0.01, iou=0.01)  # predict on an image

		# Plot results for classification
		for result in results:
			im = result.plot()  # plot a BGR numpy array of predictions

		return im  

	def model_training(self):

		# Get model type
		model_type = self.type

		# Remove previous runs
		if model_type == "Detection":
			if os.path.exists('./runs/detect'): shutil.rmtree('./runs/detect')
		elif model_type == "Classification":
			if os.path.exists('./runs/classify'): shutil.rmtree('./runs/classify')
		elif model_type == "Segmentation":
			if os.path.exists('./runs/segment'): shutil.rmtree('./runs/segment')

		# Remove MLflow runs
		if os.path.exists('./mlruns'): shutil.rmtree('./mlruns')

		# Select pretrained model
		if model_type == "Detection":
			model = YOLO("yolov8n.yaml").load("yolov8n.pt")  
		elif model_type == "Classification":
			model = YOLO("yolov8n-cls.yaml").load("yolov8n-cls.pt") 
		elif model_type == "Segmentation":
			model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  

		# Select training data
		if model_type == "Detection":
			data_dir = str(Path('./data/dataset.yaml').resolve())
		elif model_type == "Classification":
			data_dir = './data'
		elif model_type == "Segmentation":
			data_dir = str(Path('./data/dataset.yaml').resolve())
		
		# Finetune model
		model.train(data=data_dir, epochs=self.epochs, imgsz=640, save_dir=data_dir)

		# Export model
		#model.export(format="engine")

		# Load a finetuned YOLOv8n model
		if model_type == "Detection":
			self.model = YOLO('./runs/detect/train/weights/best.pt')
		elif model_type == "Classification":
			self.model = YOLO('./runs/classify/train/weights/best.pt')
		elif model_type == "Segmentation":
			self.model = YOLO('./runs/segment/train/weights/best.pt')

		# Validate the model
		metrics = self.model.val()

		# Output results
		if model_type == "Detection":
			output = metrics.maps[1]
		elif model_type == "Classification":
			output = metrics.top1
		elif model_type == "Segmentation":
			output = metrics.maps[1]

		return float(output*100)
	

if __name__ == "__main__":

	model = Model()
	import threading
	best_acc = threading.Thread(target=model.model_training)
	best_acc.start()