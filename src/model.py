# Imports
import os 
import cv2
import torch
import mlflow
import numpy as np
from utils import *
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchinfo import summary
from torchmetrics import Accuracy
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

class Model():

	def __init__(self):

		# Data dir
		self.data_dir = './data'

		# Params
		self.batch_size = 8
		self.num_epochs = 5
		self.learning_rate = 1e-3

		# Normalization parameters (comuted from numerous images)
		mean = np.array([0.5, 0.5, 0.5])
		std = np.array([0.25, 0.25, 0.25])

		# Define transforms that will be applied to the downloaded images
		self.data_transforms = {
			'train': transforms.Compose([
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean,std)
			]),
			'val': transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(), 
				transforms.Normalize(mean,std)
			]),
			'test': transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(), 
				transforms.Normalize(mean,std)
			])
		}
		
		# Get classes
		with open('./data/classes.txt', 'r') as f:
			self.class_names = [line.strip() for line in f]   

		# Search for the available training device on the computer
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		# Load model
		self.model = models.resnet18(weights='IMAGENET1K_V1')
		self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
		self.model.load_state_dict(torch.load('./models/best_model_params.pt'))
		self.model = self.model.to(self.device)


	def model_inference(self, frame):

		# Get image
		img = frame[:, :, [2, 1, 0]]
		img = Image.fromarray(img)
		img = self.data_transforms['val'](img)
		img = img.unsqueeze(0)
		img = img.to(self.device)

		# Make prediction
		self.model.eval()
		with torch.no_grad():
			outputs = self.model(img)
			_, preds = torch.max(outputs, 1)

		# Annotate image
		text = self.class_names[preds[0]]
		textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
		textX = int((frame.shape[1] - textsize[0]) / 2)
		textY = 25
		cv2.putText(frame, text, (textX, textY) , cv2.FONT_HERSHEY_SIMPLEX ,  1, (0, 255, 0) , 1, cv2.LINE_AA) 

		return frame    

	def model_training(self):

		# Do not train if no images exist
		if len(os.listdir('./data/raw')) == 0: return 0.0

		# Define source and target directories
		source_dir = 'data/raw'
		target_dirs = ['data/train', 'data/val', 'data/test']

		# Call the function to split the data
		split_data(source_dir, target_dirs)

		# Search for the available training device on the computer
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		# Create a directory to save training checkpoints
		best_model_params_path = os.path.join('./models/', 'best_model_params.pt')

		# Create model
		model = models.resnet18(weights='IMAGENET1K_V1')
		model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
		model = model.to(device)
		torch.save(model.state_dict(), best_model_params_path)

		# Define loss function
		loss_fn = nn.CrossEntropyLoss()

		# Define accuracy metric
		metric_fn = Accuracy(task="multiclass", num_classes=len(self.class_names)).to(device)

		# Observe that all parameters are being optimized
		optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

		# Decay LR by a factor of 0.1 every 7 epochs
		scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

		# Image folder object to load images from folders
		train_datasets = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), self.data_transforms['train'])
		valid_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val'), self.data_transforms['val'])
		test_datasets = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), self.data_transforms['test'])
		datasets_dict = {'train': train_datasets, 'val': valid_dataset, 'test': test_datasets}

		# Image dataloaders to load data in batches
		train_loader = torch.utils.data.DataLoader(datasets_dict['train'], batch_size=self.batch_size, shuffle=True)
		valid_loader = torch.utils.data.DataLoader(datasets_dict['val'], batch_size=self.batch_size, shuffle=True)
		test_loader = torch.utils.data.DataLoader(datasets_dict['test'], batch_size=self.batch_size, shuffle=True)
		dataloaders_dict = {'train': train_loader, 'val': valid_loader, 'test': test_loader}

		# Set class names
		self.class_names = datasets_dict['train'].classes

		# Start Mlflow
		mlflow.set_tracking_uri("http://localhost:5000")
		mlflow.set_experiment("edge_ml_model")
		with mlflow.start_run():

			# Set params
			params = {
				"epochs": self.num_epochs,
				"learning_rate": self.learning_rate,
				"batch_size": self.batch_size,
				"loss_function": loss_fn.__class__.__name__,
				"metric_function": metric_fn.__class__.__name__,
				"optimizer": "SGD",
			}

			# Log training parameters.
			mlflow.log_params(params)

			# Log model summary.
			with open("model_summary.txt", "w", encoding="utf-8") as f:
				f.write(str(summary(model)))
			mlflow.log_artifact("model_summary.txt")
		
			# Training loop
			best_acc = 0.0
			for epoch in range(1, self.num_epochs+1):
				print(f'Epoch {epoch}/{self.num_epochs}')
				print('-' * 10)

				# Each epoch has a training and validation phase
				for phase in ['train', 'val']:
					if phase == 'train':
						model.train()  # Set model to training mode
					else:
						model.eval()   # Set model to evaluate mode

					# Init batch losses
					running_loss = 0.0
					running_corrects = 0.0

					# Iterate over data.
					for batch, (inputs, labels) in enumerate(dataloaders_dict[phase]):
						inputs = inputs.to(device)
						labels = labels.to(device)

						# zero the parameter gradients
						optimizer.zero_grad()

						# forward
						# track history if only in train
						with torch.set_grad_enabled(phase == 'train'):
							outputs = model(inputs)
							loss = loss_fn(outputs, labels)
							accuracy = metric_fn(outputs, labels)

							# backward + optimize only if in training phase
							if phase == 'train':
								loss.backward()
								optimizer.step()

						# Statistics
						running_loss += loss.item()
						running_corrects += accuracy

						# Log to Mlflow
						if phase == 'train':
							step = batch * (epoch + 1)
							mlflow.log_metric("loss", f"{loss.item():2f}", step=step)
							mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)
							print(f"loss: {loss:2f} accuracy: {accuracy:2f} [{batch+1} / {len(dataloaders_dict[phase])}]")

					# Step scheduler
					if phase == 'train':
						scheduler.step()

					# Compute epoch loss and acc
					epoch_loss = running_loss / len(dataloaders_dict[phase])
					epoch_acc = running_corrects / len(dataloaders_dict[phase])

					# Log to Mlflow
					if phase == 'val':
						mlflow.log_metric("eval_loss", f"{epoch_loss:2f}", step=epoch)
						mlflow.log_metric("eval_accuracy", f"{epoch_acc:2f}", step=epoch)
						print(f"Eval metrics: \nAccuracy: {epoch_acc:.2f}, Avg loss: {epoch_loss:2f} \n")

					# deep copy the model
					if phase == 'val' and epoch_acc > best_acc:
						best_acc = epoch_acc
						torch.save(model.state_dict(), best_model_params_path)

				print()

			# Save best model weights
			torch.save(model.state_dict(), best_model_params_path)

			# Load new inference model
			self.model.load_state_dict(torch.load(best_model_params_path))

			# Save the trained model to MLflow.
			mlflow.pytorch.log_model(model, "model")

		return float(best_acc.item()*100)