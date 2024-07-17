#!/usr/bin/env python3

# Imports
import os
import cv2
import time
from utils import *
from model_yolo import Model
from flask import Flask, render_template, Response, request

# Start Mlflow server
#os.system('mlflow ui')

# Generate Flask app
app = Flask(__name__)

# Params
screen_size = (1920, 1080)

# Generate Camera object
camera = cv2.VideoCapture(0) #"/dev/video0", apiPreference=cv2.CAP_V4L2)  # use 0 for web camera
time.sleep(2.0)

# Generate model object
model = Model()

# Generate frames
global frame_saved
def generate_frames():

	global frame_saved

	# Loop
	while True:

		# Read frame
		success, frame = camera.read()  # read the camera frame
		if not success:

			# None frame
			frame_saved = None

		else:

			# Take original frame
			frame_saved = frame.copy()

			# Annotate frame
			frame = model.model_inference(frame)

			# Resize image to fit screen
			frame_resized = cv2.resize(frame, screen_size)  
			
			# Show frame in browser
			_, buffer = cv2.imencode('.jpg', frame_resized)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
	with open('./data/classes.txt', 'r') as f: options = [line.strip() for line in f]
	return render_template('index.html', options=options, count='x images', status='Not trained', accuracy='0%')

@app.route('/video')
def video():
	return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():

	# Generate frame
	global frame_saved

	# Get category
	category = request.form.get('category')

	# Check if category exist
	if not os.path.exists('./data/raw/' + category): os.makedirs('./data/raw/' + category)

	# Generate filename
	filename = f"{category}_{os.urandom(5).hex()}.jpg"

	# Save frame
	if frame_saved is not None: cv2.imwrite('./data/raw/' + category + '/' + filename, frame_saved)

	# Count files in this class
	count = len(os.listdir('./data/raw/' + category))

	return {'count': count}

@app.route('/delete', methods=['POST'])
def delete():

	# Remove data folders
	for dir in os.listdir('./data/raw'): 
		if os.path.exists('./data/raw/' + dir): shutil.rmtree('./data/raw/' + dir)

	return ('', 204)

@app.route('/drop-down', methods=['POST'])
def dropdown():

	# Get category
	category = request.form.get('category')

	# Count files in this class
	if os.path.exists('./data/raw/' + category): 
		count = len(os.listdir('./data/raw/' + category))
	else:
		count = 'No'

	return {'count': count}

@app.route('/train', methods=['POST'])
def train():

	# Do not train if no images exist
	if not len(os.listdir('./data/raw')) == 0:

		# Split the data in train, valid, test datasets
		split_data('data/raw', ['data/train', 'data/val', 'data/test'])

		# Train model
		best_acc = model.model_training()

		return {'model_status': 'Finished', 'model_accuracy': str(best_acc) + '%'}
	
	else:
		return {'model_status': 'No data', 'model_accuracy': str(0.0) + '%'}

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8000)

camera.release()
