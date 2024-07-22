#!/usr/bin/env python3

# Imports
import os
import cv2
import time
import random
import zipfile
import webbrowser
from utils import *
from model_yolo import Model
from label_studio_sdk import Client
from flask import Flask, render_template, Response, request

# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = '40fbbd6b3b72951b085078a24d07a4fca97a62cd' #(ww: Kuleuven8200)
PROJECT_TITLE = 'uf_edge_ml_model' # Title of the Label studio project that gets created when pressing 'Label'

# Generate Flask app
app = Flask(__name__, static_folder='../static', template_folder='../templates')

# Generate Camera object
camera = cv2.VideoCapture(0) #"/dev/video0", apiPreference=cv2.CAP_V4L2)  # use 0 for web camera
time.sleep(2.0)

# Generate frames
global frame_saved
global is_training
is_training = False
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
			try:
				# Predict
				frame = model.model_inference(frame)
			except:
				continue

			# Resize image to fit screen
			frame_resized = cv2.resize(frame, screen_size)  
			
			# Show frame in browser
			_, buffer = cv2.imencode('.jpg', frame_resized)
			frame = buffer.tobytes()
			yield (b'--frame\r\n'
				   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
	options = os.listdir('./data/raw')
	return render_template('index.html', options=options, count='x images', status='Not trained', accuracy='0.0%')

@app.route('/video')
def video():
	return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():

	# Generate frame
	global frame_saved

	# Get category
	category = request.form.get('category')
	category = category.strip('\n')

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

@app.route('/drop-down-category', methods=['POST'])
def dropdown_category():

	# Get category
	category = request.form.get('category')
	category = category.replace('\r', '').replace('\n', '')

	# Count files in this class
	if os.path.exists('./data/raw/' + category): 
		count = len(os.listdir('./data/raw/' + category))
	else:
		count = 'No'

	return {'count': count}

@app.route('/drop-down-model', methods=['POST'])
def dropdown_model():

	# Get model type
	model_type = request.form.get('model')

	# Change model
	model.type = model_type
	model.model_change()

	return ('', 204)

@app.route('/train', methods=['POST'])
def train():

	# Init
	global is_training
	result = 0.0

	# Check if already training
	if not is_training:

		# Do not train if no images exist
		if not len(os.listdir('./data/raw')) == 0:

			# Set training
			is_training = True

			# Split the data in train, valid, test datasets
			split_data('data/raw', ['data/train', 'data/val', 'data/test'])

			# Modify dataset.yaml
			overwrite_dataset_yaml()

			# Train model
			try:
				result = model.model_training()
			except:
				is_training = False
				return {'model_status': 'Error', 'model_accuracy': str(result) + '%'}

			# Set training
			is_training = False

			return {'model_status': 'Finished', 'model_accuracy': str(result) + '%'}
		
		else:
			
			return {'model_status': 'No data', 'model_accuracy': str(result) + '%'}
		
	else:
			
		return {'model_status': 'Training...', 'model_accuracy': str(result) + '%'}
	
@app.route('/classes', methods=['POST'])
def get_textarea():

	# Get text
	text = request.form.get('classes_content')
	text = text.strip('\r')

	# Get classes
	options = text.split("\n")

	return render_template('index.html', options=options, count='x images', status='Not trained', accuracy='0.0%')

@app.route('/label', methods=['POST'])
def label():

	# Random color generator
	def generate_random_color():
		return '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])

	# Connect to the Label Studio API
	ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
	ls.check_connection()

	# Create label config based on classes in data folder
	labels = os.listdir('./data/raw/')
	label_str = ''.join([f'<Label value="{label}" background="{generate_random_color()}"/>' for label in labels])
	label_config = f'''<View>
							<Image name="image" value="$image"/>
							<RectangleLabels name="label" toName="image">
								{label_str}
							</RectangleLabels>
						</View>'''

	# Get the project with the specified title
	projects = ls.get_projects()
	project_exists = any(project.title == PROJECT_TITLE for project in projects)

	# Get ID of the project
	if project_exists:
		project_id = next(project.id for project in projects if project.title == PROJECT_TITLE)
		ls.delete_project(project_id)

	# Create new project
	ls.create_project(
		title=PROJECT_TITLE,
		label_config=label_config
	)

	# Open Label Studio
	webbrowser.open('http://localhost:8080')

	return ('', 204)

@app.route('/download_label', methods=['POST'])
def download_label():

	# Connect to the Label Studio API
	ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
	ls.check_connection()

	# Check if the project exists
	projects = ls.get_projects()
	project_exists = any(project.title == PROJECT_TITLE for project in projects)

	# If project exist
	if project_exists:

		# Get ID
		project_id = next(project.id for project in projects if project.title == PROJECT_TITLE)

		# Get the project
		project = ls.get_project(project_id)

		# Create an export snapshot
		export_result = project.export_snapshot_create(
			title='export-test-01',
			task_filter_options={
				'view': 1,
				'finished': 'only',  # include all finished tasks (is_labeled = true)
				'annotated': 'only',  # include all tasks with at least one not skipped annotation
			}
		)
		export_id = export_result['id']

		# Wait until the snapshot is ready
		while project.export_snapshot_status(export_id).is_in_progress():
			time.sleep(1.0)

		# Download the snapshot
		status, zip_file_path = project.export_snapshot_download(
			export_id=export_id,
			export_type='YOLO',
			path='.',
		)

		# Open the zip file
		with zipfile.ZipFile('./' + zip_file_path, 'r') as zip_ref:
			# Extract all the contents of the zip file in current directory
			zip_ref.extractall(path='./data/detection2')

		# Remove zip file
		os.remove('./' + zip_file_path)

	return ('', 204)


if __name__ == "__main__":

	# Params
	screen_size = (1920, 1080)

	# Generate model object
	model = Model()

	# Run app
	app.run(host='0.0.0.0', port=8000)

# Close camera
camera.release()
