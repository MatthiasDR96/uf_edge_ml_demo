# Imports
import os
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.create import choices

# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = 'http://localhost:8080'

# API key is available at the Account & Settings > Access Tokens page in Label Studio UI
API_KEY = '40fbbd6b3b72951b085078a24d07a4fca97a62cd'

os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'
os.environ['LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT'] = "C:/Users/matth/OneDrive - KU Leuven/Python_Projects/uf_edge_ml_demo/data/raw"

# Connect to the Label Studio API and check the connection
ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# Define labeling interface
label_config = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Fuse" background="green"/>
  </RectangleLabels>
</View>
"""

# Create a project with the specified title and labeling configuration
#project = ls.projects.create(
    #title='Fuse detection test',
    #label_config=label_config
#)

# Get a list of all projects
projects = ls.projects.list()

# Print each project's ID and title
for project in projects:
    print(f"ID: {project.id}, Title: {project.title}")

# Get all images
image_paths = []
for root, dirs, files in os.walk('./data/raw'):
    for file in files:
            image_paths.append({'image': os.path.join(root, file)})

print(image_paths)

ls.projects.import_tasks(
    id=13,
    request=[{'image': './NOK/NOK_04955515e4.jpg'}])