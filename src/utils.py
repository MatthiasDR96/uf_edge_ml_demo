# Imports
import os
import json
import yaml
import shutil
import numpy as np


def json_to_yolo(source_path, dest_path):

    # Load the JSON file
    with open(source_path, 'r') as f:
        data = json.load(f)

    # Create directories if they don't exist
    os.makedirs(dest_path + '/images', exist_ok=True)
    os.makedirs(dest_path + '/labels', exist_ok=True)

    # Initialize classes and notes
    classes = []
    notes = {'categories': []}

    # Process each entry in the JSON file
    for annotation in data:

        # Get image path
        image_path = annotation["data"]["image"].split("/")[-1]
        image_folder = image_path.split('_')[0]
        full_path = './data/raw/' + image_folder + '/' + image_path

        # Copy the image file to the 'images' directory
        shutil.copy(full_path, dest_path + '/images')

        # Create a corresponding annotation file in the 'labels' directory
        with open(f'{dest_path}/labels/{os.path.splitext(os.path.basename(image_path))[0]}.txt', 'w') as f:
            for label in annotation['annotations']:
                for result in label['result']:

                    # Get class label
                    class_label = result['value']['rectanglelabels'][0]
                    class_id = classes.index(class_label) if class_label in classes else len(classes)
                    if class_label not in classes: 
                        classes.append(class_label)
                        notes['categories'].append({"id": class_id, "name": class_label})

                    # Get bounding box
                    x = result['value']["x"]
                    y = result['value']["y"]
                    width = result['value']["width"]
                    height = result['value']["height"]

                    # Convert to YOLO format (in percentages)
                    x_center = (x + width / 2) / 100
                    y_center = (y + height / 2) / 100
                    width = width /  100
                    height = height / 100

                    # Write file
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Save classes.txt
    with open(dest_path + '/classes.txt', 'w') as f:
        for cls in classes:
            f.write(f'{cls}\n')

    # Save notes.json
    with open(dest_path + '/notes.json', 'w') as f:
        json.dump(notes, f)

    # Remove zip file
    os.remove(source_path)


def overwrite_dataset_yaml():

    # Open and read the JSON file
    with open('./data/detection/notes.json', 'r') as json_file:
        data_json = json.load(json_file)

    # Open and read the YAML file
    with open('./data/dataset.yaml', 'r') as file:
        data_yaml = yaml.safe_load(file)

    # Extract the categories
    categories = data_json['categories']

    # Loop through each category
    data_yaml['names'] = {}
    for category in categories:
        # Add the id and name to the output
        data_yaml['names'][str(category['id'])] = category['name']

    # Open and write to the YAML file
    with open('./data/dataset.yaml', 'w') as yaml_file:
        yaml.dump(data_yaml, yaml_file, default_flow_style=False)

def split_data(source_dir, target_dirs, split_ratio=(0.7, 0.2, 0.1)):

    # Remove data folders
    for dir in target_dirs: 
        if os.path.exists(dir): shutil.rmtree(dir)

    # Create target directories
    for target_dir in target_dirs:
        os.makedirs(target_dir, exist_ok=True)

    # Get the class directories
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_dir in class_dirs:
        class_dir_path = os.path.join(source_dir, class_dir)

        # Get the list of image files
        images = [f for f in os.listdir(class_dir_path) if os.path.isfile(os.path.join(class_dir_path, f))]

        # Shuffle the list of images
        np.random.shuffle(images)

        # Split the images according to the provided ratios
        train_idx = int(len(images) * split_ratio[0])
        valid_idx = train_idx + int(len(images) * split_ratio[1])

        train_files = images[:train_idx]
        valid_files = images[train_idx:valid_idx]
        test_files = images[valid_idx:]

        # Function to copy files to target directories
        def copy_files(files, target_dir):
            target_class_dir = os.path.join(target_dir, class_dir)
            os.makedirs(target_class_dir, exist_ok=True)
            for file in files:
                shutil.copy2(os.path.join(class_dir_path, file), target_class_dir)

        # Copy files to target directories
        copy_files(train_files, target_dirs[0])
        copy_files(valid_files, target_dirs[1])
        copy_files(test_files, target_dirs[2])