# Imports
import os
import shutil
import numpy as np

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