import os
import random
import shutil

"""
Split data to train and validation folders.
"""

# Define the paths for the train and validation data folders
train_folder = './train_data'
val_folder = './val_data'

# Create validation folder if it doesn't exist
if not os.path.exists(val_folder):
    os.makedirs(val_folder)

# Get all image files in the train folder
image_files = [f for f in os.listdir(train_folder) if f.endswith(".png")]

# Randomly select 10 image files
selected_images = random.sample(image_files, 10)

for image_file in selected_images:
    # Get the corresponding text file (same name as image, but with .txt extension)
    base_name = os.path.splitext(image_file)[0]  # Remove the file extension
    text_file = base_name + '.txt'
    
    # Source paths
    src_image_path = os.path.join(train_folder, image_file)
    src_text_path = os.path.join(train_folder, text_file)
    
    # Destination paths
    dest_image_path = os.path.join(val_folder, image_file)
    dest_text_path = os.path.join(val_folder, text_file)
    
    # Move the image file and the corresponding text file
    shutil.move(src_image_path, dest_image_path)
    shutil.move(src_text_path, dest_text_path)
