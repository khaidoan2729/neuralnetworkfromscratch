import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images(data, labels, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    label_names = [
        '0', '1', '2', '3', '4',
        '5', '6', '7', '8', '9' ]
    
    # Create directories for each label
    for label_name in label_names:
        os.makedirs(os.path.join(output_dir, label_name), exist_ok=True)
    
    # Save each image
    for i, (img_data, label) in enumerate(zip(data, labels)):
        # Convert the data to uint8 (8-bit unsigned integers)
        img_data = np.array(img_data, dtype=np.uint8)
        
        # Reshape and transpose the image data to match the expected shape (32, 32, 3)
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        
        # Create an Image object
        img = Image.fromarray(img)
        
        # Define the path for the image
        label_name = label_names[label]
        img_path = os.path.join(output_dir, label_name, f"{i}.png")
        
        # Save the image
        img.save(img_path)

# Provide the correct path to your data file
file = "/Users/macos/Downloads/cifar-10-batches-py/data_batch_2"
output_dir = "batch_5"

data_batch_1 = unpickle(file)

# Extract the image data and labels
data = data_batch_1[b'data']
labels = data_batch_1[b'labels']

# Save the images to the specified directory
save_images(data, labels, output_dir)
