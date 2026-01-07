import os
import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# Load EMNIST ByClass
mat = loadmat(r"C:\Users\msolorzano52.HACKTHEGIBSON\Downloads\matlab\matlab\emnist-byclass.mat")

# Training data
X_train = mat['dataset']['train'][0,0]['images'][0,0]      # flattened images
y_train = mat['dataset']['train'][0,0]['labels'][0,0].flatten()

# Test data
X_test = mat['dataset']['test'][0,0]['images'][0,0]
y_test = mat['dataset']['test'][0,0]['labels'][0,0].flatten()

# Reshape images to 28x28
X_train = X_train.reshape((-1,28,28)).astype(np.uint8)
X_test = X_test.reshape((-1,28,28)).astype(np.uint8)

# Function to save images
def save_emnist_images(X, y, output_dir):
    for i in range(len(X)):
        label = str(y[i])
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        img = cv2.rotate(X[i], cv2.ROTATE_90_CLOCKWISE)      # EMNIST images are rotated
        img = cv2.flip(img, 1)                               # flip to correct orientation
        img_resized = cv2.resize(img, (64,64))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(label_dir, f"{i}.png"), img_rgb)

# Save to your data folder
save_emnist_images(X_train, y_train, "train/text")
save_emnist_images(X_test, y_test, "test/text")
