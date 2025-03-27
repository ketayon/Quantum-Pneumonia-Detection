import os
import numpy as np
import logging
from PIL import Image


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

train_path_norm = 'datasets/chest_xray/train/NORMAL'
train_path_pneu = 'datasets/chest_xray/train/PNEUMONIA'
test_path_norm = 'datasets/chest_xray/test/NORMAL'
test_path_pneu = 'datasets/chest_xray/test/PNEUMONIA'

# Remove potential .DS_Store files
for folder in [train_path_norm, train_path_pneu, test_path_norm, test_path_pneu]:
    ds_store_path = os.path.join(folder, ".DS_Store")
    if os.path.exists(ds_store_path):
        os.remove(ds_store_path)
        log.info(f"Removed .DS_Store from {folder}")


img_size = (256, 256)


def load_images_from_folder(folder, label, target_size=img_size):
    """
    Loads images from a given folder and resizes them to the specified target size.
    Converts images to RGB and flattens them into numpy arrays.

    Args:
        folder (str): Path to the folder containing images.
        label (int): Label associated with the images in the folder.
        target_size (tuple): Target size to which images will be resized (width, height).

    Returns:
        tuple: A list of image data and their corresponding labels.
    """
    data, labels = [], []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = Image.open(img_path).convert("RGB").resize(target_size)
            data.append(np.array(img).flatten())
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
    return data, labels


def load_and_limit_data(path, label, num_samples, target_size=(256, 256)):
    """
    Loads and limits the number of images from a folder to a specified number of samples.
    Ensures images are resized to the specified target size.

    Args:
        path (str): Path to the folder containing images.
        label (int): Label associated with the images in the folder.
        num_samples (int): Maximum number of samples to load.
        target_size (tuple): Target size to which images will be resized (width, height).

    Returns:
        tuple: A list of limited image data and their corresponding labels.
    """
    data, labels = load_images_from_folder(path, label, target_size)
    indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
    data = [data[i] for i in indices]
    labels = [labels[i] for i in indices]
    return data, labels

# Load Data
train_data_norm, train_labels_norm = load_and_limit_data(train_path_norm, 0, 1341, target_size=img_size)
train_data_pneu, train_labels_pneu = load_and_limit_data(train_path_pneu, 1, 3875, target_size=img_size)

test_data_norm, test_labels_norm = load_and_limit_data(test_path_norm, 0, 234, target_size=img_size)
test_data_pneu, test_labels_pneu = load_and_limit_data(test_path_pneu, 1, 390, target_size=img_size)

log.info(f"Total train images loaded: 5216")
log.info(f"Total train images loaded: 624")

X_train = np.array(train_data_norm + train_data_pneu)
y_train = np.array(train_labels_norm + train_labels_pneu)

X_test = np.asarray(test_data_norm + test_data_pneu)
y_test = np.asarray(test_labels_norm + test_labels_pneu)

log.info(f"Dataset loaded successfully: {len(X_train)} train samples, {len(X_test)} test samples.")


def count_images(directory):
    return len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.jpeg'))])
