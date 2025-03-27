import os
import numpy as np
import pytest
import cv2
from PIL import Image
from image_processing.data_loader import load_images_from_folder, load_and_limit_data
from image_processing.image_transformations import apply_grayscale, apply_gaussian_blur, apply_histogram_equalization
from image_processing.dimensionality_reduction import reduce_to_n_dimensions

# Define dataset paths for Pneumonia classification
path = "./tests/mock_dataset"
train_path_norm = f"{path}/chest_xray/train/NORMAL"
train_path_pneu = f"{path}/chest_xray/train/PNEUMONIA"
test_path_norm = f"{path}/chest_xray/test/NORMAL"
test_path_pneu = f"{path}/chest_xray/test/PNEUMONIA"

# Ensure test dataset directories exist
os.makedirs(train_path_norm, exist_ok=True)
os.makedirs(train_path_pneu, exist_ok=True)
os.makedirs(test_path_norm, exist_ok=True)
os.makedirs(test_path_pneu, exist_ok=True)

# Create mock X-ray images
def create_mock_image(file_path):
    """Creates a dummy grayscale X-ray image for testing"""
    img = Image.new("RGB", (256, 256), color="gray")
    img.save(file_path)

# Create test images
for i in range(5):
    create_mock_image(os.path.join(train_path_norm, f"normal_train_{i}.jpg"))
    create_mock_image(os.path.join(train_path_pneu, f"pneumonia_train_{i}.jpg"))
    create_mock_image(os.path.join(test_path_norm, f"normal_test_{i}.jpg"))
    create_mock_image(os.path.join(test_path_pneu, f"pneumonia_test_{i}.jpg"))

# Test Data Loader
@pytest.mark.parametrize("folder, label", [
    (train_path_pneu, 1), (train_path_norm, 0), 
    (test_path_pneu, 1), (test_path_norm, 0)
])
def test_load_images_from_folder(folder, label):
    """Tests loading chest X-ray images from pneumonia & normal folders"""
    data, labels = load_images_from_folder(folder, label)
    assert len(data) == 5
    assert len(labels) == 5
    assert all(lbl == label for lbl in labels)
    assert isinstance(data[0], np.ndarray)  # Ensure numpy array

# Test Data Limiting
@pytest.mark.parametrize("folder, label, num_samples", [
    (train_path_pneu, 1, 3), (train_path_norm, 0, 2),
    (test_path_pneu, 1, 2), (test_path_norm, 0, 1)
])
def test_load_and_limit_data(folder, label, num_samples):
    """Tests loading and limiting images to a specific number"""
    data, labels = load_and_limit_data(folder, label, num_samples)
    assert len(data) == num_samples
    assert len(labels) == num_samples
    assert all(lbl == label for lbl in labels)
    assert isinstance(data[0], np.ndarray)  # Ensure numpy array

# Test Image Transformations
def test_apply_grayscale():
    """Tests grayscale conversion on an X-ray image"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    gray_img = apply_grayscale(img)
    assert gray_img.shape == (256, 256)  # Single channel
    assert gray_img.dtype == np.uint8  # Check data type

def test_apply_gaussian_blur():
    """Tests Gaussian blur application"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    blurred_img = apply_gaussian_blur(img)
    assert blurred_img.shape == img.shape  # Shape should remain the same
    assert blurred_img.dtype == np.uint8  # Check data type

def test_apply_histogram_equalization():
    """Tests histogram equalization on grayscale X-ray image"""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    enhanced_img = apply_histogram_equalization(img)
    assert enhanced_img.shape == img.shape  # Shape should remain same
    assert enhanced_img.dtype == np.uint8  # Check data type

# Test Dimensionality Reduction
def test_reduce_to_n_dimensions():
    """Tests dimensionality reduction from original features to 8 components"""
    mock_data = np.random.rand(10, 64)  # 10 samples, 64 features
    reduced_data = reduce_to_n_dimensions(mock_data, 8)  # Reduce to 8 dimensions
    assert reduced_data.shape == (10, 8)


if __name__ == "__main__":
    pytest.main()
