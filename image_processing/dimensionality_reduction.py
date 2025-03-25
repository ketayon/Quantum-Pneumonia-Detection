import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from image_processing.data_loader import X_train, X_test

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def reduce_to_n_dimensions(data, n_dimensions):
    n_features = data.shape[1]
    if n_dimensions > n_features:
        raise ValueError("Target dimensions > input features.")

    split_size = n_features // n_dimensions
    reduced = np.zeros((data.shape[0], n_dimensions))

    for i in range(n_dimensions):
        start, end = i * split_size, (i + 1) * split_size if i < n_dimensions - 1 else n_features
        reduced[:, i] = np.mean(data[:, start:end], axis=1)
    return reduced


def reduce_to_2_dimensions(data):
    """
    Reduces the number of features in the dataset to 2 dimensions by averaging subsets of features.
    
    Args:
        data (np.ndarray): Input dataset of shape (n_samples, n_features).
    
    Returns:
        np.ndarray: Reduced dataset of shape (n_samples, 2).
    """
    n_features = data.shape[1]
    split_size = n_features // 2
    reduced_data = np.column_stack([
        np.mean(data[:, :split_size], axis=1),
        np.mean(data[:, split_size:], axis=1)
    ])
    return reduced_data


X_train_red = reduce_to_2_dimensions(X_train)
X_test_red = reduce_to_2_dimensions(X_test)

X_train_reduced = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_train_red)
X_test_reduced = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X_test_red)

log.info("Image processing pipeline completed. Data saved in datasets folder.")
