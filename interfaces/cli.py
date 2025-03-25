import argparse
import logging
import os
import sys
import numpy as np
from qiskit_machine_learning.algorithms import PegasosQSVC

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(BASE_DIR)

from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test, train_path_norm, train_path_pneu, test_path_norm, test_path_pneu
from quantum_classification.quantum_model import pegasos_svc
from workflow.workflow_manager import WorkflowManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

workflow_manager = WorkflowManager()


def count_images(directory):
    """Counts number of image files in the directory."""
    return len([f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])


def view_dataset_info():
    """Display number of normal vs pneumonia X-ray images in dataset."""
    train_normal = count_images(train_path_norm)
    train_pneumonia = count_images(train_path_pneu)
    test_normal = count_images(test_path_norm)
    test_pneumonia = count_images(test_path_pneu)

    log.info("🩺 Dataset Information:")
    log.info(f"   Train - Normal     : {train_normal}")
    log.info(f"   Train - Pneumonia  : {train_pneumonia}")
    log.info(f"   Test  - Normal     : {test_normal}")
    log.info(f"   Test  - Pneumonia  : {test_pneumonia}")


def show_model_scores():
    """Evaluate model accuracy on train and test sets."""
    try:
        # Ensure model is trained
        _ = pegasos_svc.predict(X_train_reduced[:1])
    except Exception:
        log.info("⚠️ PegasosQSVC model not trained. Training now...")
        from quantum_classification.quantum_model import train_and_save_qsvc
        train_and_save_qsvc()

    train_score = pegasos_svc.score(X_train_reduced, y_train)
    test_score = pegasos_svc.score(X_test_reduced, y_test)
    log.info(f"🎯 Quantum QSVC Train Accuracy: {train_score:.2f}")
    log.info(f"🎯 Quantum QSVC Test Accuracy : {test_score:.2f}")


def predict_sample():
    """Classify a new PCA-reduced X-ray feature vector using quantum circuit."""
    log.info("📌 Enter 18 PCA-reduced features (comma-separated):")
    raw_input = input("> ").strip()

    try:
        features = np.array([float(x) for x in raw_input.split(",")], dtype=np.float32)
        if len(features) != 18:
            raise ValueError("Expected 18 features (matching 18 qubits).")
        
        # Expand to 3 layers → 54 parameters
        full_input = np.tile(features, 3)
        prediction = workflow_manager.classify_with_quantum_circuit(full_input)
        log.info(f"🧠 Quantum Pneumonia Prediction: {prediction}")
    except Exception as e:
        log.error(f"❌ Error during classification: {e}")


def main():
    parser = argparse.ArgumentParser(description="🩺 CLI for Quantum X-ray Pneumonia Detection")
    parser.add_argument("--dataset-info", action="store_true", help="Show X-ray dataset statistics")
    parser.add_argument("--model-score", action="store_true", help="Display quantum model accuracy")
    parser.add_argument("--predict", action="store_true", help="Classify a new X-ray sample using quantum circuit")

    args = parser.parse_args()

    if args.dataset_info:
        view_dataset_info()
    elif args.model_score:
        show_model_scores()
    elif args.predict:
        predict_sample()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
