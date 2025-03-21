import argparse
import logging
import os
import sys
import numpy as np
from qiskit_machine_learning.algorithms import PegasosQSVC

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))

from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test, train_path_norm, train_path_pneu, test_path_norm, test_path_pneu
from quantum_classification.quantum_model import pegasos_svc
from workflow.workflow_manager import WorkflowManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def count_images(directory):
    """Counts number of images in a directory."""
    return len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))])


def view_dataset_info():
    """Displays dataset statistics (normal vs pneumonia image count)."""
    train_normal_count = count_images(train_path_norm)
    train_pneumonia_count = count_images(train_path_pneu)
    test_normal_count = count_images(test_path_norm)
    test_pneumonia_count = count_images(test_path_pneu)

    log.info("Dataset Information:")
    log.info(f"  - Train Normal X-rays: {train_normal_count}")
    log.info(f"  - Train Pneumonia X-rays: {train_pneumonia_count}")
    log.info(f"  - Test Normal X-rays: {test_normal_count}")
    log.info(f"  - Test Pneumonia X-rays: {test_pneumonia_count}")


def show_model_scores():
    """Displays Quantum Model accuracy on Train & Test sets."""
    train_score = pegasos_svc.score(X_train_reduced, y_train)
    test_score = pegasos_svc.score(X_test_reduced, y_test)

    log.info(f"Quantum QSVC on Training Data: {train_score:.2f}")
    log.info(f"Quantum QSVC on Test Data: {test_score:.2f}")


def recommend_treatment():
    """CLI to get treatment recommendations."""
    log.info("Enter patient biomarker data (comma-separated):")
    patient_input = input("> ").strip()
    
    try:
        patient_features = np.array([float(x) for x in patient_input.split(",")]).reshape(1, -1)
        workflow_manager = WorkflowManager()
        recommended_treatment = workflow_manager.infer_treatment(patient_features)
        log.info(f"Recommended Treatment: {recommended_treatment}")
    except Exception as e:
        log.error(f"Error processing patient data: {e}")


def main():
    parser = argparse.ArgumentParser(description="CLI for Quantum Pneumonia Detection")
    parser.add_argument("--dataset-info", action="store_true", help="View dataset statistics (normal/pneumonia count)")
    parser.add_argument("--model-score", action="store_true", help="Display Quantum Model accuracy")
    parser.add_argument("--recommend", action="store_true", help="Get a treatment recommendation")

    args = parser.parse_args()

    if args.dataset_info:
        view_dataset_info()
    elif args.model_score:
        show_model_scores()
    elif args.recommend:
        recommend_treatment()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
