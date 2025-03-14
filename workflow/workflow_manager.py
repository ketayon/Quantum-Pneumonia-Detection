import os
import logging
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import PegasosQSVC
from quantum_classification.quantum_model import pegasos_svc
from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test
from workflow.job_scheduler import JobScheduler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Define model path
MODEL_PATH = "models/PegasosQSVC_Fidelity_quantm_trainer_pneumonia.model"

# Load IBM Quantum Service
token = os.getenv("QISKIT_IBM_TOKEN")

if not token:
    raise ValueError("ERROR: QISKIT_IBM_TOKEN environment variable is not set!")

service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=token
)

backend = service.least_busy(operational=True, simulator=False)


class WorkflowManager:
    """Manages the Pneumonia Quantum Classification Workflow"""

    def __init__(self):
        """Initialize Workflow Manager"""
        self.job_scheduler = JobScheduler()
        self.model = None
        log.info("Quantum Pneumonia Workflow Initialized on Backend: %s", backend)
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load the trained model if it exists, otherwise train and save"""
        if os.path.exists(MODEL_PATH):
            log.info("Loading pre-trained Quantum Pneumonia Model...")
            self.model = PegasosQSVC.load(MODEL_PATH)
            log.info("Model loaded successfully!")
        else:
            log.info("No pre-trained model found. Training a new model...")
            self.train_quantum_model()
            log.info("Saving trained model...")
            pegasos_svc.save(MODEL_PATH)
            self.model = pegasos_svc
            log.info("Model saved at: %s", MODEL_PATH)

    def train_quantum_model(self):
        """Train the Quantum Model using Job Scheduler"""
        log.info("Scheduling Quantum Model Training...")
        self.job_scheduler.schedule_task(self._execute_training)

    def _execute_training(self):
        """Handles Quantum Training Execution"""
        log.info("Executing Quantum Pneumonia Model Training...")
        pegasos_svc.fit(X_train_reduced, y_train)
        accuracy = pegasos_svc.score(X_test_reduced, y_test)
        log.info(f"Quantum Pneumonia Model Training Completed. Accuracy: {accuracy:.2f}")

    def classify_xray_images(self, image_data):
        """Classify Chest X-ray Images using the trained model"""
        if self.model is None:
            log.error("No trained model found. Please train the model first.")
            return None
        log.info("Scheduling Pneumonia Classification Task...")
        return self.job_scheduler.schedule_task(self._infer_pneumonia, image_data)

    def _infer_pneumonia(self, image_data):
        """Infer if a chest X-ray image indicates pneumonia"""
        log.info("Performing Pneumonia Image Classification...")
        prediction = self.model.predict(image_data)
        return prediction
