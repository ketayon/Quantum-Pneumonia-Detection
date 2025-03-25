import os
import logging
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import PegasosQSVC

from quantum_classification.quantum_model import pegasos_svc, train_and_save_qsvc
from workflow.job_scheduler import JobScheduler
from quantum_classification.quantum_async_jobs import (
    submit_quantum_job,
    check_quantum_job
)
from quantum_classification.quantum_estimation import predict_with_expectation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_PATH = "models/PegasosQSVC_Fidelity_quantm_trainer_pneumonia.model"

# Ensure IBM token is set
token = os.getenv("QISKIT_IBM_TOKEN")
if not token:
    raise ValueError("ERROR: QISKIT_IBM_TOKEN environment variable is not set!")

# Initialize IBMQ runtime
service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=token
)
backend = service.least_busy(operational=True, simulator=False)


class WorkflowManager:
    """Manages the Pneumonia X-ray Quantum Classification Workflow"""

    def __init__(self):
        self.job_scheduler = JobScheduler()
        self.model = None
        log.info("🚀 Quantum Pneumonia Workflow Initialized on Backend: %s", backend)
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Load the trained QSVC model if it exists, otherwise train and save"""
        if os.path.exists(MODEL_PATH):
            log.info("📥 Loading pre-trained Quantum Pneumonia Model...")
            self.model = PegasosQSVC.load(MODEL_PATH)
            log.info("✅ Model loaded successfully.")
        else:
            log.info("🧠 No trained model found. Starting training...")
            self.train_quantum_model()
            log.info("💾 Saving trained model...")
            pegasos_svc.save(MODEL_PATH)
            self.model = pegasos_svc
            log.info("📌 Model saved at: %s", MODEL_PATH)

    def train_quantum_model(self):
        """Schedule quantum model training"""
        log.info("📅 Scheduling Quantum Model Training...")
        self.job_scheduler.schedule_task(self._execute_training)

    def _execute_training(self):
        """Train model in a separate thread"""
        log.info("🔁 Executing Pneumonia Model Training...")
        accuracy = train_and_save_qsvc()
        self.model = pegasos_svc
        log.info(f"✅ Pneumonia QSVC Training Completed. Accuracy: {accuracy:.2f}")

    def classify_xray_image(self, image_data):
        """Classify processed X-ray image using trained QSVC"""
        if self.model is None:
            log.error("❌ No trained model available. Please train first.")
            return None
        log.info("⚙️ Scheduling QSVC classification...")
        return self.job_scheduler.schedule_task(self._infer_pneumonia, image_data)

    def _infer_pneumonia(self, image_data):
        """Run classification on reduced input features"""
        log.info("🔎 Performing QSVC prediction on input features...")
        prediction = self.model.predict(image_data)
        return prediction

    @staticmethod
    def classify_with_quantum_circuit(image_features):
        """
        Classify using expectation value from real IBM Quantum backend.
        Meant for CLI/testing — not for web frontend (blocking call).
        """
        log.info("🧠 Running blocking classification via Estimator...")
        prediction = predict_with_expectation(image_features)
        log.info(f"🧬 IBM Quantum Prediction: {prediction}")
        return prediction

    @staticmethod
    def submit_quantum_job_async(image_features):
        """
        Submit async job to IBMQ for backend execution.
        Returns:
            str: Job ID
        """
        log.info("📡 Submitting async job to IBM Quantum backend...")
        return submit_quantum_job(image_features)

    @staticmethod
    def check_quantum_job_result(job_id):
        """
        Poll for job status/result from IBMQ backend.
        Returns:
            dict: {status, prediction, expectation_value}
        """
        log.info(f"🔍 Checking IBM Quantum Job: {job_id}")
        return check_quantum_job(job_id)
