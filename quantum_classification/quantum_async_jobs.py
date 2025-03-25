import os
import logging
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params

log = logging.getLogger(__name__)

# Load IBM Token from env
token = os.getenv("QISKIT_IBM_TOKEN")
if not token:
    raise ValueError("QISKIT_IBM_TOKEN environment variable is not set!")

# Initialize service and backend
service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=token
)
backend = service.least_busy(operational=True, simulator=False)
estimator = Estimator(mode=backend)

# Store job references (for demo; replace with Redis/db in production)
job_store = {}


def submit_quantum_job(features):
    """
    Submits a quantum job to IBMQ Estimator for pneumonia detection.

    Args:
        features (List[float]): Feature vector of length 54 (18 qubits * 3 layers)

    Returns:
        str: Job ID
    """
    num_qubits = 18
    layers = 3
    total_params = calculate_total_params(num_qubits, layers)

    if len(features) != total_params:
        raise ValueError(f"Expected {total_params} features, got {len(features)}")

    params = [Parameter(f"θ{i}") for i in range(total_params)]
    circuit = build_ansatz(num_qubits, params)
    observable = SparsePauliOp("Z" * num_qubits)

    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pass_manager.run(circuit)
    observable = observable.apply_layout(transpiled.layout)

    job = estimator.run([(transpiled, observable, [features])])
    job_id = job.job_id()
    job_store[job_id] = job
    log.info(f"📡 Submitted IBMQ job: {job_id}")
    return job_id


def check_quantum_job(job_id, threshold=0.5):
    """
    Checks job status and returns result (if available).

    Args:
        job_id (str): ID of submitted quantum job
        threshold (float): Decision boundary

    Returns:
        dict: status, prediction, expectation_value (if available)
    """
    job = job_store.get(job_id)
    if job is None:
        return {"status": "error", "message": f"Job ID {job_id} not found"}

    if not job.done():
        return {"status": "pending", "message": "Job is still running..."}

    result = job.result()
    value = float(result[0].data.evs)
    prediction = "Pneumonia Detected" if value > threshold else "No Pneumonia Detected"

    return {
        "status": "complete",
        "expectation_value": round(value, 4),
        "prediction": prediction
    }
