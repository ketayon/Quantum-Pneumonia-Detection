import logging
import os
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

from quantum_classification.quantum_circuit import build_ansatz, calculate_total_params

log = logging.getLogger(__name__)

token = os.getenv("QISKIT_IBM_TOKEN")
if not token:
    raise ValueError("QISKIT_IBM_TOKEN environment variable is not set!")

service = QiskitRuntimeService(
    channel="ibm_quantum",
    instance="ibm-q/open/main",
    token=token
)
backend = service.least_busy(operational=True, simulator=False)
estimator = Estimator(mode=backend)


def evaluate_ansatz_expectation(features):
    """
    Evaluate ⟨Z⊗Z⊗...Z⟩ on real IBM backend using EstimatorV2 (blocking).

    Args:
        features (List[float]): Quantum input features of length num_qubits * layers.

    Returns:
        float: Expectation value.
    """
    num_qubits = 18
    layers = 3
    total_params = calculate_total_params(num_qubits, layers)

    if len(features) != total_params:
        raise ValueError(f"Expected {total_params} features, got {len(features)}")

    # Build parameterized ansatz
    params = [Parameter(f"θ{i}") for i in range(total_params)]
    circuit = build_ansatz(num_qubits, params)

    # Observable: Z⊗Z⊗...Z
    observable = SparsePauliOp("Z" * num_qubits)

    # Transpile to backend's native gates
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled = pass_manager.run(circuit)
    observable = observable.apply_layout(transpiled.layout)

    # Submit and wait for result (blocking)
    log.info("🔁 Submitting Estimator job to IBM Quantum...")
    job = estimator.run([(transpiled, observable, [features])])
    result = job.result()
    value = float(result[0].data.evs)

    log.info(f"✅ IBM Expectation Value: {value:.4f}")
    return value


def predict_with_expectation(features, threshold=0.5):
    """
    Predict label using blocking expectation value evaluation (CLI/debug only).

    Args:
        features (List[float]): Input vector.
        threshold (float): Decision boundary.

    Returns:
        str: "Tumor Detected" or "No Tumor Detected"
    """
    value = evaluate_ansatz_expectation(features)
    return "Pneumonia" if value > threshold else "Normal"