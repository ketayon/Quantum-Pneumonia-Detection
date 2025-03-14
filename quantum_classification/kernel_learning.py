from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

# Set number of qubits and hyperparameters
num_qubits = 2
tau = 100
C = 1000

# Define feature map using ZZFeatureMap
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)

# Quantum kernel computation
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

# Export ansatz (Feature Map) for use in quantum circuits
ansatz = feature_map  # Define ansatz properly
