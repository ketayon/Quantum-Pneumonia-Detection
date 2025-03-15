import logging
from qiskit_machine_learning.algorithms import PegasosQSVC
from .kernel_learning import kernel
from image_processing.dimensionality_reduction import X_train_reduced, X_test_reduced
from image_processing.data_loader import y_train, y_test

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Hyperparameters
tau = 100
C = 1000

# Initialize and Train PegasosQSVC
pegasos_svc = PegasosQSVC(quantum_kernel=kernel, C=C, num_steps=tau)
pegasos_svc.fit(X_train_reduced, y_train)

# Evaluate Model
pegasos_score = pegasos_svc.score(X_test_reduced, y_test)
log.info(f"PegasosQSVC classification test score: {pegasos_score}")

# Save Model
pegasos_svc.save('models/PegasosQSVC_Fidelity_quantm_trainer_pneumonia.model')
