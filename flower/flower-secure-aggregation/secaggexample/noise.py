def add_noise_to_gradients(gradients, noise_scale=0.01):
    """Add Gaussian noise to gradients for defense."""
    return [g + torch.randn_like(g) * noise_scale for g in gradients]

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

def add_gaussian_noise(parameters, noise_scale=0.01):
    """Add Gaussian noise to model parameters/gradients."""
    if isinstance(parameters, list):  # Handle NumPy arrays (Flower default)
        return [param + np.random.normal(scale=noise_scale, size=param.shape) for param in parameters]
    else:  # Handle PyTorch tensors if needed
        return parameters + torch.randn_like(parameters) * noise_scale

# Store original fit method
original_fit = FlowerClient.fit

def fit_with_noise(self, parameters, config):
    # 1. Execute original training (get gradients/updates)
    updates, num_samples, metrics = original_fit(self, parameters, config)
    
    # 2. Convert updates to NumPy arrays if needed (Flower uses NDArrays)
    if not isinstance(updates, list):
        updates = parameters_to_ndarrays(updates)
    
    # 3. Add noise to updates
    noisy_updates = add_gaussian_noise(updates, noise_scale=0.05)
    
    # 4. Return noisy updates (Flower will handle SecAgg)
    return noisy_updates, num_samples, metrics

# Monkey-patch the client class
FlowerClient.fit = fit_with_noise
