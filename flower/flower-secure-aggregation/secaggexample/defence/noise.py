
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

def add_gaussian_noise(parameters, noise_scale=0.1):
    """Add Gaussian noise to model parameters/gradients."""
    if isinstance(parameters, list):  # Handle NumPy arrays (Flower default)
        return [param + np.random.normal(scale=noise_scale, size=param.shape) for param in parameters]
    else:  # Handle PyTorch tensors if needed
        return parameters + torch.randn_like(parameters) * noise_scale

# Store original fit method
original_fit = FlowerClient.fit

def fit_with_noise(self, parameters, config):
    updates, num_samples, metrics = original_fit(self, parameters, config)
    
    if not isinstance(updates, list):
        updates = parameters_to_ndarrays(updates)
    
    print(f"\n[Client {self.cid}] Original updates: {updates[0][:3]}...")  # Log first 3 values
    noisy_updates = add_gaussian_noise(updates, noise_scale=0.1)
    print(f"[Client {self.cid}] Noisy updates: {noisy_updates[0][:3]}...\n")
    
    return noisy_updates, num_samples, metrics

# Monkey-patch the client class
FlowerClient.fit = fit_with_noise