def add_noise_to_gradients(gradients, noise_scale=0.01):
    """Add Gaussian noise to gradients for defense."""
    return [g + torch.randn_like(g) * noise_scale for g in gradients]