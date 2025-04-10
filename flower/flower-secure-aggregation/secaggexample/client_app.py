"""secaggexample: A Flower with SecAgg\+ app."""

import time

import torch
import numpy as np
from unittest.mock import Mock
from secaggexample.task import Net, get_weights, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context


def add_gaussian_noise(params, stddev=0.01):
    return [p + np.random.normal(0, stddev, p.shape).astype(p.dtype) for p in params]


def clip_gradients(params, clip_value=0.1):
    return [np.clip(p, -clip_value, clip_value) for p in params]


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self, trainloader, valloader, local_epochs, learning_rate, timeout, is_demo
    ):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # For demonstration purposes only
        self.timeout = timeout
        self.is_demo = is_demo

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = {}
        if not self.is_demo:
            results = train(
                self.net,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.lr,
                self.device,
            )
        ret_vec = get_weights(self.net)

        # Apply DP defenses
        defense_config = config.get("defense", "")
        if "gaussian" in defense_config:
            ret_vec = add_gaussian_noise(ret_vec, stddev=0.05)
            print("[Defense] Gaussian noise applied to gradients")
        if "clip" in defense_config:
            ret_vec = clip_gradients(ret_vec, clip_value=0.1)
            print("[Defense] Gradient clipping applied")

        # Force a significant delay for testing purposes
        if self.is_demo:
            if config.get("drop", False):
                print(f"Client dropped for testing purposes.")
                time.sleep(self.timeout)
            else:
                print(f"Client uploading parameters: {ret_vec[0].flatten()[:3]}...")

            # Simulate a gradient inversion attack if this is the attacker client
            if config.get("attacker", False):
                print("[Attacker] Running gradient inversion attack...")
                from secaggexample.attacks.gradient_inversion import gradient_inversion_attack

                if isinstance(self.valloader, Mock):
                    print("[Attacker] Skipping attack: val loader is mocked in demo mode.")
                else:
                    # Select one sample from validation set
                    for batch in self.valloader:
                        image, label = batch["img"][0:1].to(self.device), batch["label"][0:1].to(self.device)
                        break

                    self.net.eval()
                    self.net.zero_grad()
                    output = self.net(image)
                    loss = torch.nn.functional.cross_entropy(output, label)
                    gradients = torch.autograd.grad(loss, self.net.parameters())

                    dummy_data, dummy_label = gradient_inversion_attack(
                        model=self.net,
                        target_gradients=gradients,
                        label=label,
                        num_classes=10,
                        device=self.device,
                        iters=300,
                    )

                    import torch.nn.functional as F
                    mse = F.mse_loss(dummy_data, image).item()
                    print(f"[Attacker] Gradient inversion completed. MSE similarity = {mse:.6f}")

                    # Save reconstructed and original image
                    import os
                    from torchvision.utils import save_image

                    os.makedirs("attack_outputs", exist_ok=True)

                    round_tag = config.get("current_round", config.get("round", "unknown"))
                    save_image(dummy_data.clamp(0, 1), f"attack_outputs/reconstructed_{round_tag}.png")
                    save_image(image.clamp(0, 1), f"attack_outputs/original_{round_tag}.png")
                    print(f"[Attacker] Reconstructed image saved to attack_outputs/reconstructed_{round_tag}.png")
                    print(f"[Attacker] Ground truth image saved to attack_outputs/original_{round_tag}.png")

        return ret_vec, len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = 0.0, 0.0
        if not self.is_demo:
            loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    is_demo = context.run_config["is-demo"]
    trainloader, valloader = load_data(
        partition_id, num_partitions, batch_size, is_demo
    )
    local_epochs = context.run_config["local-epochs"]
    lr = context.run_config["learning-rate"]
    # For demostrations purposes only
    timeout = context.run_config["timeout"]

    # Return Client instance
    return FlowerClient(
        trainloader, valloader, local_epochs, lr, timeout, is_demo
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)