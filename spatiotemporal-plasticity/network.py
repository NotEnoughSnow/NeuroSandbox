import numpy as np
import torch

class Kwta_RN_batch:
    # k-Winners-Take-All Recurrent Network
    def __init__(self, hp):
        self.n = hp["n"]
        self.k = hp["k"]
        self.n_networks = hp["n_networks"]

    def init_network(self):
        # vectorized initialization
        W = np.random.uniform(0, 0.1, (self.n_networks, self.n, self.n))
        mask = np.random.rand(self.n_networks, self.n, self.n) < 0.1
        W = W * mask
        # zero diagonal for all networks at once
        diag_idx = np.arange(self.n)
        W[:, diag_idx, diag_idx] = 0

        H = np.random.normal(0, 0.1, (self.n_networks, self.n))
        X = self.generate_random_kwta_state()
        return W, H, X


    def generate_random_kwta_state(self):
        x = np.zeros((self.n_networks, self.n))

        for k in range(self.n_networks):
            active_neurons = np.random.choice(self.n, self.k, replace=False)

            for neuron in active_neurons:
                x[k][neuron] = 1

        return x

    def kWTA(self, input_tensor, k):
        top_k_indices = np.argsort(input_tensor, axis=-1)[:, -k:]
        output_tensor = np.zeros_like(input_tensor)
        batch_indices = np.arange(self.n_networks)[:, None]
        output_tensor[batch_indices, top_k_indices] = 1
        return output_tensor

    def kWTA_batch_torch(self, input_tensor, k):
        _, top_k_idx = torch.topk(input_tensor, k, dim=1)
        result = torch.zeros_like(input_tensor)
        result.scatter_(1, top_k_idx, 1.0)
        return result