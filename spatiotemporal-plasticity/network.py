import numpy as np
import torch

class Kwta_RN:
    # k-Winners-Take-All Recurrent Network
    def __init__(self, hp):
        self.n = hp["n"]
        self.k = hp["k"]

    def init_network(self):
        # weight matrix: sparse, uniform [0, 0.1], no self connections

        # Generate uniform values for ALL connections (including diagonal)
        w = np.random.uniform(0, 0.1, (self.n, self.n))

        # Generate mask for ALL connections (including diagonal)
        mask = np.random.rand(self.n, self.n) < 0.1

        # Apply mask
        w = w * mask

        # Zero diagonal (no self connections)
        np.fill_diagonal(w, 0)

        # threshold array, h[i] ~ Gaussian(mean=0, std=0.1)
        h = np.random.normal(0, 0.1, self.n)

        x = self.generate_random_kwta_state()

        return w, h, x


    def generate_random_kwta_state(self):
        # Initial network state: random kWTA-valid state
        # pick 12 random neurons to be active

        active_neurons = np.random.choice(self.n, self.k, replace=False)

        x = np.zeros(self.n)

        for neuron in active_neurons:
            x[neuron] = 1

        return x

    def kWTA(self, input_vector, k):
        # kWTA function

        # get indices of top k values
        top_k_indices = np.argsort(input_vector)[-k:]
        # create output vector with 1s at top k indices, else 0
        output_vector = np.zeros_like(input_vector)
        output_vector[top_k_indices] = 1
        return output_vector

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
        # Initial network state: random kWTA-valid state
        # pick 12 random neurons to be active


        x = np.zeros((self.n_networks, self.n))

        for k in range(self.n_networks):
            active_neurons = np.random.choice(self.n, self.k, replace=False)

            for neuron in active_neurons:
                x[k][neuron] = 1

        return x

    def kWTA(self, input_tensor, k):
        # kWTA function

        # get indices of top k values
        top_k_indices = np.argsort(input_tensor, axis=-1)[:, -k:]
        # create output vector with 1s at top k indices, else 0
        output_tensor = np.zeros_like(input_tensor)
        batch_indices = np.arange(self.n_networks)[:, None]
        output_tensor[batch_indices, top_k_indices] = 1
        return output_tensor

    def kWTA_batch_torch(self, input_tensor, k):
        # activation shape: (N, n)
        _, top_k_idx = torch.topk(input_tensor, k, dim=1)
        # top_k_idx shape: (N, k)
        result = torch.zeros_like(input_tensor)
        result.scatter_(1, top_k_idx, 1.0)
        # scatter_ sets result[n, top_k_idx[n, j]] = 1 for all n, j
        return result