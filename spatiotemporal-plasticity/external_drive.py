import numpy as np

class Drive_batch:

    def __init__(self, hp):

        self.strength = hp["drive_strength"]
        self.n = hp["n"]
        self.num_symbols = hp["n_symbols"]
        self.total_timesteps = hp["timesteps"]["total"]
        self.n_networks = hp["n_networks"]

        # Receptive fields: fixed, non-overlapping
        # RF[A] = neurons 0–14
        # RF[B] = neurons 15–29
        # RF[C] = neurons 30–44
        # RF[D] = neurons 45–59

        self.rf = {
            0: list(range(0, 15)),     # A
            1: list(range(15, 30)),    # B
            2: list(range(30, 45)),    # c
            3: list(range(45, 60))     # D
        }
        # neurons 60–99 belong to no receptive field

    def get_symbol(self, symbols, t):
        current_symbols = symbols[t]
        return self.make_drive(current_symbols)

    def generate_symbols(self, l):
        return np.random.randint(self.num_symbols, size=(l, self.n_networks))

    def make_drive(self, symbols_at_t):
        # build drive vector
        drive = np.zeros((self.n_networks, self.n))

        for batch_idx, symbol in enumerate(symbols_at_t):
            for neuron in self.rf[symbol]:
                drive[batch_idx, neuron] = self.strength

        return drive

    def precompute_all_drives(self, symbols):
        timesteps = len(symbols)
        drives = np.zeros((timesteps, self.n_networks, self.n))

        for t in range(timesteps):
            drives[t] = self.make_drive(symbols[t])

        return drives

