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

    def generate_symbols(self, l, randomize):

        if randomize:
            symbols = np.random.randint(self.num_symbols, size=(l, self.n_networks))
        else:
            symbols = self.generate_symbols_markov(l)

        return symbols

    def generate_symbols_markov(self, length):
        """
        generates symbols from a Markov chain with 85% probability
        of following the cycle A->B->C->D->A, and 5% for all other transitions.
        returns shape (length, n_networks)
        """
        # transition matrix: trans[i][j] = P(next=j | current=i)
        # cycle: 0->1->2->3->0 with p=0.85
        trans = np.array([
            [0.05, 0.85, 0.05, 0.05],  # from A: go to B with 85%
            [0.05, 0.05, 0.85, 0.05],  # from B: go to C with 85%
            [0.05, 0.05, 0.05, 0.85],  # from C: go to D with 85%
            [0.85, 0.05, 0.05, 0.05],  # from D: go to A with 85%
        ])

        symbols = np.zeros((length, self.n_networks), dtype=int)

        # random starting symbol for each network
        symbols[0] = np.random.randint(0, 4, size=self.n_networks)

        for t in range(1, length):
            for i in range(self.n_networks):
                current = symbols[t - 1, i]
                symbols[t, i] = np.random.choice(4, p=trans[current])

        return symbols

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

class Drive_batch_parity:

    def __init__(self, hp):
        self.strength = hp["drive_strength"]
        self.n = hp["n"]
        self.num_symbols = 2           # binary: {0, 1}
        self.n_networks = hp["n_networks"]

        # receptive fields: 2 symbols, 40 neurons each
        # symbol 0: neurons 0-39
        # symbol 1: neurons 40-79
        # neurons 80-99: purely recurrent, no direct input
        self.rf = {
            0: list(range(0, 40)),
            1: list(range(40, 80)),
        }

    def generate_symbols(self, length):
        # binary uniform random: shape (length, n_networks)
        return np.random.randint(0, 2, size=(length, self.n_networks))

    def make_drive(self, symbols_at_t):
        # symbols_at_t shape: (n_networks,)
        # returns drive shape: (n_networks, n)
        drive = np.zeros((self.n_networks, self.n))
        for batch_idx, symbol in enumerate(symbols_at_t):
            drive[batch_idx, self.rf[symbol]] = self.strength
        return drive

    def precompute_all_drives(self, symbols):
        # symbols shape: (timesteps, n_networks)
        # returns drives shape: (timesteps, n_networks, n)
        timesteps = len(symbols)
        drives = np.zeros((timesteps, self.n_networks, self.n))
        for t in range(timesteps):
            drives[t] = self.make_drive(symbols[t])
        return drives
