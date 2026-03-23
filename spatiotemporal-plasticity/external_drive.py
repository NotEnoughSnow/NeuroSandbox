import numpy as np

class Drive:

    def __init__(self, hp):

        self.strength = hp["drive_strength"]
        self.n = hp["n"]
        self.num_symbols = hp["n_symbols"]
        self.total_timesteps = hp["timesteps"]["total"]

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

        #self.symbols = self.generate_symbols(self.total_timesteps)
        #print(f"Total symbol stream length: {len(symbols)}")
        #print("First 10 symbols:", symbols[:10])
        #print(f"Symbol counts: { {i: np.sum(symbols == i) for i in range(4)} }")

        # quick sanity check
        #test_drive = self.make_drive(0)
        #print(f"Drive vector for symbol A:")
        #print(f"  neurons 0-14:  {test_drive[:15]}")  # should be 0.25
        #print(f"  neurons 15-29: {test_drive[15:30]}")  # should be 0.0
        #print(f"  nonzero count: {np.sum(test_drive > 0)}")  # should be 15

    def get_symbol(self, symbols, t):
        # get symbol at timestep t
        symbol = symbols[t]
        return self.make_drive(symbol)

    def generate_symbols(self, l):
        # create a random sequence of symbols for T timesteps
        # symbols = [random.choice([A,B,C,D]) for _ in range(total_timesteps)]
        return np.random.randint(self.num_symbols, size=l)

    def make_drive(self, symbol):
        # build drive vector
        drive = np.zeros(self.n)
        for neuron in self.rf[symbol]:
            drive[neuron] = self.strength

        return drive



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

        #self.symbols = self.generate_symbols(self.total_timesteps)
        #print(f"Total symbol stream length: {len(symbols)}")
        #print("First 10 symbols:", symbols[:10])
        #print(f"Symbol counts: { {i: np.sum(symbols == i) for i in range(4)} }")

        # quick sanity check
        #test_drive = self.make_drive(0)
        #print(f"Drive vector for symbol A:")
        #print(f"  neurons 0-14:  {test_drive[:15]}")  # should be 0.25
        #print(f"  neurons 15-29: {test_drive[15:30]}")  # should be 0.0
        #print(f"  nonzero count: {np.sum(test_drive > 0)}")  # should be 15

    def get_symbol(self, symbols, t):
        # get symbol at timestep t
        current_symbols = symbols[t]
        return self.make_drive(current_symbols)

    def generate_symbols(self, l):
        # create a random sequence of symbols for T timesteps
        # symbols = [random.choice([A,B,C,D]) for _ in range(total_timesteps)]
        return np.random.randint(self.num_symbols, size=(l, self.n_networks))

    def make_drive(self, symbols_at_t):
        # build drive vector
        drive = np.zeros((self.n_networks, self.n))

        for batch_idx, symbol in enumerate(symbols_at_t):
            # For each network, find which neurons to activate based on the symbol
            for neuron in self.rf[symbol]:
                drive[batch_idx, neuron] = self.strength

        return drive

    def precompute_all_drives(self, symbols):
        """
        precomputes drive vectors for all timesteps at once.

        symbols shape: (timesteps, n_networks) - one symbol per network per timestep
        returns: drives shape (timesteps, n_networks, n)
        """
        timesteps = len(symbols)
        drives = np.zeros((timesteps, self.n_networks, self.n))

        for t in range(timesteps):
            drives[t] = self.make_drive(symbols[t])
            # make_drive takes symbols_at_t shape (n_networks,)
            # returns drive shape (n_networks, n)

        return drives

