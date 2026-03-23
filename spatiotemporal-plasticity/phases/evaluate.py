import numpy as np
from npeet import entropy_estimators as ee

class Evaluator:

    def __init__(self, network, external_drive, hp):
        # Calculate Entropy and mutual information

        self.n_samples = 5000 # number of samples to estimate H and MI
        self.k = hp["k"]
        n_symbols = hp["n_symbols"]
        self.network = network
        self.external_drive = external_drive

        # collects network states and 3-step input encodings.
        # estimates entropy H(X) and mutual information I(U, X).

        # the 3-bit encoding for symbols (equal pairwise hamming distances):
        # A=0 -> [0,0,0]
        # B=1 -> [0,1,1]
        # C=2 -> [1,0,1]
        # D=3 -> [1,1,0]
        self.symbol_encoding = {
            0: [0, 0, 0],
            1: [0, 1, 1],
            2: [1, 0, 1],
            3: [1, 1, 0],
        }

    def eval(self, w, h):
        x = self.network.generate_random_kwta_state() # fresh random start for analysis
        states_info = []
        input_sequences = []

        # use a fresh section of the symbol stream
        # or just regenerate for the info phase
        info_symbols = self.external_drive.generate_symbols(self.n_samples + 3)

        for t in range(3, self.n_samples + 3):
            symbol = info_symbols[t]
            drive = self.external_drive.make_drive(symbol)

            pre_activation = w @ x - h + drive
            x = self.network.kWTA(pre_activation, k=self.k)

            states_info.append(x.copy())

            # encode the 3 most recent symbols as a flat vector
            u = (self.symbol_encoding[info_symbols[t-2]] + self.symbol_encoding[info_symbols[t-1]] + self.symbol_encoding[info_symbols[t]])

            input_sequences.append(u)

        return states_info, input_sequences

    def calculate_entropy(self, states):
        # entropy H(X): how many distinct states does the network visit,
        # and how uniformly?

        states = np.array(states)

        '''
        # run this before calling ee.entropy() to see what's happening
        print("DIAGNOSIS")
        print(f"Number of states collected: {len(states)}")
        print(f"Number of UNIQUE states: {len(set(map(tuple, states.astype(int))))}")
        print(f"State dimensionality: {states.shape[1]}")

        # check if states are too similar
        from scipy.spatial.distance import pdist
        distances = pdist(states[:500], metric='hamming')
        print(f"Min pairwise distance: {distances.min():.4f}")
        print(f"Mean pairwise distance: {distances.mean():.4f}")
        print(f"Fraction of zero distances: {np.mean(distances == 0):.4f}")
        print("DIAGNOSIS")
        '''

        # estimate H(X)
        H = ee.entropy(states)
        return H

    def calculate_mutual_info(self, input_sequences, states):

        # mutual information I(U, X): how much does knowing the
        # last 3 symbols tell you about the current network state?

        input_sequences = np.array(input_sequences, dtype=float)  # shape (n_samples, 9)

        # estimate I(U, X)
        MI = ee.mi(input_sequences, states)
        return MI

    def generate_metrics(self, w, h):

        states_info, input_sequences = self.eval(w, h)

        states = np.array(states_info)  # shape (n_samples, n)

        H = self.calculate_entropy(states_info)
        MI = self.calculate_mutual_info(input_sequences, states)

        '''
        print(f"Entropy H(X):          {H:.4f} bits")
        print(f"Mutual Info I(U,X):    {MI:.4f} bits")
        print(f"Optimal MI would be:   6.0000 bits (log2(4^3) for 3-step input)")
        '''

        return H, MI