import numpy as np
from scipy.linalg import lstsq
import torch


def run_batch(W, H, X, external_drive, network, hp):

    train_timesteps = hp["timesteps"]["train"]
    k = hp["k"]
    n = hp["n"]
    tau = hp["tau"]
    n_symbols = hp["n_symbols"]
    verbosity = hp["verbosity"]
    device = hp["device"]
    n_networks = hp["n_networks"]

    states_list = [] # to store states for analysis
    labels_list = [] # to store corresponding symbols for analysis

    symbols = external_drive.generate_symbols(train_timesteps)
    all_drives = external_drive.precompute_all_drives(symbols)  # shape (t_plasticity, N, n)
    all_drives = torch.tensor(all_drives, dtype=torch.float64, device=device)

    for t in range(train_timesteps):

        drive = all_drives[t]

        pre_activation = torch.bmm(W, X.unsqueeze(-1)).squeeze(-1) - H + drive

        X = network.kWTA_batch_torch(pre_activation, k=hp['k'])

        label_t = t + tau
        if 0 <= label_t < len(symbols):
            states_list.append(X.cpu().numpy())  # (N, n)
            labels_list.append(symbols[label_t])  # (N,) - one label per network

    print("Done training.")

    states = np.array(states_list)
    labels = np.array(labels_list)

    T_valid = len(states_list)

    # W_out shape: (N, n, n_symbols)
    W_out = np.zeros((n_networks, n, n_symbols))

    for i in range(n_networks):
        # X_train for network i: (T_valid, n)
        X_train = states[:, i, :]

        # one-hot encode labels for network i: (T_valid, n_symbols)
        Y_train = np.zeros((T_valid, n_symbols))
        Y_train[np.arange(T_valid), labels[:, i]] = 1

        # solve: w_out = (X^T X)^{-1} X^T Y
        w_out_i, _, _, _ = lstsq(X_train, Y_train)  # (n, n_symbols)
        W_out[i] = w_out_i

    return W_out

