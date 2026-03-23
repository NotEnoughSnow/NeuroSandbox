import numpy as np
from scipy.linalg import lstsq
import torch

def run(w, h, x, external_drive, network, hp):

    train_timesteps = hp["timesteps"]["train"]
    plasticity_timesteps = hp["timesteps"]["plasticity"]
    k = hp["k"]
    n = hp["n"]
    tau = hp["tau"]
    n_symbols = hp["n_symbols"]
    verbosity = hp["verbosity"]

    states_train = [] # to store states for analysis
    labels_train = [] # to store corresponding symbols for analysis

    symbols = external_drive.generate_symbols(train_timesteps)

    for t in range(train_timesteps):

        drive = external_drive.get_symbol(symbols, t)

        pre_activation = w @ x - h + drive
        x = network.kWTA(pre_activation, k=k)

        # for lag τ = -1: label is the symbol from 1 step ago
        # so at time t, the label is symbols[20000 + t - 1]
        label_t = t + tau
        if 0 <= label_t < len(symbols):
            states_train.append(x.copy()) # store current state
            labels_train.append(symbols[label_t])

    print("Done training.")

    # convert to matrice
    X_train =  np.array(states_train) # shape (train_timesteps -1, n)
    y_train = np.array(labels_train) # shape (train_timesteps -1,)

    Y_train = np.zeros((len(y_train), n_symbols)) # shape (train_timesteps - 1, 4) # one-hot matrix of shape (4999, 4)
    Y_train[np.arange(len(y_train)), y_train] = 1

    if verbosity == 1:
        print("Training data shape:", X_train.shape)
        print("Training labels shape:", Y_train.shape)

    w_out, _, _, _ = lstsq(X_train, Y_train) # shape (n, 4)

    if verbosity == 1:
        print("w_out shape:", w_out.shape)  # should be (100, 4)

    return w_out

def run_batch(W, H, X, external_drive, network, hp):

    train_timesteps = hp["timesteps"]["train"]
    plasticity_timesteps = hp["timesteps"]["plasticity"]
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

        #drive = external_drive.get_symbol(symbols, t)
        drive = all_drives[t]

        #pre_activation = w @ x - h + drive
        pre_activation = torch.bmm(W, X.unsqueeze(-1)).squeeze(-1) - H + drive

        #x = network.kWTA(pre_activation, k=k)
        X_new = network.kWTA_batch_torch(pre_activation, k=hp['k'])



        # for lag τ = -1: label is the symbol from 1 step ago
        # so at time t, the label is symbols[20000 + t - 1]
        label_t = t + tau
        if 0 <= label_t < len(symbols):
            states_list.append(X_new.cpu().numpy())  # (N, n)
            labels_list.append(symbols[label_t])  # (N,) - one label per network

    print("Done training.")

    # stack collected data
    # states: (T_valid, N, n)
    # labels: (T_valid, N)
    states = np.array(states_list)
    labels = np.array(labels_list)

    T_valid = len(states_list)

    # solve linear regression separately for each network
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

    #if verbosity == 1:
    #    print("Training data shape:", X_train.shape)
    #    print("Training labels shape:", Y_train.shape)

    #if verbosity == 1:
    #    print("w_out shape:", W_out.shape)  # should be (100, 4)

    return W_out

