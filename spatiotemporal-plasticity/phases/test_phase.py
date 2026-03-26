import numpy as np
import torch

def run_batch(W, H, X, W_out, network, external_drive, hp):
    # testing phase

    train_timesteps = hp["timesteps"]["train"]
    test_timesteps = hp["timesteps"]["test"]
    tau = hp["tau"]
    n_symbols = hp["n_symbols"]
    verbosity = hp["verbosity"]
    n_networks = hp["n_networks"]
    device = hp["device"]

    X = torch.tensor(network.generate_random_kwta_state(), dtype=torch.float64, device=device) # start from a random state for testing

    states_list = []
    labels_list = []

    symbols = external_drive.generate_symbols(train_timesteps)
    all_drives = external_drive.precompute_all_drives(symbols)  # shape (t_plasticity, N, n)
    all_drives = torch.tensor(all_drives, dtype=torch.float64, device=device)

    for t in range(test_timesteps):

        drive = all_drives[t]

        pre_activation = torch.bmm(W, X.unsqueeze(-1)).squeeze(-1) - H + drive

        X = network.kWTA_batch_torch(pre_activation, k=hp['k'])

        label_t = t + tau
        if 0 <= label_t < len(symbols):
            states_list.append(X.cpu().numpy())  # (N, n)
            labels_list.append(symbols[label_t])  # (N,) - one label per network

    states = np.array(states_list)    # (T_valid, N, n)
    labels = np.array(labels_list)    # (T_valid, N)

    # apply readout and compute performance for each network
    performance = np.zeros(n_networks)

    for i in range(n_networks):
        X_test = states[:, i, :]  # (T_valid, n)
        scores = X_test @ W_out[i]  # (T_valid, n_symbols)
        predictions = np.argmax(scores, axis=1)  # (T_valid,)
        performance[i] = np.mean(predictions == labels[:, i]) * 100

    print(f"Done testing.")

    if verbosity == 1:
        print(f"Testing done.")
        print(f"Performance at tau={tau}: {performance:.2f}%")
        print(f"Chance level: {100 / n_symbols:.2f}%")

    return performance