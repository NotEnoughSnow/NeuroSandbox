from numpy import clip
import numpy as np
import torch

def run_batch(external_drive, network, hp):

    W, H, X = network.init_network()

    plasticity_timesteps = hp["timesteps"]["plasticity"]
    k = hp["k"]
    n = hp["n"]
    eta_sp = hp["eta_sp"]
    eta_ip = hp["eta_ip"]
    verbosity = hp["verbosity"]
    n_networks = hp["n_networks"]
    device = hp["device"]

    # move to device
    W = torch.tensor(W, dtype=torch.float64, device=device)
    H = torch.tensor(H, dtype=torch.float64, device=device)
    X = torch.tensor(X, dtype=torch.float64, device=device)

    diag_idx = torch.arange(n, device=device)

    symbols = external_drive.generate_symbols(plasticity_timesteps)
    all_drives = external_drive.precompute_all_drives(symbols)  # shape (t_plasticity, N, n)
    all_drives = torch.tensor(all_drives, dtype=torch.float64, device=device)

    for t in range(plasticity_timesteps):
        drive = all_drives[t]

        # compute pre-activation for all neurons
        pre_activation = torch.bmm(W, X.unsqueeze(-1)).squeeze(-1) - H + drive
        # result shape: (N, n)

        # apply kWTA: set top k pre-activations to 1, rest to 0
        X_new = network.kWTA_batch_torch(pre_activation, k=hp['k'])

        # STDP update (needs both x at t and x_new at t+1)
        # outer products for all networks
        potentiation = torch.bmm(X_new.unsqueeze(2), X.unsqueeze(1))  # (N, n, n)
        depression = torch.bmm(X.unsqueeze(2), X_new.unsqueeze(1))  # (N, n, n)

        dW = eta_sp * (potentiation - depression)  # (N, n, n)
        W = torch.clamp(W + dW, 0, 1)  # (N, n, n)

        # zero diagonal for all networks
        W[:, diag_idx, diag_idx] = 0

        # IP update
        dH = eta_ip * (X_new - k / n)  # (N, n)
        H = H + dH  # (N, n)

        X = X_new

    if verbosity == 1:
        print("Done running plasticity.")

    if verbosity == 2:
        print(f"w range: [{W.min():.4f}, {W.max():.4f}]")
        print(f"h range: [{H.min():.4f}, {H.max():.4f}]")

    return W, H, X