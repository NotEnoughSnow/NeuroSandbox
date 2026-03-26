
import numpy as np
import torch



def run_batch(x, hp):

    k = hp["k"]
    pi = hp["pi"]
    verbosity = hp["verbosity"]
    n_networks = hp["n_networks"]
    device = hp["device"]

    X_np = x.cpu().numpy()
    X_init = X_np.copy()

    for i in range(n_networks):
        firing = np.where(X_np[i] == 1)[0]  # indices of active neurons
        silent = np.where(X_np[i] == 0)[0]  # indices of silent neurons

        flip_off = np.random.choice(firing, pi, replace=False)
        flip_on = np.random.choice(silent, pi, replace=False)

        X_init[i, flip_off] = 0
        X_init[i, flip_on] = 1

        # verify kWTA constraint still holds
        assert X_init[i].sum() == k

    X_purturb = torch.tensor(X_init, dtype=torch.float64, device=device)

    if verbosity == 1:
        print("Perturbed initial state:", X_purturb)

    print("Done running perturbation.")

    return X_purturb