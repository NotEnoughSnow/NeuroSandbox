from numpy import clip
import numpy as np
import torch

def run(external_drive, network, hp):

    w, h, x = network.init_network()

    plasticity_timesteps = hp["timesteps"]["plasticity"]
    k = hp["k"]
    n = hp["n"]
    eta_sp = hp["eta_sp"]
    eta_ip = hp["eta_ip"]
    verbosity = hp["verbosity"]

    w = w.copy()
    h = h.copy()
    x = x.copy()

    # track these over time
    unique_states_log = []
    w_mean_log = []
    h_mean_log = []
    active_neurons_log = []  # how many neurons ever fire

    recent_states = []
    neurons_fired = set()

    symbols = external_drive.generate_symbols(plasticity_timesteps)

    print("3move at plasticity ", symbols)

    for t in range(plasticity_timesteps):


        drive = external_drive.get_symbol(symbols, t)


        # 2. compute pre-activation for all neurons
        # pre_activation = w @ x - h + d, shape (100,)
        pre_activation = w @ x - h + drive

        # 3. apply kWTA: set top k pre-activations to 1, rest to 0
        x_new = network.kWTA(pre_activation, k=hp['k'])

        # 4. STDP update (needs both x at t and x_new at t+1)
        dw = eta_sp * (np.outer(x_new, x) - np.outer(x, x_new))
        w = w + dw
        w = np.clip(w, 0, 1)
        np.fill_diagonal(w, 0)


        # SLOW
        #for i in range(n):
        #    for j in range(n):
                #delta_w = eta_sp * (x[j] * x_new[i] - x_new[j] * x[i])  # STDP rule
                #delta_w = eta_sp * (x[j] * x_new[i] - x_new[j] * x_new[i]) XXX test with the wrong version for fun

        #        w[i][j] = clip(w[i][j] + delta_w, 0, 1)  # clip updates to prevent explosion

        # ensure no self connections
        #np.fill_diagonal(w, 0)

        # 5. IP update
        dh = eta_ip * (x_new - k / n)
        h = h + dh

        # SLOW
        #for i in range(n):
        #    delta_h = eta_ip * (x_new[i] - k / n)  # IP rule, target activity 0.1
        #    h[i] = h[i] + delta_h

        # 6. advance state
        x = x_new

        # logging
        recent_states.append(tuple(x_new.astype(int)))
        neurons_fired.update(np.where(x_new == 1)[0].tolist())

        # progress check
        if (t + 1) % 1000 == 0:
            unique_in_window = len(set(recent_states[-1000:]))
            unique_states_log.append(unique_in_window)
            w_mean_log.append(w.mean())
            h_mean_log.append(h.mean())
            active_neurons_log.append(len(neurons_fired))
            neurons_fired = set()  # reset window

            '''
            print(f"step {t + 1:5d} | "
                  f"unique states (last 1k): {unique_in_window:4d} | "
                  f"w mean: {w.mean():.4f} | "
                  f"h mean: {h.mean():.4f} | "
                  f"h range: [{h.min():.4f}, {h.max():.4f}] | "  # ADD THIS
                  f"active neurons: {active_neurons_log[-1]:3d}") '''



    print("Done running plasticity.")

    if verbosity == 1:
        print(f"w range: [{w.min():.4f}, {w.max():.4f}]")
        print(f"h range: [{h.min():.4f}, {h.max():.4f}]")

    return w, h, x

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

    # track these over time
    unique_states_log = []
    w_mean_log = []
    h_mean_log = []
    active_neurons_log = []  # how many neurons ever fire

    recent_states = []
    neurons_fired = set()


    diag_idx = torch.arange(n, device=device)

    symbols = external_drive.generate_symbols(plasticity_timesteps)
    all_drives = external_drive.precompute_all_drives(symbols)  # shape (t_plasticity, N, n)
    all_drives = torch.tensor(all_drives, dtype=torch.float64, device=device)

    #print("3move ", symbols)

    for t in range(plasticity_timesteps):


        #drive = torch.tensor(external_drive.get_symbol(symbols, t))
        drive = all_drives[t]

        # 2. compute pre-activation for all neurons
        # W shape: (N, n, n)
        # X shape: (N, n) → unsqueeze to (N, n, 1) for bmm
        pre_activation = torch.bmm(W, X.unsqueeze(-1)).squeeze(-1) - H + drive
        # result shape: (N, n)

        # 3. apply kWTA: set top k pre-activations to 1, rest to 0
        X_new = network.kWTA_batch_torch(pre_activation, k=hp['k'])

        # 4. STDP update (needs both x at t and x_new at t+1)
        # X shape:     (N, n) - old state
        # X_new shape: (N, n) - new state

        # outer products for all networks
        potentiation = torch.bmm(X_new.unsqueeze(2), X.unsqueeze(1))  # (N, n, n)
        depression = torch.bmm(X.unsqueeze(2), X_new.unsqueeze(1))  # (N, n, n)

        dW = eta_sp * (potentiation - depression)  # (N, n, n)
        W = torch.clamp(W + dW, 0, 1)  # (N, n, n)

        # zero diagonal for all networks
        W[:, diag_idx, diag_idx] = 0


        # 5. IP update
        # X_new shape: (N, n)
        # H shape:     (N, n)
        # k/n is a scalar, broadcasts automatically

        dH = eta_ip * (X_new - k / n)  # (N, n)
        H = H + dH  # (N, n)



        # 6. advance state
        X = X_new

        # logging
        #recent_states.append(tuple(X_new.astype(int)))
        #neurons_fired.update(np.where(X_new == 1)[0].tolist())

        # progress check
        if (t + 1) % 1000 == 0:
            #unique_in_window = len(set(recent_states[-1000:]))
            #unique_states_log.append(unique_in_window)
            #w_mean_log.append(W.mean())
            #h_mean_log.append(H.mean())
            #active_neurons_log.append(len(neurons_fired))
            #neurons_fired = set()  # reset window

            '''
            print(f"step {t + 1:5d} | "
                  f"unique states (last 1k): {unique_in_window:4d} | "
                  f"w mean: {w.mean():.4f} | "
                  f"h mean: {h.mean():.4f} | "
                  f"h range: [{h.min():.4f}, {h.max():.4f}] | "  # ADD THIS
                  f"active neurons: {active_neurons_log[-1]:3d}") '''

    if verbosity == 1:
        print("Done running plasticity.")

    if verbosity == 2:
        print(f"w range: [{W.min():.4f}, {W.max():.4f}]")
        print(f"h range: [{H.min():.4f}, {H.max():.4f}]")

    return W, H, X