import numpy as np
from network import Kwta_RN_batch
from external_drive import Drive_batch
from phases import plasticity_phase
from phases import pertubation_phase
from phases import train_phase
from phases import test_phase
from phases import evaluate
import torch

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # reproducibility
    np.random.seed(42)

    timesteps = {
        # phases
        "plasticity": 20000,  # originally 20000
        "train": 5000,  # originally 5000
        "test": 5000,  # originally 5000
    }

    timesteps["total"] = (timesteps["plasticity"] +
                          timesteps["train"] +
                          timesteps["train"])

    tau_array = list(range(-8,9))
    #tau_array = [-1]
    print(tau_array)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    hyperparameters = {
        # network parameters
        "n": 100,  # num neurons
        "k": 12,  # winners
        "eta_sp": 0.001,  # learning rate for synaptic plasticity
        "eta_ip": 0.001,  # learning rate for intrinsic plasticity
        # very poor performance when eta_sp and eta_ip are 0.01

        # task parameters
        "n_symbols": 4,  # A, B, C, D
        "rf_size": 15,  # receptive field size for each symbol
        "drive_strength": 0.25,  # input drive value

        "timesteps": timesteps,

        "verbosity": 0,

        # Perturbation
        "pi": 12,

        # number of networks to run paralelly
        "n_networks": 1,

        "device": device,

    }

    network_types = ["SIP"]
    n_networks = hyperparameters["n_networks"]

    performance_temp = []

    for t in tau_array:
        print(f"For timelag {t} --------")

        hyperparameters["tau"] = t

        drive = Drive_batch(hp=hyperparameters)
        network = Kwta_RN_batch(hp=hyperparameters)

        w, h, x_p = plasticity_phase.run_batch(external_drive=drive, network=network, hp=hyperparameters)

        x_per = pertubation_phase.run_batch(x=x_p, hp=hyperparameters)

        x_out = train_phase.run_batch(W=w, H=h, X=x_per, external_drive=drive, network=network, hp=hyperparameters)

        performance = test_phase.run_batch(W=w, H=h, X=x_per, W_out=x_out, external_drive=drive, network=network, hp=hyperparameters)

        performance_temp.append(performance)

    performance_array = np.mean(performance_temp, axis=1)
    print(performance_array)

    plt.figure(figsize=(6, 5))

    # Plot performance
    plt.plot(tau_array, performance_array,
             color='blue', marker='o', linestyle='-', markersize=4,
             label='Performance')

    # Add chance level line at 25%
    plt.axhline(y=25, color='red', linestyle='--', label='Chance level (25%)', alpha=0.7)

    # Labels and formatting
    plt.xlabel('Time lag τ')
    plt.ylabel('Performance (%)')
    plt.title('Performance vs Time Lag')
    plt.ylim(0, 105)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

