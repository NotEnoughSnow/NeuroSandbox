import numpy as np
from network import Kwta_RN
from external_drive import Drive
from phases import plasticity_phase
from phases import pertubation_phase
from phases import train_phase
from phases import test_phase
from phases import evaluate

import matplotlib.pyplot as plt



def graph_results(results, tau_array=None):
    """
    Graph performance, entropy, and mutual information results

    Parameters:
    -----------
    results : dict
        Nested dictionary with structure: results[metric][condition] = list of values
        metrics: 'performance', 'H', 'MI'
        conditions: 'SIP', 'SP', 'IP', 'nP' (etc.)
    tau_array : list, optional
        List of tau values for x-axis. If None, will use index numbers
    """

    # If tau_array not provided, create default
    if tau_array is None:
        # Try to determine length from first available data
        for metric in results:
            for condition in results[metric]:
                tau_array = list(range(len(results[metric][condition])))
                break
            break

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Colors for different conditions (useful when you add more)
    colors = {'SIP': 'blue', 'SP': 'green', 'IP': 'orange', 'nP': 'purple'}

    # 1. Performance graph
    ax = axes[0]
    for condition in results.get('performance', {}):
        if results['performance'][condition]:  # Check if not empty
            ax.plot(tau_array[:len(results['performance'][condition])],
                    results['performance'][condition],
                    label=condition, color=colors.get(condition, 'black'),
                    marker='o', linestyle='-', markersize=4)

    # Add chance level line at 25%
    ax.axhline(y=25, color='red', linestyle='--', label='Chance level (25%)', alpha=0.7)

    ax.set_xlabel('Time lag τ')
    ax.set_ylabel('Performance (%)')
    ax.set_title('Performance vs Time Lag')
    ax.set_ylim(0, 105)  # Give some headroom above 100%
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 2. Entropy graph H(X)
    ax = axes[1]
    for condition in results.get('H', {}):
        if results['H'][condition]:
            ax.plot(tau_array[:len(results['H'][condition])],
                    results['H'][condition],
                    label=condition, color=colors.get(condition, 'black'),
                    marker='s', linestyle='-', markersize=4)

    ax.set_xlabel('Time lag τ')
    ax.set_ylabel('Entropy H(X) (bits)')
    ax.set_title('State Entropy vs Time Lag')
    ax.set_ylim(0, 14)  # Set Y-axis from 0 to 14
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # 3. Mutual Information graph I(X, U)
    ax = axes[2]
    for condition in results.get('MI', {}):
        if results['MI'][condition]:
            ax.plot(tau_array[:len(results['MI'][condition])],
                    results['MI'][condition],
                    label=condition, color=colors.get(condition, 'black'),
                    marker='^', linestyle='-', markersize=4)

    ax.set_xlabel('Time lag τ')
    ax.set_ylabel('Mutual Information I(X, U) (bits)')
    ax.set_title('Mutual Information vs Time Lag')
    ax.set_ylim(0, 6)  # Set Y-axis from 0 to 6
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Adjust layout and show
    plt.tight_layout()
    plt.show()

    return fig, axes


def run_task(tau_array, network_types, hyperparameters, task_name, n_networks=100):

    # storage: results[metric][network_type][tau_index] = list of values
    # across n_networks runs
    results = {}
    for metric in ["performance", "H", "MI"]:
        results[metric] = {}
        for nt in network_types:
            # list of lists: one list per tau, each containing n_networks values
            results[metric][nt] = [[] for _ in tau_array]

    performance_temp = []

    for network_idx in range(n_networks):

        for t in tau_array:
            print(f"For timelag {t} --------")

            for nt in network_types:

                drive = Drive(hp=hyperparameters)
                network = Kwta_RN(hp=hyperparameters)

                w, h, x_p = plasticity_phase.run(external_drive=drive, network=network, hp=hyperparameters)

                x_per = pertubation_phase.run(x=x_p, hp=hyperparameters)

                x_out = train_phase.run(w=w, h=h, x=x_per, external_drive=drive, network=network, hp=hyperparameters)

                print("x_out : ", x_out)

                performance = test_phase.run(w=w, h=h, x=x_per, w_out=x_out, external_drive=drive, network=network, hp=hyperparameters)

                eval = evaluate.Evaluator(network=network, external_drive=drive, hp=hyperparameters)
                H, MI = eval.generate_metrics(w=w, h=h)

                results["H"][nt][t].append(H)
                results["MI"][nt][t].append(MI)
                results["performance"][nt][t].append(performance)


                performance_temp.append(performance)

    print(performance_temp)


    #print(f"performance for SIP :{[float(x) for x in results['performance']['SIP']]}")
    #print(f"H for SIP :{[float(x) for x in results['H']['SIP']]}")
    #print(f"MI for SIP :{[float(x) for x in results['MI']['SIP']]}")

    #graph_results(results, tau_array)

    return results


    '''

    print("=" * 40)
    print("RESULTS SUMMARY: SIP-RN on RAND×4")
    print("=" * 40)
    print(f"Time lag tau:         {hyperparameters['tau']}")
    print(f"Perturbation pi:      {hyperparameters['pi']}")
    print(f"Performance:          {performance:.2f}%")
    print(f"Chance level:         {100 / hyperparameters['n_symbols']:.2f}%")
    print(f"Above chance by:      {performance - 100 / hyperparameters['n_symbols']:.2f}%")
    print(f"Entropy H(X):         {H:.4f} bits")
    print(f"Mutual Info I(U,X):   {MI:.4f} bits")
    
    '''


def process_results(results_raw, network_types, tau_array):
    results = {}
    for metric in ["performance", "H", "MI"]:
        results[metric] = {}
        for nt in network_types:
            results[metric][nt] = [
                np.mean(results_raw[metric][nt][tau_idx])
                for tau_idx in range(len(tau_array))
            ]
    return results



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

    #tau_array = list(range(-8,9))
    tau_array = [-1]
    print(tau_array)

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
        "tau": -1,  # time lag

        "timesteps": timesteps,

        "verbosity": 0,

        # Perturbation
        "pi": 12,

    }


    results_raw = run_task(
        tau_array=tau_array,
        network_types=["SIP"],
        task_name="RAND4",
        hyperparameters = hyperparameters,
        n_networks=1,
    )

    # process results into single arrays
    results = process_results(results_raw, ["SIP"], tau_array)


    print(f"performance for SIP :{[float(x) for x in results['performance']['SIP']]}")
    print(f"H for SIP :{[float(x) for x in results['H']['SIP']]}")
    print(f"MI for SIP :{[float(x) for x in results['MI']['SIP']]}")

    # graph results
    # expects results dictionary, with items performance, H, and MI.
    # each item should contain a dictionary for each network type, and each dictionary would contain an array of length timelags
    '''
    example
    results = {

        "H": networks1,
        "MI": networks2,
    }
    networks1 = {
        "SIP": r1,
        "SP": r2,
    }
    r1 = [H values for each timelag: 15 values]
    '''
    graph_results(results, tau_array)

