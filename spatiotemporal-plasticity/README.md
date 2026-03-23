# Spatiotemporal Computations of an Excitable and Plastic Brain


Paper: 

**Toutounji, H. & Pipa, G. (2014).** *Spatiotemporal Computations of an Excitable and Plastic Brain: Neuronal Plasticity Leads to Noise-Robust and Noise-Constructive Computations.* PLoS Computational Biology, 10(3), e1003512. DOI: [10.1371/journal.pcbi.1003512](https://doi.org/10.1371/journal.pcbi.1003512)


## Overview

This folder contains a naive implementation exploring key results from the above paper. 

## Paper Summary

### What the paper is about

The paper addresses a core question in computational neuroscience: how do neurons in recurrent cortical networks learn to process stimuli that are extended in both space and time? The authors propose a geometric theory of how two common plasticity mechanisms, spike-timing-dependent synaptic plasticity (STDP) and intrinsic plasticity (IP), interact in recurrent networks to enable spatiotemporal computation.

The network model used is a k-Winner-Take-All (kWTA) recurrent network, analyzed as a nonautonomous dynamical system, meaning its attractor landscape shifts with the input rather than being fixed.

### Primary objectives

- Show that the combination of STDP and IP is necessary for learning useful neural representations of spatiotemporal inputs, with neither mechanism sufficient on its own
- Characterize the resulting neural code in terms of entropy and mutual information between network states and input sequences
- Demonstrate that the jointly plastic network exhibits two key properties with respect to noise: it tolerates noise (noise-robust), and in certain dynamic regimes it actually benefits from noise (noise-constructive), a form of stochastic resonance

### Key findings

- Networks trained with both STDP and IP (SIP-RNs) significantly outperform those trained with either mechanism alone on memory, prediction, and nonlinear computation tasks
- STDP alone drives the network into an input-insensitive regime, locking onto a minimal, low-entropy code; IP alone increases entropy but without synaptic structure, mutual information with the input remains low
- Together, STDP and IP produce a code that is both redundant and input-specific, the two properties identified as necessary for noise robustness
- Post-plasticity perturbation experiments reveal that the network has at least two dynamic regimes, and that larger perturbations (including noise) can push the system into the regime better suited for computation

