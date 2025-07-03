"""
File: test_topology.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-07-03

File to show the liquid's topology

References:
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
    - R. de Azambuja, F. B. Klein, S. V. Adams, M. F. Stoelen, A. Cangelosi
        Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

from lsm import LSM
from utils import connections_parameters, plot_neurons_trace, plot_liquid

p_inh           = 0.2
apply_dale      = True
dt              = 2e-3
n_steps         = 1000
T               = n_steps * dt
n_input         = 300

np.random.seed(42)
lsm = LSM(
    N_liquid=600,
    N_input=n_input,
    liquid_net_shape=(20,5,6),
    connections_parameters=connections_parameters,
    p_inh=p_inh,
    apply_dale=apply_dale,
    dt=dt,
    enable_stp=True)

plot_liquid(lsm.topology, lsm.inh_liquid)