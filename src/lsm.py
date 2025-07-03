"""
File: lsm.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-06-15

Class implementing a Liquid State Machine.

References:
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
    
    - R. de Azambuja, F. B. Klein, S. V. Adams, M. F. Stoelen, A. Cangelosi
        Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller.
"""

import numpy as np 
import matplotlib.pyplot as plt 
import sys

from utils import make_liquid_topology, assign_exc_inh
from utils import liquid_default_parameters

from synapse import Synapse
from lif import LIF

class LSM:
    """
    LSM Framework from scratch following 'Real-Time Computing Without Stable States: A New
    Framework for Neural Computation Based on Perturbations' Maass 2002
    Based on the work of Ricardo de Azambuja and Robert Kim

    Initialisation of the reservoir's topology
    """
    def __init__(self, N_liquid, N_input, liquid_net_shape, connections_parameters, p_inh, apply_dale, dt, enable_stp=True, lbd=1.2):
        """
        Reservoir initialisation method

        N_liquid: number of liquid neurons
        N_input: number of input layer neuron
        liquid_net_shape: shape of the cortical column
        connections_parameters: dictionnary to build the topology of the liquid
        p_inh: probability of a neuron being inihibitory
        apply_dale: apply Dale's principle
        dt: time step used for the simulation
        enable_stp: bool to indicate if we use STP or no
        lbd:
        """

        self.N_liquid = N_liquid  
        self.N_input = N_input                                                        
        self.liquid_net_shape = liquid_net_shape

        self.connections_parameters = connections_parameters
        self.p_inh = p_inh
        self.dt = dt
        self.lbd = lbd
        self.apply_dale = apply_dale
        self.inh_liquid, self.exc_liquid, self.n_inh_liquid, self.n_exc_liquid = assign_exc_inh(N_liquid, apply_dale, p_inh)

        self.topology = make_liquid_topology(connections_parameters, liquid_net_shape, self.inh_liquid, lbd)
        
        self.neurons = LIF(N=N_liquid, params=liquid_default_parameters, inh_repartition=self.inh_liquid, dt=dt)
        self.synapses = Synapse(self.topology['exc'] + self.topology['inh'], dt, enable_stp=enable_stp)
        self.W_in = np.random.normal(size=(N_liquid, N_input)) #inverser les dim
    
    
    def step(self, inp, Ic, noise_bool):
        """
        Advance the LSM by one time step

        Pipeline: Looping
            - Update short-term plasticity on all synapses
            - Inject the current from the synapses to the postsynaptic neurons
            - Inject the current from the input layer (ie. from the encoded input signal) to the liquid's neurons
            - Perfrom one iteration of the neurons of the liquid
        """

        # array of containing the index of all the presynaptic neuron
        pre_id = self.synapses.i.astype(np.int64)

        # array of containing the index of all the presynaptic neuron
        post_id = self.synapses.j.astype(np.int64)

        # mask indicating which presynaptic neuron spiked
        spike_mask = self.neurons.spiked_before[pre_id]
        
        # compute the synaptic current
        I = self.synapses.update_stp(spike_mask)
        
        # injecting the synaptic current into the postsynaptic neurons
        inh = I<0
        exc = I>0
        self.neurons.ii[post_id[inh]] += I[inh]
        self.neurons.ie[post_id[exc]] += I[exc]

        # injecting the current from the input layer to the liquid neurons
        W_pos = np.maximum(self.W_in, 0)
        W_neg = -np.minimum(self.W_in, 0)
        I_exc = W_pos @ inp
        I_inh = W_neg @ inp

        self.neurons.ie+= I_exc * Ic
        self.neurons.ii+= I_inh * Ic
        #self.liquid_neurons.ie+= (self.W_in @inp) * Ic
        
        # update the neurons of the liquid         
        self.neurons.iteration(noise_bool=noise_bool)
        #faut mettre dans ie et ii selon la synapse

    def reset(self):
        self.neurons.reset()
        self.synapses.reset()

    def paramaters(self):
        print(f'\n----------------------------------------------\n')
        print(f'Model\'s parameters :')
        print(f'\t N_r : {self.N_liquid}')
        print(f'\t N_i : {self.N_input}')
        print(f'\t p_inh : {self.p_inh}')
        print(f'\t lbd : {self.lbd}')
        print(f'\t connections_parameters : {self.connections_parameters}')
        print(f'\t apply_dale : {self.apply_dale}')
        print(f'\t n_inh : {self.n_inh_liquid}')
        print(f'\t n_exc : {self.n_exc_liquid}')
        print(f'\n----------------------------------------------\n')

