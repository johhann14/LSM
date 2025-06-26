import numpy as np 
import matplotlib.pyplot as plt 
import sys
from utils import euclidean_distance, generate_neurons, probability_connection, make_input_layer, make_liquid_topology, assign_exc_inh, generate_synapses
from utils import liquid_default_parameters
from synapse import Synapse
from lif import LIF
from lif import LIF

class LSM:
    """
    LSM Framework from scratch following 'Real-Time Computing Without Stable States: A New
    Framework for Neural Computation Based on Perturbations' Maass 2002
    Based on the work of Ricardo de Azambuja and Robert Kim

    Initialisation of the reservoir's topology
    """
    def __init__(self, N_liquid, N_input, liquid_net_shape, connections_parameters, p_inh, apply_dale, dt, lbd=1.2):
        """
        Reservoir initialisation method
        N_r: number of internal units (neurons)
        N_i : number of input units
        net_shape: shape of the cortical column
        w_in:
        w_out:
        distribution:
        p_inh: probability of a neuron being inihibitory
        refractory_time:
        connections_parameters:
        apply_dale: apply Dale's principle
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

        self.liquid_topology = make_liquid_topology(connections_parameters, liquid_net_shape, self.inh_liquid, lbd)
        
        self.liquid_neurons = LIF(N=N_liquid, params=liquid_default_parameters, dt=dt)
        self.synapses = Synapse(self.liquid_topology['exc'] + self.liquid_topology['inh'], dt)
        self.W_in = np.random.normal(size=(1, N_liquid)) #inverser les dim

    def forward(self, inp=None, Ic=None):
        if inp:    
            for i in range(self.N_liquid):
                self.liquid_neurons.ie[i]+= self.W_in[0][i] * inp * Ic
        
        self.liquid_neurons.iteration()
        pre_id = self.synapses.i.astype(np.int64)
        post_id = self.synapses.j.astype(np.int64)
        spike_bool = self.liquid_neurons.spiked_before[pre_id]
        I = self.synapses.propagate(spike_bool)
        self.liquid_neurons.ie[post_id] += I

    def reset(self):
        self.liquid_neurons.reset()
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

