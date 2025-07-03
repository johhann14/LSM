"""
File: synapse.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-06-15

Class implementing Short-Term synaptic plasticity.

References:
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
    
    - R. de Azambuja, F. B. Klein, S. V. Adams, M. F. Stoelen, A. Cangelosi
        Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller.
"""

import numpy as np
import matplotlib.pyplot as plt

class Synapse:
    """
    Synapse object following Markram and Tsodysk : Short-term synaptic plasticity: http://www.scholarpedia.org/article/Short-term_synaptic_plasticity 
    
    Modelize the synapse and its dynamic behavior

    """
    def __init__(self, list_connection_infos, dt, enable_stp=True):
        """
        N: Number of synapses
        i: index of the presynaptic neuron
        j: index of the postsynaptic neuron
        p_connection: ndarray of the synapses connections probablities
        W_n: ndarray of scale 
        U_ds: ndarray of use
        D_ds: ndarray of depression time constant
        F_ds: ndarray of facilitation time constant
        delay_trans: ndarray of transmission delay
        tau_s: ndarray of synapse time constant
        t_connection: ndarray of connection type
        x: ndarray of variables reprentating STD effect
        u: ndarray of variables reprentating STF effect
        enable_stp: bool indicating if we use STP or no
        dt: time step used for the simulation
        step: 
        """

        self.N = len(list_connection_infos)
        self.i = np.zeros(self.N)
        self.j = np.zeros(self.N)
        self.p_connection = np.zeros(self.N)
        self.W_n = np.zeros(self.N)
        self.U_ds = np.zeros(self.N)
        self.D_ds = np.zeros(self.N)
        self.F_ds = np.zeros(self.N)
        self.delay_trans = np.zeros(self.N)
        self.tau_s = np.zeros(self.N)
        self.t_connection = np.zeros(self.N)
        self.x = np.ones(self.N)
        self.u = np.zeros(self.N)
        self.enable_stp = enable_stp
        for s in range(self.N):
            self.i[s] = list_connection_infos[s][0][0]
            self.j[s] = list_connection_infos[s][0][1]
            self.p_connection[s] = list_connection_infos[s][1]
            self.W_n[s], self.U_ds[s], self.D_ds[s], self.F_ds[s] = list_connection_infos[s][2]
            self.delay_trans[s] = list_connection_infos[s][3]
            self.t_connection[s] = list_connection_infos[s][4][0] # here only taking i from t_connection = (i,j) / 1 is E and 0 is I
            self.tau_s[s] = 3e-3 if self.t_connection[s] == 1 else 6e-3
        self.dt = dt
        self.step = 0
        self.x_trace = []
        self.u_trace = []
        self.I_trace = []
    
    def reset(self):
        # Plasticity variables
        self.x = np.ones(self.N)
        self.u = np.zeros(self.N)
        self.step = 0
        self.x_trace = []
        self.u_trace = []
        self.I_trace = []
    
    def update_stp(self, spike_mask):
        """
        Update synaptic facilitation and depression variables according to the Tsodysk-Markram STP model.
        Then, compute the instantenous synaptic current for each synapse

        Params:
        
        spike_mask: ndarray of bool indicating which presynaptic neuron fired at the previous step

        """
        I_instant = np.zeros(self.N)

        # Update of depression and facilitation variables for the synapses whom their presynaptic neuron spiked at the previous time
        self.u[spike_mask]+= self.U_ds[spike_mask] * (1- self.u[spike_mask])
        self.x[spike_mask] *= (1-self.u[spike_mask])
            
        # Update of depression and facilitation variables for the synapses whom their presynaptic neuron did not spike at the previous time
        self.u[~spike_mask] *= np.exp(-self.dt/self.F_ds[~spike_mask])    
        self.x[~spike_mask] = 1 - (1 - self.x[~spike_mask]) * np.exp(-self.dt/self.D_ds[~spike_mask])
        
        # compute the instantenous synaptic current for each synapse
        I_instant[spike_mask] = self.W_n[spike_mask] * self.u[spike_mask] *self.x[spike_mask]
        
        # if we disable stp, compute the instantenous synaptic current 
        if not self.enable_stp:
            I_instant = self.W_n

        # keeping trace of everything
        self.x_trace.append(self.x.copy())
        self.u_trace.append(self.u.copy())
        self.I_trace.append(I_instant.copy())

        return I_instant
    


    def plot(self, N):
        """
        Plot the depression, facilitation variables and
        the delivered current of each synapse through time.

        Params:

        N: how many synapses we want to plot (choosed randomly)
        """
        fig = plt.figure(figsize=(8,6))
        fig.suptitle("Synapse\'s parameters")
        x_trace = np.array(self.x_trace)
        u_trace = np.array(self.u_trace)
        I_trace = np.array(self.I_trace)
        index = np.random.choice(np.arange(0, self.N), N, False)
        ax1 = fig.add_subplot(3,1,1)
        ax2 = fig.add_subplot(3,1,2)
        ax3 = fig.add_subplot(3,1,3)
        print(x_trace.shape, u_trace.shape, I_trace.shape)
        for s in index:
            ax1.plot(x_trace[:, s])
            ax2.plot(u_trace[:, s])
            ax3.plot(I_trace[:, s])
        ax1.set_ylabel('x')
        ax2.set_ylabel('u')
        ax3.set_ylabel('I')
        ax3.set_xlabel('t')
        # plt.savefig("results/test_lsm_uniform_input/synapses.png")
        plt.show()

        

    """
    def parameters(self):
        print(f'\n----------------------------------------------\n')
        print(f'Synapse\'s parameters :')
        print(f'\t Neuron i : {self.i}')
        print(f'\t Neuron_j : {self.j}')
        print(f'\t p_connection  : {self.p_connection}')
        print(f'\t W_n: {self.W_n}')
        print(f'\t U_ds : {self.U_ds}')
        print(f'\t D_ds : {self.D_ds}')
        print(f'\t F_ds : {self.F_ds}')
        print(f'\t delay_trans : {self.delay_trans}')
        print(f'\t t_connection : {self.t_connection}')
        print(f'\t tau_s : {self.tau_s}')
        print(f'\t dt : {self.dt}')
        print(f'\t x : {self.x}')
        print(f'\t u : {self.u}')
        print(f'\t I : {self.I}')
        print(f'\t steps : {self.step}')
        print(f'\t nb : {self.nb}')
        print(f'\n----------------------------------------------\n')
    """