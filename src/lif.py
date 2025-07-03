"""
File: lif.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-06-15

Class implementing a group of LIF neurons.

References:
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
    
    - R. de Azambuja, F. B. Klein, S. V. Adams, M. F. Stoelen, A. Cangelosi
        Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller.
"""

import numpy as np
import matplotlib.pyplot as plt

class LIF:
    """
    Group of LIF neurons initialisation method
    """

    def __init__(self, N, params, inh_repartition, dt):
        """
        N: number of neurons
        params: parameters values for the liquid 
        inh_repartition: ndarray of bool that represents the inhibitory repartition among the group of neurons
        dt: time step used for the simulation
        U_init: ndarray of neurons initial potential
        U_reset : ndarray of neurons reset potential
        U_threshold: threshold potential
        taum: membrane time constant
        Cm: membrane capacitance
        tau_se: synapse time constant (exc.)
        tau_si: synapse time constant (inh) PROBLEME SI ON FAIT EVOLUER L'INTENSITE TOTALE PR CHAQUE NEURONE
        U: ndarray of neurons potential
        i_offset : ndarray of noisy offset initial currents
        i_noise_mu: loc for gaussian input current noise
        i_noise_scale: scale for gaussian input current noise
        t_delay_e: transmission delay (exc.)
        t_delay_i: transmission delay (inh.)
        refrac_period_e: refractory period (exc.)
        refrac_period_i: refractory period (inh.)
        t_ref : refractory counter
        """

        self.N = N
        self.U_init = np.random.uniform(params['U_initial'][0], params['U_initial'][1], size=N)
        self.U_reset = np.random.uniform(params['U_reset'][0], params['U_reset'][1], size=N)
        self.U_threshold = params['U_threshold']
        self.taum = params['taum']
        self.Cm = params['Cm']
        self.tau_se = params['tau_se']
        self.tau_si = params['tau_si']
        self.U = self.U_init[:]
        self.i_offset = np.random.uniform(params['i_offset'][0], params['i_offset'][1], size=N)
        self.i_noise_mu = params['i_noise_mu']
        self.i_noise_scale = params['i_noise_scale']
        self.t_delay_e = params['transmission_delay_e']
        self.t_delay_i = params['transmission_delay_i']
        self.refrac_period_e = params['refrac_period_e']
        self.refrac_period_i = params['refrac_period_i']
        self.inh_repartition = inh_repartition

        # refractory counter
        self.t_ref = np.zeros(N)

        self.dt = dt
        self.ii = np.zeros(N)
        self.ie = np.zeros(N)

        self.step = 0
        
        self.spiked_before = np.array([False]*N)
        
        # at the end of the simulation (T, n_liquid)
        self.U_trace = []
        self.spike_trace = []  
        self.ii_trace = []
        self.ie_trace = []

        self.spiked_before = np.array([False]*N)

    def iteration(self, noise_bool):
        self.spiked_before[:] = False

        #update the refractory counter
        self.t_ref= np.maximum(self.t_ref - self.dt, 0)

        # array of bool, True if spiked else false
        index_spiked = self.U > self.U_threshold

        # Dealing with the neurons who spiked
        if index_spiked.any():

            # Reinitialise their membrane's potential
            self.U[index_spiked] = self.U_reset[index_spiked]
            self.spiked_before[index_spiked] = True
            
            # cut synaptic current
            self.ii[index_spiked] = 0
            self.ie[index_spiked] = 0

            # reinitialize the refractorid period
            if self.inh_repartition is None:
                self.t_ref[index_spiked] = self.refrac_period_e
            else:
                self.t_ref[index_spiked] = np.where(self.inh_repartition[index_spiked], self.refrac_period_i, self.refrac_period_e)


        # for all the neurons, decay of the membrane current for both inh. and exc. current
        self.ii *= np.exp(-self.dt / self.tau_si)
        self.ie *= np.exp(-self.dt/self.tau_se)

        # array of bool, True if neuron can integrate else False
        can_integrate = self.t_ref <=0
        
        i_noise = np.random.normal(loc=self.i_noise_mu, scale=self.i_noise_scale, size=self.N)
        
        if noise_bool:
            #if we allow noise into the liquid 
            I_syn = (self.ii + self.ie + self.i_offset + i_noise) / self.Cm
        else:
            # no noise into the lisquid
            I_syn = (self.ii + self.ie ) / self.Cm

        leak = (0 - self.U) / self.taum # u_rest instead of 0?

        # potentiel integration for the neurons who can
        self.U[can_integrate] += (I_syn[can_integrate] + leak[can_integrate]) * self.dt
        
        # keeping trace of everything
        self.U_trace.append(self.U.copy())
        self.spike_trace.append(self.spiked_before.copy())
        self.ii_trace.append(self.ii.copy())
        self.ie_trace.append(self.ie.copy())
        
        self.step+=1


    def reset(self):
        """
        Reset the group of neurons
        """
        self.U = self.U_init.copy()
        self.ii = np.zeros(self.N)
        self.ie = np.zeros(self.N)
        self.t_ref = np.zeros(self.N)
        self.step = 0
        self.U_trace = []
        self.ii_trace = []
        self.ie_trace = []
        self.spiked_before = np.array([False]*self.N)
        self.spike_trace = []