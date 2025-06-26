import numpy as np
import matplotlib.pyplot as plt

class Synapse:
    """
    Synapse object following Markram and Tsodysk : Short-term synaptic plasticity: http://www.scholarpedia.org/article/Short-term_synaptic_plasticity 
    Modelize the synapse and its dynamic behavior
    """
    def __init__(self, list_connection_infos, dt):
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
        self.x = np.ones(self.N)
        self.u = np.zeros(self.N)
        for s in range(self.N):
            self.i[s] = list_connection_infos[s][0][0]
            self.j[s] = list_connection_infos[s][0][1]
            self.p_connection[s] = list_connection_infos[s][1]
            self.W_n[s], self.U_ds[s], self.D_ds[s], self.F_ds[s] = list_connection_infos[s][2]
            self.delay_trans[s] = list_connection_infos[s][3]
            self.tau_s[s] = 3e-3 if self.i[s]==1 else 6e-3
        
        self.dt = dt
        self.step = 0
    
    def reset(self):
        # Plasticity variables
        self.x = np.ones(self.N)
        self.u = np.zeros(self.N)
        self.step = 0
    
    def propagate(self, spike_bool):

        #all neurons that spiked en regardant spiie_trace de s.i:

        I_instant = np.zeros(self.N)
        self.u[spike_bool]+= self.U_ds[spike_bool] * (1- self.u[spike_bool])
        self.x[spike_bool] *= (1-self.u[spike_bool])
        I_instant[spike_bool] = self.W_n[spike_bool] * self.u[spike_bool] *self.x[spike_bool]
            
        self.u[~spike_bool] *= np.exp(-self.dt/self.F_ds[~spike_bool])    
        self.x[~spike_bool] *= 1 - (1 - self.x[~spike_bool]) * np.exp(-self.dt/self.D_ds[~spike_bool])

        return I_instant
    
    def plot(self):
        """
        if self.t_connection[0] == 1: # EXC
            col='red'
        else:
            col='blue'
        """
        fig = plt.figure(figsize=(8,6))
        fig.suptitle("Synapse\'s parameters")
        ax1 = fig.add_subplot(3,1,1)
        ax1.plot(self.x_trace, c='red')
        ax1.set_ylabel('x')
        ax1.set_xlim(0, 300)
        ax2 = fig.add_subplot(3,1,2)
        ax2.plot(self.u_trace, c='blue')
        ax2.set_ylabel('u')
        ax2.set_xlim(0,300)
        ax3 = fig.add_subplot(3,1,3)
        ax3.plot(self.I_trace, c='green')
        ax3.set_ylabel('I')
        ax3.set_xlabel('t')
        ax3.set_xlim(0,300)
        plt.show()

        


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
