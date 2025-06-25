import numpy as np
import matplotlib.pyplot as plt

class Synapse:
    """
    Synapse object following Markram and Tsodysk : Short-term synaptic plasticity: http://www.scholarpedia.org/article/Short-term_synaptic_plasticity 
    Modelize the synapse and its dynamic behavior
    """
    def __init__(self, connection_infos, dt):
        self.i = connection_infos[0][0]
        self.j = connection_infos[0][1]
        self.p_connection = connection_infos[1]
        self.W_n, self.U_ds, self.D_ds, self. F_ds = connection_infos[2]
        self.delay_trans = connection_infos[3]
        self.t_connection = connection_infos[4]

        self.tau_s = 3e-3 if connection_infos[4][0]==1 else 6e-3
        self.dt = dt
        # Plasticity variables
        self.x = 1
        self.u = 0
        self.I = 0
        self.step = 0
        self.nb = 0
        #Trace
        self.x_trace = [self.x]
        self.u_trace = [self.u]
        self.I_trace = [self.I]
        self.spike_trace = []
        self.delay_steps = 1
        self.spike_buffer = [False] * (self.delay_steps + 1)
        self.buffer_index = 0
        self.bool_trace = [] 
    
    def reset(self):
        # Plasticity variables
        self.x = 1
        self.u = 0
        self.I = 0
        self.step = 0
        self.nb = 0
        #Trace
        self.x_trace = [self.x]
        self.u_trace = [self.u]
        self.I_trace = [0]
        self.spike_trace = []
        self.delay_steps = 1
        self.spike_buffer = [False] * (self.delay_steps + 1)
        self.buffer_index = 0 
    

    def propagate(self, spike_bool):

        if spike_bool:
            self.u += self.U_ds * (1-self.u)
          #  self.x += -self.u * self.x
            self.x *= (1-self.u)
            #self.I += self.W_n * self.u * self.x
            I_instant = self.W_n * self.u * self.x
        else:
            self.u *= np.exp(-self.dt/self.F_ds)
            #self.u = self.dt * ((self.U_ds -self.u)/self.F_ds)
           # self.x += ((1-self.x)/self.D_ds) * self.dt
            self.x = 1 - (1 - self.x) * np.exp(-self.dt/self.D_ds)
            #self.I -= (self.I/self.tau_s)*self.dt
            I_instant = 0
       
        self.u = np.clip(self.u, 0, 1)
        self.x = np.clip(self.x, 0, 1)

        self.spike_trace.append(spike_bool)
        self.x_trace.append(self.x)
        self.u_trace.append(self.u)
        self.I_trace.append(I_instant)

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
