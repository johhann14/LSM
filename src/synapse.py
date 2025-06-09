import numpy as np

class Synapse:
    """
    Synapse object following Markram and Tsodysk : Short-term synaptic plasticity: http://www.scholarpedia.org/article/Short-term_synaptic_plasticity 
    Modelize the synapse and its dynamic behavior
    """
    def __init__(self, connection_infos, dt):
        print(f'Creation of the Synapase : Begin...')
        self.connection_infos = connection_infos
        self.i = connection_infos[0][0]
        self.j = connection_infos[0][1]
        self.p_connection = connection_infos[1]
        self.W_n, self.U_ds, self.D_ds, self. F_ds = connection_infos[2]
        self.delay_trans = connection_infos[3]
        self.t_connection = connection_infos[4]
        self.tau_s = 3e-3 if connection_infos[4]==1 else 6e-3
        self.dt = dt
        
        # Plasticity variables
        self.x = 0
        self.u = 0
        self.I = 0
        print(f'Creation of the Synapase : Done!')
    
    def propagate(self, spike_bool, t=None):
        
        if spike_bool:
            self.u += self.U_ds * (1-self.u)
            self.I += self.W_n * self.u * self.x
            self.x += -self.u * self.x
        else:
            self.u += self.dt * (-self.u/self.D_ds)
            self.x += (1-self.x)*self.D_ds * self.dt
            self.I += (-self.I/self.tau_s)*self.dt
        return self.I
    

    #a un instant t on doit calculer I_tot que j recoit et on mets a jour le neuron j selon LIF avec I_tot
    def parameters(self):
        print(f'\n----------------------------------------------\n')
        print(f'Synapse\'s parameters :')
        print(f'\t Connection infos : {self.connection_infos}')
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
        print(f'\n----------------------------------------------\n')
