import numpy as np

class Synapse:
    """
    Synapse object following Markram and Tsodysk : Short-term synaptic plasticity
    Modelize the synapse and its dynamic behavio
    """
    def __init__(self, connection_infos, dt):
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

