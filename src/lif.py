import numpy as np
import matplotlib.pyplot as plt

class LIF:

    def __init__(self, N, params, dt):
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


        self.dt = dt
        self.ii = np.zeros(N)
        self.ie = np.zeros(N)

        self.step = 0
        self.itot = 0

        self.U_trace = []
        self.spike_trace = []
        self.itot_trace = []
        self.a_trace = []
        self.b_trace = []
        self.ii_trace = []
        self.ie_trace = []

        self.spiked_before = np.array([False]*N)

    def iteration(self):

        index_spiked = self.U > self.U_threshold # bool array
        # for the neurons who spiked
        self.U[index_spiked] = self.U_reset[index_spiked]
        self.spiked_before[index_spiked] = True
        self.ii[index_spiked] = 0
        self.ie[index_spiked] = 0


        # for the neurons who not spiked

        self.spiked_before[~index_spiked] = False

        self.ii[~index_spiked] *= np.exp(-self.dt / self.tau_si)
        self.ie[~index_spiked] *= np.exp(-self.dt/self.tau_se)

        i_noise = np.random.normal(loc=self.i_noise_mu, scale=self.i_noise_scale, size=self.N)
        i_tot = (self.ii[~index_spiked] + self.ie[~index_spiked] + self.i_offset[~index_spiked] + i_noise[~index_spiked]) / self.Cm
        b = (self.U_init[~index_spiked] - self.U[~index_spiked]) / self.taum
        self.U[~index_spiked] += (i_tot + b) * self.dt
        self.U_trace.append(self.U.copy()) #pq self.U[:] marche pas
        self.spike_trace.append(self.spiked_before.copy())
        self.ii_trace.append(self.ii.copy())
        self.ie_trace.append(self.ie.copy())
        self.step+=1


    def reset(self):
        self.U = self.U_init.copy()
        self.step = 0
        self.U_trace = []
        self.ii_trace = []
        self.ie_trace = []
        self.spiked_before = np.array([False]*self.N)
        self.spike_trace = []