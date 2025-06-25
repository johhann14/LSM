import numpy as np
import matplotlib.pyplot as plt
class LIF:
    def __init__(self, params, dt):
        self.U_init = np.random.uniform(params['U_initial'][0], params['U_initial'][1])
        self.U_reset = np.random.uniform(params['U_reset'][0], params['U_reset'][1])
        self.U_threshold = params['U_threshold']
        self.taum = params['taum']
        self.Cm = params['Cm']
        self.tau_se = params['tau_se']
        self.tau_si = params['tau_si']
        self.U = self.U_init
        self.i_offset = np.random.uniform(params['i_offset'][0], params['i_offset'][1])
        self.i_noise_mu = params['i_noise_mu']
        self.i_noise_scale = params['i_noise_scale']
        self.t_delay_e = params['transmission_delay_e']
        self.t_delay_i = params['transmission_delay_i']

        self.dt = dt
        self.ii = 0
        self.ie = 0

        self.step = 0
        self.itot = 0

        self.U_trace = []
        self.spike_trace = []
        self.itot_trace = []
        self.a_trace = []
        self.b_trace = []
        self.ii_trace = []
        self.ie_trace = []

        self.spiked_before = False

    def receive_input_current(self, I):
        self.Itot += I
    
    def iteration(self):
        
        if self.U > self.U_threshold:
            self.spike_trace.append(True)
            self.spiked_before = True
            self.U = self.U_reset
            self.ie = 0
            self.ii = 0

        else:
            self.spike_trace.append(False)
            self.spiked_before = False
            i_noise = np.random.normal(loc=self.i_noise_mu, scale=self.i_noise_scale)
            self.ie *= np.exp(-self.dt / self.tau_se)
            self.ii *= np.exp(-self.dt/self.tau_si)
            a = (self.ii + self.ie + self.i_offset + i_noise)/self.Cm
            b = (self.U_init - self.U)/self.taum
            self.a_trace.append(a)
            self.b_trace.append(b)
            self.U += self.dt * (a + b)

        self.U_trace.append(self.U)
        self.step+=1
        #self.ie += self.dt * (-self.ie/self.tau_se)
       # self.ii += self.dt * (-self.ii/self.tau_si)
        self.ie_trace.append(self.ie)
        self.ii_trace.append(self.ii)
        

    def reset(self):
        self.U = self.U_init
        self.step = 0
        self.itot = 0

        self.U_trace = []
        self.itot_trace = []
        self.spike_trace = []
        self.spiked_before = False
        self.ii = 0
        self.ie = 0


        self.a_trace = []
        self.b_trace = []
        self.ii_trace = []
        self.ie_trace = []

        self.spiked_before = False


    def plot(self, d):
        fig, ax = plt.subplots(4,1, figsize=(8,6))
        fig.suptitle('Lapicque\'s Neuron Model With input spikes (reset mechanism)')
        ax[0].eventplot(d, color='green')
        ax[0].set_ylabel('Input Spikes')
        ax[0].set_title('Poisson Spikes')
        
        ax[1].plot(self.U_trace)
        ax[1].axhline(self.threshold, linestyle='--', color='black', label='threshold')
        ax[1].legend()
        ax[1].set_ylabel('Neuron\'s Potential')
        
        ax[2].eventplot(self.spike_trace, color='red')
        ax[2].set_ylabel('Output Spikes')
        ax[2].set_xlabel('Time step')
        ax[3].plot(self.Itot_trace, color='red')
        ax[3].set_ylabel('Itot trace')
        ax[3].set_xlabel('Time step')
        plt.savefig("results/lif_basic_implementation.png") 
        plt.show()



    def paramaters(self):
        print(f'LIF\'s parameters')
        print(f'U_init : {self.U_init}')
        print(f'U_reset : {self.U_reset}')
        print(f'threshold : {self.U_threshold}')
        print(f'dt : {self.dt}')
        print(f'Cm : {self.Cm}')
        print(f'U : {self.U}')
        print(f'step : {self.step}')



