import numpy as np
import matplotlib.pyplot as plt
class LIF:
    def __init__(self, U_init, U_reset, dt, T, R, C, threshold, Ib):
        self.U_init = U_init
        self.U_reset = U_reset
        self.threshold = threshold
        self.dt = dt
        self.R = R
        self.C = C
        self.U = U_init
        self.Ib = Ib
        self.step = 0
        self.T = T
        self.U_trace = []
        self.spike_trace = []
        self.spiked_before = False
        self.Itot = 0
        self.Itot_trace = []

    def receive_input_current(self, I):
        self.Itot += I
    

    def euler_iteration(self, I):
        if self.U > 100:
            self.prob = True
        else:
            self.prob = False
        self.U_trace.append(self.U)
        self.Itot_trace.append(self.Itot)
        if self.U > self.threshold:
            self.spiked_before = True
            self.U = self.U_reset
            self.spike_trace.append(self.step)
            self.step+=1
            self.Itot = 0
            return self.U
        
        else:
            self.spiked_before = False
            tau = self.R * self.C
            self.U += (self.dt/tau) * (-self.U + I*self.R)
            self.step+=1
            self.Itot = 0
            return self.U

    def reset(self):
        self.U= self.U_init
        self.U_trace = []
        self.spike_trace = []
        self.step = 0

    def plot(self, d, rate):
        fig, ax = plt.subplots(3,1, figsize=(8,6))
        fig.suptitle('Lapicque\'s Neuron Model With input spikes (reset mechanism)')
        ax[0].eventplot(d, color='green')
        ax[0].set_ylabel('Input Spikes')
        ax[0].set_title('Poisson Spikes with r=%d' %rate)
        
        ax[1].plot(self.U_trace)
        ax[1].axhline(self.threshold, linestyle='--', color='black', label='threshold')
        ax[1].legend()
        ax[1].set_ylabel('Neuron\'s Potential')
        
        ax[2].eventplot(self.spike_trace, color='red')
        ax[2].set_ylabel('Output Spikes')
        ax[2].set_xlabel('Time step')
        plt.savefig("results/lif_basic_implementation.png") 
        plt.show()

    def paramaters(self):
        print(f'LIF\'s parameters')
        print(f'U_init : {self.U_init}')
        print(f'U_reset : {self.U_reset}')
        print(f'threshold : {self.threshold}')
        print(f'Ib : {self.Ib}')
        print(f'dt : {self.dt}')
        print(f'T : {self.T}')
        print(f'R : {self.R}')
        print(f'C : {self.C}')
        print(f'U : {self.U}')
        print(f'step : {self.step}')



