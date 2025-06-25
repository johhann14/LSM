import numpy as np 
import matplotlib.pyplot as plt

from lif import LIF
class InputEncoder():

    def __init__(self,T, dt):
        self.T = T
        self.dt = dt


    def encode(self, input_signal):
        n = int(self.T/self.dt)
        smax = np.max(input_signal)
        smin = np.min(input_signal)
        norm  = (input_signal - smin) / (smax - smin)
        spikes = np.random.rand(n) < norm
        return spikes
    
    def plot(self, input_signal, encoded_input):
        fig = plt.figure(figsize=(10,8))
        fig.suptitle("InputEncoder")
        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(input_signal)
        ax1.set_title("Input signal")
        ax2 = fig.add_subplot(2,1,2)
        ax2.eventplot(np.where(encoded_input)[0], color='k')
        ax2.set_title("Encoded input signal as spikes")
        ax2.set_xlabel("time step t")
        plt.show()



