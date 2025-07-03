"""
File: input_encoder.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-06-15

Class to encode a continous signal into spikes.

"""

import numpy as np 
import matplotlib.pyplot as plt
import torch
import snntorch.spikegen as spikegen

class InputEncoder():

    def __init__(self, n_steps, n_input):
        self.n_steps = n_steps
        self.n_input = n_input

    
    def rate_encoding(self, input_signal):

        # Normalisation of the input signal
        smax = np.max(input_signal)
        smin = np.min(input_signal)
        norm = (input_signal - smin ) / (smax - smin)


        data = torch.from_numpy(norm).float()
        data = data.unsqueeze(1).repeat(1,self.n_input)

        spike_train = spikegen.rate(data, time_var_input=True)

        return spike_train.detach().cpu().numpy()
    
    
    def plot(self, input_signal, encoded_signal):
        times, neurons = np.nonzero(encoded_signal)
        fig = plt.figure(figsize=(10,8))
        fig.suptitle("InputEncoder")
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        ax1.set_ylabel("Raw signal")
        ax1.plot(input_signal)
        ax2.scatter(times, neurons, s=1, alpha=0.3)     
        ax2.set_xlabel("Time step")
        ax2.set_ylabel("Input neurons index")
        # plt.savefig("results/test_lsm_uniform_input/input.png")
        plt.show()


