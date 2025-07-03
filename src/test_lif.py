"""
File: test_lif.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-07-03

File to plot the LIF model with a random input spikes

References:
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
    - R. de Azambuja, F. B. Klein, S. V. Adams, M. F. Stoelen, A. Cangelosi
        Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller.
"""
import numpy as np
import matplotlib.pyplot as plt

from lif import LIF
from utils import plot_neurons_trace, liquid_default_parameters, assign_exc_inh
from poisson_generator import PoissonSpikeGenerator

apply_dale = True
N_liquid = 1
p_inh = 0.2
inh_repartition = assign_exc_inh(N=N_liquid, apply_dale=apply_dale, p_inh=p_inh)[0]
Ic = 80e-9
dt = 2e-3
T = 0.5
n_steps = int(T/dt)
rate = 100
generator = PoissonSpikeGenerator(dt=dt, T=T)
spikes = generator.generate(rate=rate)
input_spikes = np.zeros(n_steps)
input_spikes[spikes] = 1

neurons = LIF(N=N_liquid, params=liquid_default_parameters, inh_repartition=inh_repartition, dt=dt)

# forward
for steps in range(n_steps):
    neurons.ie+= Ic * input_spikes[steps]
    neurons.iteration(noise_bool=False)


#Plot
U_trace = np.array(neurons.U_trace.copy())
spike_trace = np.array(neurons.spike_trace.copy())
fig = plt.figure(figsize=(12,10))
fig.suptitle("LIF test")
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.set_xlim(-1, n_steps+1)
ax2.set_xlim(-1, n_steps+1)
ax3.set_xlim(-1, n_steps+1)
ax1.set_title("input spikes")
ax2.set_title("Neurons potential")
ax3.set_title("output spikes")

ax1.eventplot(spikes, colors='k')
ax2.axhline(liquid_default_parameters["U_threshold"], c='k', linestyle='--')
for n in range(N_liquid):
    ax2.plot(U_trace[:, n])
index_spikes = np.where(spike_trace==True)[0]
ax3.eventplot(index_spikes, colors='red')
#plt.savefig("results/lif_test.png")
plt.show()
