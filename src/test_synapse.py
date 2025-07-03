"""
File: test_synapse.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-07-03

File to plot the synapse model with random presynaptic neurons spikes

References:
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
    - R. de Azambuja, F. B. Klein, S. V. Adams, M. F. Stoelen, A. Cangelosi
        Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller.
"""
import numpy as np
import matplotlib.pyplot as plt

from synapse import Synapse
from utils import assign_exc_inh, make_liquid_topology, connections_parameters
from poisson_generator import PoissonSpikeGenerator

apply_dale = True
N_liquid = 1
p_inh = 0.2
inh_repartition = assign_exc_inh(N=N_liquid, apply_dale=apply_dale, p_inh=p_inh)[0]
Ic = 80e-9
dt = 2e-3
T = 0.5
n_steps = int(T/dt)
rate = 30
generator = PoissonSpikeGenerator(dt=dt, T=T)
spikes = generator.generate(rate=rate)
input_spikes = np.zeros(n_steps)
input_spikes[spikes] = 1

topology = make_liquid_topology(
    connections_parameters=connections_parameters,
    net_shape=(20,5,6),
    inh_liquid=inh_repartition,
    lbd=1.2
)

synapses = Synapse(list_connection_infos=topology['exc'] + topology['inh'], dt=dt, enable_stp=True)


# forward
for step in range(n_steps):
    b = False
    if input_spikes[step] == 1:
        b = True
    spike_mask = np.array([b]*synapses.N)
    synapses.update_stp(spike_mask=spike_mask)



# Plot
n_synapses_to_show = 5
fig = plt.figure(figsize=(8,6))
fig.suptitle("Synapse\'s parameters")
x_trace = np.array(synapses.x_trace)
u_trace = np.array(synapses.u_trace)
I_trace = np.array(synapses.I_trace)
index = np.random.choice(np.arange(0, synapses.N), n_synapses_to_show, False)
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)
ax1.set_xlim(-1, n_steps+1)
ax2.set_xlim(-1, n_steps+1)
ax3.set_xlim(-1, n_steps+1)
ax4.set_xlim(-1, n_steps+1)
ax1.eventplot(spikes, colors='k')
for s in index:
    ax2.plot(x_trace[:, s])
    ax3.plot(u_trace[:, s])
    ax4.plot(I_trace[:, s])
ax1.set_ylabel('presynaptic neuron\'s spikes')
ax2.set_ylabel('x')
ax3.set_ylabel('u')
ax4.set_ylabel('I')
ax4.set_xlabel('t')
#plt.savefig("results/test_synapse.png")
plt.show()
