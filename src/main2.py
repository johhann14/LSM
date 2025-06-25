import numpy as np 
import matplotlib.pyplot as plt
import time
import sys
from lsm import LSM
from lif import LIF
from input_encoder import InputEncoder
from poisson_generator import PoissonSpikeGenerator
from synapse import Synapse

np.random.seed(42)

w_in = None
w_out = None
distribution = 'gaussian'
p_inh = 0.2
refractory_time = 1e-3
apply_dale = True
dt = 1e-3
T = 3
n_steps = int(T/dt)
threshold = 15e-3

# 1 = E
# 0 = I
connections_parameters = {
    #(i,j) = [C, U(use), D(time constant for depression in s), F(time constant for facilitation in s), A(scaling parameter in nA), transmission delay]
    (0,0) : [0.1,
             0.32,
             0.144,
             0.06,
             -19e-9,
             0.8e-3],

    (0,1) : [0.4,
             0.25,
             0.7,
             0.02,
             -19e-9,
             0.8e-3],

    (1,0) : [0.2,
             0.05,
             0.125,
             1.2,
             60e-9,
             0.8e-3,],

    (1,1) : [0.3,
             0.5,
             1.1,
             0.05,
             30e-9,
             1.5e-3,]

}

#github (20,5,6) et (10,5,6)
# mon test (15,3,3) (10,3,3)


#=======================LSM INSTANTATION==================
reservoir = LSM(225, 150, (15,5,3), (10,5,3), w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale)
reservoir.paramaters()
c = reservoir.input_topology
reservoir.plot_lsm_topology(reservoir.input_topology, reservoir.liquid_topology)
#=======================POISSON SPIKE GENERATOR=================
generator = PoissonSpikeGenerator(dt, T)
generator.parameters()
rate = 20
spike_inputs = generator.generate(rate)

#======================INPUT SPIKE CURRENT=======================
I = 30e-9
encoded_input = np.zeros(generator.n_steps)
encoded_input[spike_inputs] = I


#=====================SIN TO SPIKE - RATE CODING===================
T_period = 100e-3
encoder = InputEncoder(T,dt)
range_t = np.linspace(0,T,int(T/dt))
s = np.sin(range_t * 2 * np.pi / T_period)
spikes1 = encoder.encode(s)
spikes = spikes1.astype(int)
spikes = spikes * s * I

#=============================FORWARD=========================
start_time = time.perf_counter()
reservoir.forward(T,dt, spikes)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f'forward() execution time : {elapsed_time}')

#===========================PLOT INPUT==========================
encoder.plot(s, spikes1)

#==========================INPUT VOLTAGE====================
plt.figure()
random_index = np.random.choice(np.arange(0, len(reservoir.input_layer), 1), 70, False)
for i in random_index:
    plt.plot(reservoir.input_layer[i].U_trace)
    plt.axhline(threshold, linestyle='--', color='k')

plt.title("INput Neuron\'s voltage")
plt.savefig("results/input_layer_voltage.png")
plt.show()

#==========================LIQUID VOLTAGE====================
plt.figure()
random_index = np.random.choice(np.arange(0, len(reservoir.liquid_neurons), 1), 70, False)
for i in random_index:
    plt.plot(reservoir.liquid_neurons[i].U_trace)
    plt.axhline(threshold, linestyle='--', color='k')
plt.ylim(0,30e-3)
plt.title("Liquid Neuron\'s voltage")
plt.savefig("results/liquid_voltage.png")
plt.show()

#==========================Current INPUT SYNAPSE==================
fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.set_ylabel("x")
ax2.set_ylabel("u")
ax3.set_ylabel("I")
ax3.set_xlabel("t")

random_index = np.random.choice(np.arange(0, len(reservoir.input_synapses), 1), 70, False)
for i in random_index:
    ax1.plot(reservoir.input_synapses[i].x_trace)
    ax2.plot(reservoir.input_synapses[i].u_trace)
    ax3.plot(reservoir.input_synapses[i].I_trace)
plt.suptitle("Input Synapse")
plt.savefig("results/input_layer_synapses.png")
plt.show()
#==========================Current LIQUID SYNAPSE==================
fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.set_ylabel("x")
ax2.set_ylabel("u")
ax3.set_ylabel("I")
ax3.set_xlabel("t")
random_index = np.random.choice(np.arange(0, len(reservoir.liquid_synapses), 1), 70, False)
for i in random_index:
    ax1.plot(reservoir.liquid_synapses[i].x_trace)
    ax2.plot(reservoir.liquid_synapses[i].u_trace)
    ax3.plot(reservoir.liquid_synapses[i].I_trace)

fig.suptitle("Liquid Synapse")
plt.savefig("results/liquid_synapses.png")
plt.show()
#=========================LIQUID TRACE========================
plt.figure()
trial = [n.spike_trace for n in reservoir.liquid_neurons]
for neuron, t in enumerate(trial):
    plt.scatter(t, [neuron]*len(t), color='k', s=1)

plt.title("Paterns of the liquid")
plt.savefig("results/liquid_pattern.png")
plt.show()

#==============================Itot trace===================
plt.figure()
random_index = np.random.choice(np.arange(0, len(reservoir.liquid_neurons), 1), 70, False)
for i in random_index:
    plt.plot(reservoir.liquid_neurons[i].Itot_trace)
plt.show()


#=========================INPUT TRACE========================
plt.figure()
trial = [n.spike_trace for n in reservoir.input_layer]
for neuron, t in enumerate(trial):
    plt.scatter(t, [neuron]*len(t), color='k', s=1)

plt.title("Paterns of the INPUT")
plt.savefig("results/input_layer_pattern.png")
plt.show()

lif94 = reservoir.liquid_neurons[94]
plt.figure()
plt.plot(lif94.U_trace)
plt.show()

plt.figure()
plt.plot(lif94.Itot_trace)
plt.show()
prob_index = []
for s in range (len(reservoir.liquid_synapses)):
    if reservoir.liquid_synapses[s].j == 94:
        print(s)
        prob_index.append(s)
prob_index1 = []
print("ON CHERCHE DANS INPUT SUNAPSE")
"""
for s in range (len(reservoir.input_synapses)):
    if reservoir.input_synapses[s].j == 94:
        print(s)
        prob_index1.append(s)

for i in prob_index1:
    print(i)
    reservoir.input_synapses[i].plot()
"""


#62 probleme


reservoir.input_synapses[62].plot()

#step 134 lepb, x u explose 
# qui est le post neuron

ip  = reservoir.input_synapses[62].i
print(ip)

np = reservoir.input_layer[ip].plot(spike_inputs, 44)
reservoir.input_synapses[62].parameters()
print(reservoir.input_synapses[62].x_trace[15:27])
# a t = 24 ca explose , pq ?
print(reservoir.input_synapses[62].spike_trace[0:27])
jp = reservoir.input_synapses[62].j
reservoir.liquid_neurons[jp].plot(spike_inputs, 77)