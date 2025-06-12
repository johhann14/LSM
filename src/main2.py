import numpy as np 
import matplotlib.pyplot as plt
import time
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
             2.8e-9,
             0.8],

    (0,1) : [0.4,
             0.25,
             0.7,
             0.02,
             3.0e-9,
             0.8],

    (1,0) : [0.2,
             0.05,
             0.125,
             1.2,
             1.6e-9,
             0.8,],

    (1,1) : [0.3,
             0.5,
             1.1,
             0.05,
             1.2e-9,
             1.5,]

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
I = 10e-9
encoded_input = np.zeros(generator.n_steps)
encoded_input[spike_inputs] = I


#=====================SIN TO SPIKE - RATE CODING===================
T_period = 100e-3
encoder = InputEncoder(T,dt)
range_t = np.linspace(0,T,int(T/dt))
s = np.sin(range_t * 2 * np.pi / T_period)
spikes1 = encoder.encode(s)
spikes = spikes1.astype(int)
spikes = spikes * s * 5
print(spikes[:50]) 

#=============================FORWARD=========================
start_time = time.perf_counter()
reservoir.forward(T,dt, spikes)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f'forward() execution time : {elapsed_time}')

#===========================PLOT INPUT==========================
encoder.plot(s, spikes1)

#==========================LIQUID VOLTAGE====================
plt.figure()
random_index = np.random.choice(np.arange(0, len(reservoir.liquid_neurons), 1), 70, False)
for i in random_index:
    plt.plot(reservoir.liquid_neurons[i].U_trace)
    plt.axhline(threshold, linestyle='--', color='k')

plt.title("Liquid Neuron\'s voltage")
plt.show()

#=========================LIQUID TRACE========================
plt.figure()
trial = [n.spike_trace for n in reservoir.liquid_neurons]
for neuron, t in enumerate(trial):
    plt.scatter(t, [neuron]*len(t), color='k', s=1)

plt.title("Paterns of the liquid")
plt.show()