import numpy as np 
import matplotlib.pyplot as plt
import time
import sys
from scipy.spatial import distance
from lsm import LSM
from lif import LIF
from input_encoder import InputEncoder
from poisson_generator import PoissonSpikeGenerator
from synapse import Synapse
from utils import generate_neurons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import generate_input

np.random.seed(42)
w_in = None
w_out = None
distribution = 'gaussian'
p_inh = 0.2
refractory_time = 1e-3
apply_dale = True
dt = 1e-3
T = 0.5
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
reservoir = LSM(135, 150, (15,3,3), (10,5,3), w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale)
reservoir.make_liquid_topology()
reservoir.generate_synapses(reservoir.liquid_topology)
reservoir.paramaters()
reservoir.plot_liquid(reservoir.liquid_topology)


#=====================Generate pattern0==========
rate = 10
generator = PoissonSpikeGenerator(dt,T)

input_list = []
generator_list = []
n_input = 40
for i in range(n_input):
    tmp = np.zeros(n_steps)
    spike_inputs = generator.generate(rate)
    tmp[spike_inputs] = 1
    generator_list.append(tmp)
    input_list.append(spike_inputs)



plt.figure()
for neuron, t in enumerate(input_list):
    plt.scatter(t, [neuron]*len(t), color='k', s=1)

plt.title("Patterns of the INPUT")
plt.show()

#===================Connecting input to liquid===============

sparsity = 0.1
W = (np.random.rand(40, 135) < sparsity).astype(int)
W = W * 0.005

for step in range(n_steps):
    for j in range(reservoir.N_liquid):
        for i in range(n_input):
            reservoir.liquid_neurons[j].receive_input_current(W[i,j] * generator_list[i][step])
    reservoir.STPv2()
    reservoir.update_liquid()

for n in reservoir.liquid_neurons:
    print(f'STEP: {n.step}')

plt.figure()
trial = [n.spike_trace for n in reservoir.liquid_neurons]
for neuron, t in enumerate(trial):
    plt.scatter(t, [neuron]*len(t), color='k', s=1)

plt.title("Paterns of the liquid")
plt.show()

#=====================Distance input=============

input1 = generate_input(generator, n_input=n_input, rate=20)
input2 = generate_input(generator, n_input=n_input, rate=40)

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
for neuron, t in enumerate(input1[0]):
    ax1.scatter(t, [neuron]*len(t), color='k', s=1)

for neuron, t in enumerate(input2[0]):
    ax2.scatter(t, [neuron]*len(t), color='k', s=1)
plt.suptitle("Patterns of the INPUT")
plt.show()

g1 = np.array(input1[1])
g2 = np.array(input2[1])
print(g1.shape, g2.shape)

print(g1.shape)
d= 0
for train in range(n_input):
    u = g1[train]
    v = g2[train]
    d+= distance.hamming(u,v)
d = d/n_input
print(f'distance : {d}')

s = 0
for i in range(200):
    input1 = generate_input(generator, n_input=n_input, rate=20)
    input2 = generate_input(generator, n_input=n_input, rate=20)
    g1 = np.array(input1[1])
    g2 = np.array(input2[1])
    d = 0
    for train in range(n_input):
        u = g1[train]
        v = g2[train]
        d+= distance.hamming(u,v)
    d = d/n_input
    s += d
s = s/200
print(f'distance sur 200 fois avec rate 20 et 40: {s}')

reservoir.reset_liquid()

input1 = generate_input(generator, n_input=n_input, rate=20)
input2 = generate_input(generator, n_input=n_input, rate=40)

for step in range(n_steps):
    for j in range(reservoir.N_liquid):
        for i in range(n_input):
            reservoir.liquid_neurons[j].receive_input_current(W[i,j] * input1[1][i][step])
    reservoir.STPv2()
    reservoir.update_liquid()

l1_trace = []
for neuron in reservoir.liquid_neurons:
    n = np.zeros(n_steps)
    n[neuron.spike_trace] = 1
    l1_trace.append(n)

reservoir.reset_liquid()


for step in range(n_steps):
    for j in range(reservoir.N_liquid):
        for i in range(n_input):
            reservoir.liquid_neurons[j].receive_input_current(W[i,j] * input2[1][i][step])
    reservoir.STPv2()
    reservoir.update_liquid()
    
l2_trace = []
for neuron in reservoir.liquid_neurons:
    n = np.zeros(n_steps)
    n[neuron.spike_trace] = 1
    l2_trace.append(n)

l1 = np.array(l1_trace)
l2 = np.array(l2_trace)

D = []
for t in range(n_steps):
    u = l1[:, t]
    v = l2[:, t]
    d = distance.hamming(u,v)
    D.append(d)

plt.figure()
plt.plot(D)
plt.show()

#reservoir.plot_lsm_topology(reservoir.input_topology, reservoir.liquid_topology)
#=======================POISSON SPIKE GENERATOR=================
"""
for step in range(generator.n_steps):
    for j in range(135):
        for i in range(40):
            reservoir.liquid_neurons[j].receive_input_current(list_generator[i][step]*w[i,j])
        reservoir.liquid_neurons[j].euler_iteration(reservoir.liquid_neurons[j].Itot)
    reservoir.STPv2()
#===========================PLOT INPUT==========================
lif_pif = reservoir.liquid_neurons[34]
lif_pif.plot(list_spike[0], 34)
plt.figure()
for neuron, t in enumerate(list_spike):
    plt.scatter(t, [neuron]*len(t), color='k', s=1)

plt.title("Patterns of the INPUT")
plt.savefig("results/40_poisson_input/input_pattern.png")
plt.show()



plt.figure()
random_index = np.random.choice(np.arange(0, len(reservoir.liquid_neurons), 1), 70, False)
for i in random_index:
    plt.plot(reservoir.liquid_neurons[i].Itot_trace)
plt.savefig("results/40_poisson_input/input_current_sent_to_each_neurons.png")
plt.show()


plt.figure()
random_index = np.random.choice(np.arange(0, len(reservoir.liquid_neurons), 1), 70, False)
for i in random_index:
    plt.plot(reservoir.liquid_neurons[i].U_trace)
    plt.axhline(threshold, linestyle='--', color='k')
plt.ylim(0,30e-3)
plt.title("Liquid Neuron\'s voltage")
plt.savefig("results/40_poisson_input/neurons_voltage.png")
plt.show()

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
plt.savefig("results/40_poisson_input/liquid_synapses.png")
plt.show()

plt.figure()
trial = [n.spike_trace for n in reservoir.liquid_neurons]
for neuron, t in enumerate(trial):
    plt.scatter(t, [neuron]*len(t), color='k', s=1)

plt.title("Paterns of the liquid")
plt.savefig("results/40_poisson_input/liquid_pattern.png")
plt.show()
output = np.zeros((reservoir.N_liquid, n_steps))
for i in range(len(reservoir.liquid_neurons)):
    index_spike = reservoir.liquid_neurons[i].spike_trace
    output[i, index_spike] = 1

print(output)
print(np.where(output == 1))

# fenetre 300 steps
X = np.zeros((len(reservoir.liquid_neurons), 10))
for i in range(len(reservoir.liquid_neurons)):
    for j in range(10):
        f = j * 300
        s = np.sum(output[i, f:f+300])
        X[i, j] = s

print(X)

"""
"""
list_spike1 = []
list_spike2 = []
np.random.seed(42)
reservoir1 = LSM(135, 150, (15,3,3), (10,5,3), w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale)
np.random.seed(42)
reservoir2 = LSM(135, 150, (15,3,3), (10,5,3), w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale)
pattern0 = []
pattern1 = []
pattern2 = []
for i in range(50):
    list_generator = []
    list_generator1 = []
    list_generator2 = []
    for i in range(40):
        spike_inputs = generator.generate(20)
        encoded_input = np.zeros(generator.n_steps)
        encoded_input[spike_inputs] = I
        list_spike.append(spike_inputs)
        list_generator.append(encoded_input)

        spike_inputs1 = generator.generate(30)
        encoded_input1 = np.zeros(generator.n_steps)
        encoded_input1[spike_inputs1] = I
        list_spike1.append(spike_inputs1)
        list_generator1.append(encoded_input1)
        
        spike_inputs2 = generator.generate(40)
        encoded_input2 = np.zeros(generator.n_steps)
        encoded_input2[spike_inputs2] = I
        list_spike2.append(spike_inputs2)
        list_generator2.append(encoded_input2)
    pattern0.append(list_generator)
    pattern1.append(list_generator1)
    pattern2.append(list_generator1)
print("Pattern0 data train generated")
print("Pattern1 data train generated")
print("Pattern2 data train generated")


X0 = []
X1 = []
X2 = []
for train in range(len(pattern0)):
    print(f'train0 : {train}')
    for step in range(generator.n_steps):
        for j in range(135):
            for i in range(40):
                reservoir.liquid_neurons[j].receive_input_current(pattern0[train][i][step] * w[i,j])
            reservoir.liquid_neurons[j].euler_iteration(reservoir.liquid_neurons[j].Itot)
        reservoir.STPv2()

    output = np.zeros((reservoir.N_liquid, n_steps))
    for i in range(len(reservoir.liquid_neurons)):
        index_spike = reservoir.liquid_neurons[i].spike_trace
        output[i, index_spike] = 1
    X = np.zeros((len(reservoir.liquid_neurons), 10))
    for i in range(len(reservoir.liquid_neurons)):
        for j in range(10):
            f = j * 300
            s = np.sum(output[i, f:f+300])
            X[i, j] = s
    X_flat = X.flatten()
    X0.append(X_flat)
    reservoir.reset()


    print(f'train1 : {train}')
    for step in range(generator.n_steps):
        for j in range(135):
            for i in range(40):
                reservoir1.liquid_neurons[j].receive_input_current(pattern1[train][i][step] * w[i,j])
            reservoir1.liquid_neurons[j].euler_iteration(reservoir1.liquid_neurons[j].Itot)
        reservoir1.STPv2()

    output1 = np.zeros((reservoir1.N_liquid, n_steps))
    for i in range(len(reservoir1.liquid_neurons)):
        index_spike = reservoir1.liquid_neurons[i].spike_trace
        output[i, index_spike] = 1
    X = np.zeros((len(reservoir1.liquid_neurons), 10))
    for i in range(len(reservoir1.liquid_neurons)):
        for j in range(10):
            f = j * 300
            s = np.sum(output[i, f:f+300])
            X[i, j] = s
    X_flat = X.flatten()
    X1.append(X_flat)
    reservoir1.reset()

    print(f'train2 : {train}')
    for step in range(generator.n_steps):
        for j in range(135):
            for i in range(40):
                reservoir2.liquid_neurons[j].receive_input_current(pattern2[train][i][step] * w[i,j])
            reservoir2.liquid_neurons[j].euler_iteration(reservoir2.liquid_neurons[j].Itot)
        reservoir2.STPv2()

    output2 = np.zeros((reservoir2.N_liquid, n_steps))
    for i in range(len(reservoir2.liquid_neurons)):
        index_spike = reservoir2.liquid_neurons[i].spike_trace
        output[i, index_spike] = 1
    X = np.zeros((len(reservoir2.liquid_neurons), 10))
    for i in range(len(reservoir2.liquid_neurons)):
        for j in range(10):
            f = j * 300
            s = np.sum(output[i, f:f+300])
            X[i, j] = s
    X_flat = X.flatten()
    X2.append(X_flat)
    reservoir2.reset()



X0_data = np.array(X0)
X1_data = np.array(X1)
X2_data = np.array(X1)
np.save("x0_data.npy", X0_data)
np.save("x1_data.npy", X1_data)
np.save("x2_data.npy", X2_data)
print("FINIS")
"""