import numpy as np
import matplotlib.pyplot as plt
import copy

from lsm import LSM
from lif import LIF
from utils import connections_parameters, plot_liquid, liquid_default_parameters, plot_neurons_trace, trajectory_distance, maass_distance
p_inh = 0.2
apply_dale = True
dt = 2e-3
T = 0.5
n_steps = int(T/dt)

np.random.seed(42)
lsm = LSM(N_liquid=600, N_input=300, liquid_net_shape=(20,5,6), connections_parameters=connections_parameters, p_inh=p_inh, apply_dale=apply_dale, dt=dt)
#plot_liquid(lsm.liquid_topology, lsm.inh_liquid)
inp1 = (np.random.rand(n_steps) < 0.1).astype(int)
inp2 = (np.random.rand(n_steps) < 0.9).astype(int)

inp3 = (np.random.rand(n_steps) < 0.3).astype(int)
inp4 = (np.random.rand(n_steps) < 0.3).astype(int)

inp5 = (np.random.rand(n_steps) < 0.3).astype(int)
inp6 = (np.random.rand(n_steps) < 0.8).astype(int)

inp7 = (np.random.rand(n_steps) < 0.3).astype(int)
inp8 = inp7[:]
inp8[[1,10,37,100,89, 30, 23, 144, 233, 244, 38, 32]] = 1
Ic = 70e-9
print(f'forward()')
for step in range(n_steps):
    for i in range(len(lsm.liquid_neurons)):
        lsm.liquid_neurons[i].ie+= lsm.W_in[0][i] * inp1[step] * Ic
    lsm.test()


liquid1 = copy.deepcopy(lsm.liquid_neurons)
#plot_neurons_trace(lsm.liquid_neurons)


lsm.reset()
print(lsm.liquid_neurons[3].U_trace)

"""
for n in lsm.liquid_neurons:
    print(n.step, n.ii, n.ie, n.U_trace)
for s in lsm.synapses_exc:
    print(s.x, s.u)

"""
print(f'forward()')
for step in range(n_steps):
    for i in range(len(lsm.liquid_neurons)):
        lsm.liquid_neurons[i].ie+= lsm.W_in[0][i] * inp2[step] * Ic
    lsm.test()

liquid2 = copy.deepcopy(lsm.liquid_neurons)
#plot_neurons_trace(lsm.liquid_neurons)

lsm.reset()
print(f'forward()')
for step in range(n_steps):
    for i in range(len(lsm.liquid_neurons)):
        lsm.liquid_neurons[i].ie+= lsm.W_in[0][i] * inp3[step] * Ic
    lsm.test()


liquid3 = copy.deepcopy(lsm.liquid_neurons)

#plot_neurons_trace(lsm.liquid_neurons)
lsm.reset()
print(f'forward()')
for step in range(n_steps):
    for i in range(len(lsm.liquid_neurons)):
        lsm.liquid_neurons[i].ie+= lsm.W_in[0][i] * inp4[step] * Ic
    lsm.test()


liquid4 = copy.deepcopy(lsm.liquid_neurons)

#plot_neurons_trace(lsm.liquid_neurons)
lsm.reset()
print(f'forward()')
for step in range(n_steps):
    for i in range(len(lsm.liquid_neurons)):
        lsm.liquid_neurons[i].ie+= lsm.W_in[0][i] * inp5[step] * Ic
    lsm.test()


liquid5 = copy.deepcopy(lsm.liquid_neurons)

#plot_neurons_trace(lsm.liquid_neurons)
lsm.reset()
print(f'forward()')
for step in range(n_steps):
    for i in range(len(lsm.liquid_neurons)):
        lsm.liquid_neurons[i].ie+= lsm.W_in[0][i] * inp6[step] * Ic
    lsm.test()


liquid6 = copy.deepcopy(lsm.liquid_neurons)

#plot_neurons_trace(lsm.liquid_neurons)
lsm.reset()
print(f'forward()')
for step in range(n_steps):
    for i in range(len(lsm.liquid_neurons)):
        lsm.liquid_neurons[i].ie+= lsm.W_in[0][i] * inp7[step] * Ic
    lsm.test()


liquid7 = copy.deepcopy(lsm.liquid_neurons)
#plot_neurons_trace(lsm.liquid_neurons)
lsm.reset()
print(f'forward()')
for step in range(n_steps):
    for i in range(len(lsm.liquid_neurons)):
        lsm.liquid_neurons[i].ie+= lsm.W_in[0][i] * inp8[step] * Ic
    lsm.test()


liquid8 = copy.deepcopy(lsm.liquid_neurons)
d1 = maass_distance(inp1, inp2, dt)
d2 = maass_distance(inp3, inp4, dt)
d3 = maass_distance(inp5, inp6, dt)
d4 = maass_distance(inp7, inp8, dt)
print(f'input distance 12 : {d1}')
print(f'input distance 34: {d2}')
print(f'input distance 56: {d3}')
print(f'input distance 78: {d4}')
plt.figure()
plt.plot(trajectory_distance(liquid1, liquid2, dt, T), label='d(u,v)=%.3f' % d1)
plt.plot(trajectory_distance(liquid3, liquid4, dt, T), label='d(u,v) = %.3f' %d2)
plt.plot(trajectory_distance(liquid5, liquid6, dt, T), label='d(u,v) = %.3f' %d3)
plt.plot(trajectory_distance(liquid7, liquid8, dt, T), label='d(u,v) = %.3f' %d4)
plt.legend()
plt.show()

"""

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
for s in lsm.synapses_exc:
    ax1.plot(s.x_trace)
    ax2.plot(s.u_trace)
    ax3.plot(s.I_trace)
plt.show()
"""