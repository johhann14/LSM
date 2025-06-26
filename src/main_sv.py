import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from lsm_v3 import LSM
from utils import connections_parameters, plot_liquid, liquid_default_parameters, plot_neurons_trace, trajectory_distance, maass_distance
from utils import plot_neurons_trace, trajectory_distance
p_inh = 0.2
apply_dale = True
dt = 2e-3
T = 0.5
n_steps = int(T/dt)

Ic = 70e-9
np.random.seed(42)
lsm = LSM(N_liquid=600, N_input=300, liquid_net_shape=(20,5,6), connections_parameters=connections_parameters, p_inh=p_inh, apply_dale=apply_dale, dt=dt)

inp1 = (np.random.rand(n_steps) < 0.1).astype(int)
inp2 = (np.random.rand(n_steps) < 0.9).astype(int)

inp3 = (np.random.rand(n_steps) < 0.3).astype(int)
inp4 = (np.random.rand(n_steps) < 0.3).astype(int)

inp5 = (np.random.rand(n_steps) < 0.3).astype(int)
inp6 = (np.random.rand(n_steps) < 0.8).astype(int)

inp7 = (np.random.rand(n_steps) < 0.3).astype(int)
inp8 = inp7[:]
inp8[[1,10,37,100,89, 30, 23, 144, 233, 244, 38, 32]] = 1
start_time = time.time()
for step in range(n_steps):
    lsm.forward(inp1[step], Ic=Ic)

liquid1 = copy.deepcopy(lsm.liquid_neurons)
lsm.reset()


for step in range(n_steps):
    lsm.forward(inp2[step], Ic=Ic)

liquid2 = copy.deepcopy(lsm.liquid_neurons)
lsm.reset()

for step in range(n_steps):
    lsm.forward(inp3[step], Ic=Ic)

liquid3 = copy.deepcopy(lsm.liquid_neurons)


lsm.reset()
for step in range(n_steps):
    lsm.forward(inp4[step], Ic=Ic)

liquid4 = copy.deepcopy(lsm.liquid_neurons)


lsm.reset()
for step in range(n_steps):
    lsm.forward(inp5[step], Ic=Ic)

liquid5 = copy.deepcopy(lsm.liquid_neurons)


lsm.reset()
for step in range(n_steps):
    lsm.forward(inp6[step], Ic=Ic)

liquid6 = copy.deepcopy(lsm.liquid_neurons)


lsm.reset()
for step in range(n_steps):
    lsm.forward(inp7[step], Ic=Ic)

liquid7 = copy.deepcopy(lsm.liquid_neurons)
lsm.reset()


for step in range(n_steps):
    lsm.forward(inp8[step], Ic=Ic)

liquid8 = copy.deepcopy(lsm.liquid_neurons)
lsm.reset()

end_time = time.time()
print(f"Elapsed time: {end_time - start_time}")
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