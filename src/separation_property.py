"""
File: separation_property.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-06-15

File to test if we have the separation property 

References:
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

from lsm import LSM
from utils import connections_parameters, plot_liquid, liquid_default_parameters, plot_neurons_trace, trajectory_distance, maass_distance
from utils import trajectory_distance

p_inh = 0.2
apply_dale = True
dt = 2e-3
n_steps = 1000
T = n_steps * dt
Ic = 70e-9
n_trial = 200
noise_bool = False
np.random.seed(42)
lsm = LSM(N_liquid=600, N_input=1, liquid_net_shape=(20,5,6), connections_parameters=connections_parameters, p_inh=p_inh, apply_dale=apply_dale, dt=dt)

# 0, 0.1, 0.2, 0.4

#average over n_trial random generate pairs

d0 = []
while len(d0) <n_trial:
    inp1 = (np.random.rand(n_steps) < 0.3).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.32).astype(int)
    m = maass_distance(inp1, inp1, dt, T=T)
    d = abs(m - 0)
    if d < 0.01:
        d0.append((inp1, inp1))
    print(m)
    print(len(d0))



d1 = []
while len(d1) <n_trial:
    inp1 = (np.random.rand(n_steps) < 0.5).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.5).astype(int)
    m = maass_distance(inp1, inp2, dt, T)
    d = abs(m - 0.2)
    if d < 0.01:
        d1.append((inp1, inp2))
    inp1 = (np.random.rand(n_steps) < 0.8).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.8).astype(int)
    m = maass_distance(inp1, inp2, dt, T)
    d = abs(m - 0.2)
    if d < 0.01:
        d1.append((inp1, inp2))
    print("d1 : ", m)
    print(len(d1))


d2 = []
while len(d2) <n_trial:
    inp1 = (np.random.rand(n_steps) < 0.8).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.65).astype(int)
    m = maass_distance(inp1, inp2, dt, T)
    d = abs(m - 0.3)
    if d < 0.01:
        d2.append((inp1, inp2))
    inp1 = (np.random.rand(n_steps) < 0.5).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.35).astype(int)
    m = maass_distance(inp1, inp2, dt, T)
    d = abs(m - 0.3)
    if d < 0.01:
        d2.append((inp1, inp2))
    print("d2 :", m)
    print(len(d2))

d3 = []
while len(d3) <n_trial:
    inp1 = (np.random.rand(n_steps) < 0.8).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.53).astype(int)
    m = maass_distance(inp1, inp2, dt, T)
    d = abs(m - 0.4)
    if d < 0.01:
        d3.append((inp1, inp2))
    inp1 = (np.random.rand(n_steps) < 0.4).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.13).astype(int)
    m = maass_distance(inp1, inp2, dt, T)
    d = abs(m - 0.4)
    if d < 0.01:
        d3.append((inp1, inp2))
    print("d3 : ", m)
    print(len(d3))

d0 = d0[:n_trial]
d1 = d1[:n_trial]
d2 = d2[:n_trial]
d3 = d3[:n_trial]
print("D0 building...")
D0 = np.zeros(int(T/dt))
for c in d0:
    lsm.reset()
    u,v =c
    u = np.array(u)
    v = np.array(v)
    u = u[:, None]
    v = v[:, None]
    for step in range(n_steps):
        lsm.step(u[step], Ic, noise_bool=noise_bool)
    liquid1 = copy.deepcopy(lsm.neurons)
    lsm.reset()
    for step in range(n_steps):
        lsm.step(v[step], Ic, noise_bool=noise_bool)

    liquid2 = copy.deepcopy(lsm.neurons)
    D0 += trajectory_distance(liquid1, liquid2, dt, T)

D0 = D0/n_trial


print("D1 building...")
D1 = np.zeros(int(T/dt))
for c in d1:
    lsm.reset()
    u,v =c
    u = np.array(u)
    v = np.array(v)
    u = u[:, None]
    v = v[:, None]
    for step in range(n_steps):
        lsm.step(u[step], Ic, noise_bool=noise_bool)
    liquid1 = copy.deepcopy(lsm.neurons)
    lsm.reset()
    for step in range(n_steps):
        lsm.step(v[step], Ic, noise_bool=noise_bool)

    liquid2 = copy.deepcopy(lsm.neurons)
    D1 += trajectory_distance(liquid1, liquid2, dt, T)

D1 = D1/n_trial

print("D2 building...")
D2 = np.zeros(int(T/dt))
for c in d2:
    lsm.reset()
    u,v =c
    u = np.array(u)
    v = np.array(v)
    u = u[:, None]
    v = v[:, None]
    for step in range(n_steps):
        lsm.step(u[step], Ic, noise_bool=noise_bool)
    liquid1 = copy.deepcopy(lsm.neurons)
    lsm.reset()
    for step in range(n_steps):
        lsm.step(v[step], Ic, noise_bool=noise_bool)

    liquid2 = copy.deepcopy(lsm.neurons)
    D2 += trajectory_distance(liquid1, liquid2, dt, T)

D2 = D2/n_trial



print("D3 building...")
D3 = np.zeros(int(T/dt))
for c in d3:
    lsm.reset()
    u,v =c
    u = np.array(u)
    v = np.array(v)
    u = u[:, None]
    v = v[:, None]
    for step in range(n_steps):
        lsm.step(u[step], Ic, noise_bool=noise_bool)
    liquid1 = copy.deepcopy(lsm.neurons)
    lsm.reset()
    for step in range(n_steps):
        lsm.step(v[step], Ic, noise_bool=noise_bool)

    liquid2 = copy.deepcopy(lsm.neurons)
    D3 += trajectory_distance(liquid1, liquid2, dt, T)

D3 = D3/n_trial

plt.figure(figsize=(10,8))
plt.title("Separation Property")
plt.plot(D0, label='d(u,v)=%.3f' % 0)
plt.plot(D1, label='d(u,v) = %.3f' % 0.2)
plt.plot(D2, label='d(u,v) = %.3f' % 0.3)
plt.plot(D3, label='d(u,v) = %.3f' % 0.4)
plt.xlabel("time step dt=2ms")
plt.ylabel("trajectory distance")
plt.grid()
plt.legend()
plt.show()



