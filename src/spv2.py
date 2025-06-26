import numpy as np
import matplotlib.pyplot as plt
import copy

from lsm import LSM
from utils import connections_parameters, plot_liquid, liquid_default_parameters, plot_neurons_trace, trajectory_distance, maass_distance
from utils import trajectory_distance

p_inh = 0.2
apply_dale = True
dt = 1e-3
T = 0.5
n_steps = int(T/dt)
Ic = 70e-9
n_trial = 200
np.random.seed(42)
lsm = LSM(N_liquid=600, N_input=300, liquid_net_shape=(20,5,6), connections_parameters=connections_parameters, p_inh=p_inh, apply_dale=apply_dale, dt=dt)

# 0, 0.1, 0.2, 0.4

#average over 200 randoml generate pairs

d0 = []
while len(d0) <n_trial:
    inp1 = (np.random.rand(n_steps) < 0.3).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.32).astype(int)
    m = maass_distance(inp1, inp1, dt)
    d = abs(m - 0)
    if d < 0.01:
        d0.append((inp1, inp1))
    print(m)
    print(len(d0))



d1 = []
while len(d1) <n_trial:
    inp1 = (np.random.rand(n_steps) < 0.5).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.5).astype(int)
    m = maass_distance(inp1, inp2, dt)
    d = abs(m - 0.2)
    if d < 0.01:
        d1.append((inp1, inp2))
    inp1 = (np.random.rand(n_steps) < 0.8).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.8).astype(int)
    m = maass_distance(inp1, inp2, dt)
    d = abs(m - 0.2)
    if d < 0.01:
        d1.append((inp1, inp2))
    print("d1 : ", m)
    print(len(d1))


d2 = []
while len(d2) <n_trial:
    inp1 = (np.random.rand(n_steps) < 0.8).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.65).astype(int)
    m = maass_distance(inp1, inp2, dt)
    d = abs(m - 0.3)
    if d < 0.01:
        d2.append((inp1, inp2))
    inp1 = (np.random.rand(n_steps) < 0.5).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.35).astype(int)
    m = maass_distance(inp1, inp2, dt)
    d = abs(m - 0.3)
    if d < 0.01:
        d2.append((inp1, inp2))
    print("d2 :", m)
    print(len(d2))

d3 = []
while len(d3) <n_trial:
    inp1 = (np.random.rand(n_steps) < 0.8).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.53).astype(int)
    m = maass_distance(inp1, inp2, dt)
    d = abs(m - 0.4)
    if d < 0.01:
        d3.append((inp1, inp2))
    inp1 = (np.random.rand(n_steps) < 0.4).astype(int)
    inp2 = (np.random.rand(n_steps) < 0.13).astype(int)
    m = maass_distance(inp1, inp2, dt)
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
    for step in range(n_steps):
        lsm.forward(u[step], Ic)
    liquid1 = copy.deepcopy(lsm.liquid_neurons)
    lsm.reset()
    for step in range(n_steps):
        lsm.forward(v[step], Ic)

    liquid2 = copy.deepcopy(lsm.liquid_neurons)
    D0 += trajectory_distance(liquid1, liquid2, dt, T)

D0 = D0/n_trial


print("D1 building...")
D1 = np.zeros(int(T/dt))
for c in d1:
    lsm.reset()
    u,v =c
    for step in range(n_steps):
        lsm.forward(u[step], Ic)
    liquid1 = copy.deepcopy(lsm.liquid_neurons)
    lsm.reset()
    for step in range(n_steps):
        lsm.forward(v[step], Ic)

    liquid2 = copy.deepcopy(lsm.liquid_neurons)
    D1 += trajectory_distance(liquid1, liquid2, dt, T)

D1 = D1/n_trial

print("D2 building...")
D2 = np.zeros(int(T/dt))
for c in d2:
    lsm.reset()
    u,v =c
    for step in range(n_steps):
        lsm.forward(u[step], Ic)
    liquid1 = copy.deepcopy(lsm.liquid_neurons)
    lsm.reset()
    for step in range(n_steps):
        lsm.forward(v[step], Ic)

    liquid2 = copy.deepcopy(lsm.liquid_neurons)
    D2 += trajectory_distance(liquid1, liquid2, dt, T)

D2 = D2/n_trial



print("D3 building...")
D3 = np.zeros(int(T/dt))
for c in d3:
    lsm.reset()
    u,v =c
    for step in range(n_steps):
        lsm.forward(u[step], Ic)
    liquid1 = copy.deepcopy(lsm.liquid_neurons)
    lsm.reset()
    for step in range(n_steps):
        lsm.forward(v[step], Ic)

    liquid2 = copy.deepcopy(lsm.liquid_neurons)
    D3 += trajectory_distance(liquid1, liquid2, dt, T)

D3 = D3/n_trial

plt.figure()
plt.plot(D0, label='d(u,v)=%.3f' % 0)
plt.plot(D1, label='d(u,v) = %.3f' % 0.2)
plt.plot(D2, label='d(u,v) = %.3f' % 0.3)
plt.plot(D3, label='d(u,v) = %.3f' % 0.4)
plt.legend()
plt.show()



