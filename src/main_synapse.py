import numpy as np 
import matplotlib.pyplot as plt

from lsm import LSM
from lif import LIF
from poisson_generator import PoissonSpikeGenerator
from synapse import Synapse
from utils import generate_neurons

w_in = None
w_out = None
distribution = 'gaussian'
p_inh = 0.5
refractory_time = 1e-3
apply_dale = True
T = 3
dt = 1e-3
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


reservoir = LSM(135, 90, (15,3,3), (10,3,3), w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale)
reservoir.paramaters()

internal_topology = reservoir.internal_topology()
connection_infos = internal_topology['exc'][0]

s2 = Synapse(connection_infos,dt)
#s2.W_n = 9e-9
"""
s2.U_ds = 0.45
s2.tau_s = 20e-3
s2.D_ds = 750e-3
s2.F_ds = 50e-3
"""
s2.parameters()
generator2 = PoissonSpikeGenerator(dt, T)
r = 15 
s2.R = r
spike_inputs = generator2.generate(r)

plt.figure()
plt.eventplot(spike_inputs)
plt.show()
print("fffff",spike_inputs)
bool_input = [False]*generator2.n_steps
bool_input = np.array(bool_input)
bool_input[spike_inputs] = True
print(bool_input)
I_trace = []

lif_list = generate_neurons(3)
lif1 = lif_list[0]
lif1.paramaters()
lif3 = lif_list[2]
dd=[]
Ib = 13.5e-9
for step in range(generator2.n_steps):
    I_s = s2.propagate(bool_input[step])
    lif1.receive_input_current(I_s)
    lif1.euler_iteration(I_s + Ib)
    dd.append(I_s + Ib)
    lif3.euler_iteration(dd[-1])
s2.plot()
plt.figure()
plt.plot(dd)
plt.show()

lif1.plot(spike_inputs, r)

lif2 = lif_list[1]
for step in range(generator2.n_steps):
    lif2.euler_iteration(dd[step])
lif2.plot(spike_inputs,r)

"""
s2.u= s2.U_ds * ((1+s2.F_ds*s2.R)/(1+s2.F_ds*s2.R))
s2.x = 1/(1+s2.u*s2.D_ds*s2.R)
s2.I = s2.tau_s * s2.W_n * s2.u*s2.x*s2.R
s2.u_trace = [s2.u]
s2.x_trace = [s2.x]
s2.I_trace = [s2.I]
for step in range(generator2.n_steps):
    s2.test()

s2.plot()

"""