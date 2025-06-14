import numpy as np 
import matplotlib.pyplot as plt

from lsm import LSM
from lif import LIF
from poisson_generator import PoissonSpikeGenerator
from synapse import Synapse


w_in = None
w_out = None
distribution = 'gaussian'
p_inh = 0.5
refractory_time = 1e-3
apply_dale = True
dt = 1e-3
T = 3

# 1 = E
# 0 = I
connections_parameters = {
    #(i,j) = [C, U(use), D(time constant for depression in s), F(time constant for facilitation in s), A(scaling parameter in nA), transmission delay]
    (0,0) : [0.1,
             0.32,
             0.144,
             0.06,
             2.8,
             0.8],

    (0,1) : [0.4,
             0.25,
             0.7,
             0.02,
             3.0,
             0.8],

    (1,0) : [0.2,
             0.05,
             0.125,
             1.2,
             1.6,
             0.8,],

    (1,1) : [0.3,
             0.5,
             1.1,
             0.05,
             1.2,
             1.5,]

}


reservoir = LSM(135, 90, (15,3,3), (10,3,3), w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale)
reservoir.paramaters()
internal_topology = reservoir.internal_topology()


#========================GENERATOR==========================
generator = PoissonSpikeGenerator(dt, T)
generator.parameters()
rate = 15
spike_inputs = generator.generate(rate)

plt.figure()
plt.eventplot(spike_inputs, color='black')
plt.show()

#========================INPUT CURRENT==========================
I = 1.4
I_trace = np.zeros(generator.n_steps)
I_trace[spike_inputs] = I

#=======================Simple synapse tryout==============
connection_infos = internal_topology['exc'][0]
s = Synapse(connection_infos, dt)
lif4 = LIF(0.4, 0, dt, T, 5.1, 5e-3, 0.3)
lif5 = LIF(0.4, 0, dt, T, 5.1, 5e-3, 0.3)
inputt = []

for step in range(generator.n_steps):
    lif4.euler_iteration(I_trace[step])
    I_s = s.propagate(lif4.spiked_before)
    inputt.append(I_s)
    lif5.euler_iteration(I_s)
lif4.plot(spike_inputs, 4)
lif5.plot(spike_inputs, 5)


s.plot()
plt.figure()
plt.eventplot(lif4.spike_trace)
plt.savefig('results/output_precell.png')
plt.show()

s2 = Synapse(connection_infos,dt)
s2.W_n = 1
s2.U_ds = 0.45
s2.tau_s = 20e-3
s2.D_ds = 750e-3
s2.F_ds = 50e-3

generator2 = PoissonSpikeGenerator(dt, T)
r = 15 
spike_inputs = generator.generate(r)

plt.figure()
plt.eventplot(spike_inputs)
plt.show()
print("fffff",spike_inputs)
bool_input = [False]*generator2.n_steps
bool_input = np.array(bool_input)
bool_input[spike_inputs] = True

I_trace = []
for step in range(generator.n_steps):
    I_s = s2.propagate(bool_input[step])
s2.plot()

x_trace = s2.x_trace
plt.figure()
plt.plot(x_trace)
plt.xlim(10)
plt.show()


