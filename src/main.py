import numpy as np 
import matplotlib.pyplot as plt

from model import LSM
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
c = reservoir.internal_topology()
#reservoir.plot_reservoir(c)

c1 = c['exc'][9]
print(c1)
synapse = Synapse(c1, dt)
synapse.parameters()


#========================GENERATOR==========================
generator = PoissonSpikeGenerator(dt, T)
generator.parameters()
rate = 15
spike_inputs = generator.generate(rate)

plt.figure()
plt.eventplot(spike_inputs, color='black')
plt.show()
#===========================================================


#========================LIF==========================
lif1 = LIF(0.4, 0, dt, T, 5.1, 5e-3, 0.3)
lif1.paramaters()
#===========================================================

#========================INPUT CURRENT==========================
I = 1.4
I_trace = np.zeros(generator.n_steps)
I_trace[spike_inputs] = I
#===============================================================

#=======================LIF SIMULATION========================
for step in range(generator.n_steps):
    lif1.euler_iteration(I_trace[step])

lif1.plot(spike_inputs, rate)

#=======================Simple synapse tryout==============

lif2 = LIF(0.4, 0, dt, T, 5.1, 5e-3, 0.3)
lif3 = LIF(0.4, 0, dt, T, 5.1, 5e-3, 0.3)
print(c1)
z = I_trace[:]
z[:] = 0
print(z)

for step in range(generator.n_steps):
    lif2.euler_iteration(I_trace[step])
    spike = False
    if step>0:
        if (step-1) in lif2.spike_trace:
            spike = True
    z[step] = synapse.propagate(spike, step-1)
print("okkkkkkkkkkkkk")

    

c1 = reservoir.input_reservoir_topology()
print(len(c1['exc']), len(c1['inh']))
reservoir.plot_lsm_topology(c1, c)
