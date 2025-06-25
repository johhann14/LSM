from lif import LIF
from poisson_generator import PoissonSpikeGenerator
import matplotlib.pyplot as plt
import numpy as np




dt = 1e-3
T = 3
n_steps = int(T/dt)

generator = PoissonSpikeGenerator(dt, T)
generator.parameters()
rate = 15
spike_inputs = generator.generate(rate)


plt.figure()
plt.eventplot(spike_inputs, color='black')
plt.show()
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