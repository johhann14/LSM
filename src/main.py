import numpy as np 

from model import LSM
from lif import LIF
from poisson_generator import PoissonSpikeGenerator


w_in = None
w_out = None
distribution = 'gaussian'
p_inh = 0.5
refractory_time = 1e-3
apply_dale = True
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


reservoir = LSM(135, (15,3,3), w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale)
reservoir.paramaters()
c = reservoir.connections_reservoir()
reservoir.plot_reservoir(c)

