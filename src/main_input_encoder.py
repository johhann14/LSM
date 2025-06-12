import numpy as np
import matplotlib.pyplot as plt
from input_encoder import InputEncoder

T = 3           # 3sec
dt = 1e-3       # 1ms
T_period = 100e-3
encoder = InputEncoder(T,dt)
range_t = np.linspace(0,T,int(T/dt))
s = np.sin(range_t * 2 * np.pi / T_period)
spikes = encoder.encode(s)
print(spikes)
encoder.plot(s, spikes)