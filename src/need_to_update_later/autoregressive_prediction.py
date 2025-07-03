import numpy as np
import matplotlib.pyplot as plt
import copy

from lsm import LSM
from utils import connections_parameters, plot_neurons_trace
from input_encoder import InputEncoder
from need_to_update_later.readout import Readout
from sklearn.metrics import r2_score

p_inh = 0.2
apply_dale = True
dt = 2e-3
T = 0.5
n_steps = int(T/dt)
Ic = 10e-9
n_trial = 200
n_input = 300
T_period = 25e-3
noise_bool = False

np.random.seed(42)

lsm = LSM(
    N_liquid=600,
    N_input=n_input,
    liquid_net_shape=(20,5,6),
    connections_parameters=connections_parameters,
    p_inh=p_inh,
    apply_dale=apply_dale,
    dt=dt)

# input preparation

n_range = np.linspace(0, T, n_steps)
y_range = n_range[1:]

n_range = n_range[:-1]
print(n_range.shape, y_range.shape)
u_train = np.sin(n_range * 2 * np.pi / T_period)
y_train = np.sin(y_range * 2 * np.pi / T_period)
encoder = InputEncoder(n_steps=n_steps -1, n_input=n_input)


spikes = encoder.encode(u_train)
#encoder.plot(u_train, spikes)

inp = spikes.astype(int)
print(type(inp))
print(inp.shape)

print(f"Forward()...")
for step in range(n_steps -1):
    lsm.step(inp=inp[step], Ic=Ic, noise_bool=noise_bool)
print(f"Forward() : Done!")


liquid = copy.deepcopy(lsm.neurons.U_trace)
liquid = np.array(liquid)
print(liquid.shape)
plot_neurons_trace(lsm.neurons)
lsm.synapses.plot(80)

mu = np.mean(liquid, axis=0)
sigma = np.std(liquid, axis=0)

liquid_norm = (liquid - mu)/sigma


readout = Readout(N_readout=1, lsm=lsm, n_steps=n_steps)
readout.train(liquid[100:], y_train=y_train[100:])
print(f"training score : {readout.Wout.score(liquid, y_train)}")

steps_ahead = 500
u = np.sin(T)
y_pred = readout.predict(u, steps_ahead, Ic, n_input, noise_bool, mu, sigma)

n_target = np.linspace(T+dt, T+dt + steps_ahead*dt, steps_ahead)
y_target = np.sin(n_target * 2 * np.pi / T_period)
print(f"test score : {r2_score(y_target, y_pred)}")

fig = plt.figure()
fig.suptitle("Training, target and predicted signal")
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.plot(u_train, label='u_train')
ax1.plot(y_train, label='y_train')
ax2.plot(y_pred, label='predicted signal')
ax2.plot(y_target, label='target signal')
ax2.set_xlabel('time step')
plt.legend()
plt.show()
print(y_pred[0], y_target[0])



