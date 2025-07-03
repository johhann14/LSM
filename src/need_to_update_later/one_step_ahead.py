import numpy as np
import matplotlib.pyplot as plt
import copy

from lsm import LSM
from utils import connections_parameters, plot_neurons_trace
from utils import encode2
from input_encoder import InputEncoder
from need_to_update_later.readout import Readout
from sklearn.metrics import r2_score

noise_bool = False
p_inh = 0.2
apply_dale = True
dt = 2e-3
T = 1
n_steps = int(T/dt)
Ic = 10e-9
n_trial = 200
n_input = 300
n_readout = 1
T_period = 30e-3

np.random.seed(42)

lsm = LSM(
    N_liquid=600,
    N_input=n_input,
    liquid_net_shape=(20,5,6),
    connections_parameters=connections_parameters,
    p_inh=p_inh,
    apply_dale=apply_dale,
    dt=dt,
    enable_stp=True)

#=================== Input preparation ==============
n_range = np.linspace(0, T, n_steps)
y_range = n_range[1:]

n_range = n_range[:-1]
u_train = np.sin(n_range * 2 * np.pi / T_period)
y_train = np.sin(y_range * 2 * np.pi / T_period)
encoder = InputEncoder(n_steps=n_steps -1, n_input=n_input)


spikes = encoder.rate_encoding(u_train)
encoder.plot(u_train, spikes)

#================= Forward ======================
print(f"Forward()...")
for step in range(n_steps -1):
    lsm.step(inp=spikes[step], Ic=Ic, noise_bool=noise_bool)
print(f"Forward() : Done!")


liquid = copy.deepcopy(lsm.neurons.U_trace)
liquid = np.array(liquid)


#================== Liquid Trace ===============
plot_neurons_trace(lsm.neurons)
lsm.synapses.plot(N=60)


#=================== Readout =======================
readout = Readout(N_readout=n_readout, lsm=lsm, n_steps=n_steps)
readout.train(liquid[100:], y_train=y_train[100:])
print(f"training score : {readout.Wout.score(liquid, y_train)}")

steps_ahead = 500
pred = np.zeros((steps_ahead, n_readout))
lsm.reset()
n_test = np.linspace(T, T+ steps_ahead*dt, steps_ahead)
x_test = np.sin(n_test * 2 * np.pi / T_period)
x_test = encoder.rate_encoding(x_test)
prediction = np.zeros((steps_ahead, readout.N_readout))
for step in range(steps_ahead):
    u = x_test[step]
    lsm.step(u, Ic, noise_bool=noise_bool)
    liquid_state = lsm.neurons.U.copy()
    y = readout.output(liquid_state=liquid_state)
    prediction[step] = y
    


n_target = np.linspace(T+dt, T+ dt + steps_ahead*dt, steps_ahead)
y_target = np.sin(n_target * 2 * np.pi / T_period)

print(f"test score : {r2_score(y_target, prediction)}")

fig = plt.figure()
fig.suptitle("Training, target and predicted signal")
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.plot(u_train, label='u_train')
ax1.plot(y_train, label='y_train')
ax2.plot(y_target, label='target signal')
ax2.plot(prediction, label='predicted signal')
ax2.set_xlabel('time step')
plt.legend()
plt.show()



