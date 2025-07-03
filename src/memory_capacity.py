"""
File: memory_capacity.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-06-30

File to test if we have the memory capacity

References:
    - H. Jaeger.
        The "echo state" approach to analysing and training recurrent neural networks - with an Erratum note
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
"""
import numpy as np
import matplotlib.pyplot as plt
import copy

from lsm import LSM
from utils import connections_parameters, plot_neurons_trace
from input_encoder import InputEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split

p_inh           = 0.2
apply_dale      = True
dt              = 2e-3
n_steps         = 1000
T               = n_steps * dt
Ic              = 7e-11
n_trial         = 200
n_input         = 300
enable_noise    = False

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

# input preparation

#u = np.random.normal(loc=0, scale=1, size=n_steps)
u = np.random.uniform(-0.5, 0.5, size=n_steps)


encoder = InputEncoder(n_steps=n_steps, n_input=n_input)
input_spikes = encoder.rate_encoding(u)
encoder.plot(u, input_spikes)

washout = 200
X = np.zeros((n_steps - washout, lsm.N_liquid))
for step in range(n_steps):
    lsm.step(inp=input_spikes[step], Ic=Ic, noise_bool=enable_noise)
    if step >= washout:
        liquid = lsm.neurons.U.copy()
        X[step - washout,:] = liquid

index = np.random.choice(np.arange(0, lsm.N_liquid, 1),30, False)

plot_neurons_trace(lsm.neurons)
lsm.synapses.plot(50)
sp = np.array(lsm.neurons.spike_trace)  # shape (500, N_liquid)
mean_rate = sp.mean() / dt  # en Hz
print(f"Spike rate moyen : {mean_rate:.1f} Hz par neurone")
U = u[washout:]
print(U.shape, X.shape)

D_max = 100
mc_d = []

for d in range(1, D_max+1):
    print(f"d : {d}")
    X_d = X[d:] #enleve les d premiers
    y_d = U[:-d] # enelver les d derniers
    # a t, est ce qu'on se souvient de t-d
    # a partir de x(t) on veut predire u(t-d)
    clf = LinearRegression().fit(X_d, y_d)
    y_pred = clf.predict(X_d)
    c = np.corrcoef(y_d, y_pred)[0,1]**2
    mc_d.append(c)
MC = np.sum(mc_d)
print(f"MC = {MC}")

plt.figure(figsize=(8,6))
plt.title("Memory capacity (Shor-term memory from Jaeger) with dt=2ms and T= 2s")
plt.plot(range(1, D_max+1), mc_d, label='Memory Curve')
plt.ylabel("Correlation coefficient")
plt.xlabel("Delay step d")
plt.ylim(0,1.2)
plt.xlim(0, D_max +2)
plt.legend()
plt.grid()
# plt.savefig("results/test_lsm_uniform_input/memory_capacity.png")
plt.show()









