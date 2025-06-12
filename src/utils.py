from lif import LIF
import numpy as np 

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def probability_connection(C, distance, lbd):
    """
    Compute the probability of the connection between two neurons

    Returns
        p: probability of the connection between two neurons

    """
    p = C * np.exp(-np.power((distance/lbd), 2))
    return p



def generate_neurons(N):
    """
    Generate N LIF neurons

    Returns
        Array of neurons
    """
    print(f'Generate neurons : Begin...')
    Ib = 13.5e-9            # constant nonspecific background current 13.5nA 
    R = 1e6                 # input resistance 1MOmega
    threshold = 15e-3       # threshold 15mV
    U_reset = 13.5e-3       # reset voltage 13.5mV
    tau = 30e-3
    C = tau/R             # membrane time constant 30ms
    U = 0# uniform distribution from the interval [13.5mV, 15.0V]

    neurons_list = [LIF(U_init=np.random.uniform(13.5e-3, 15e-3), U_reset=U_reset, dt=1e-3, T=3, R=R, C=C, threshold=threshold, Ib=Ib) for _ in range(N)]
    return neurons_list