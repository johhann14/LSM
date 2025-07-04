"""
File: utils.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-06-15

File containing helper functions and STP/LIF parameters.

References:
    - W. Maass, T. Natschläger, H. Markram (2002). 
        Real-Time Computing Without Stable States: A New Framework for Neural Computation Based on Perturbations.
    
    - R. de Azambuja, F. B. Klein, S. V. Adams, M. F. Stoelen, A. Cangelosi
        Short-Term Plasticity in a Liquid State Machine Biomimetic Robot Arm Controller.
"""


from lif import LIF
import numpy as np 
import matplotlib.pyplot as plt
from poisson_generator import PoissonSpikeGenerator
from scipy.signal import convolve
from synapse import Synapse


# E = 1
# I = 0
connections_parameters = {
    #(i,j) = [C, U(use), D(time constant for depression in s), F(time constant for facilitation in s), A(scaling parameter in nA), transmission delay]
    #(i,j) = [C, U, D, F, A, transmission_delay]
    (0,0) : [0.1, 0.32, 0.144, 0.06, -47e-9, 0.8e-3],       # (II)

    (0,1) : [0.4, 0.25, 0.7, 0.02, -47e-9, 0.8e-3],         # (IE)

    (1,0) : [0.2, 0.05, 0.125, 1.2, 150e-9, 0.8e-3],        # (EI)

    (1,1) : [0.3, 0.5, 1.1, 0.05, 70e-9, 1.5e-3,]           # (EE)

}

liquid_default_parameters = {
    'taum' : 30e-3,                     # Membrane time constant (tau_m) s
    'Cm' : 30e-9,                       # Membrane capacitance (Cm) F
    'tau_se' : 3e-3,                    # Synapse time constant (exc. tausyne) s
    'tau_si' : 6e-3,                    # Synapse time constant (inh. taysyni) s
    'refrac_period_e' : 3e-3,           # Refactory period (exc.) s
    'refrac_period_i' : 2e-3,           # Refractory perdiod (inh.) s
    'U_threshold' : 15e-3,              # Membrane threshold V
    'U_reset' : [13.8e-3, 14.5e-3],     # Membrane Reset V
    'U_initial' : [13.5e-3, 14.9e-3],   # Membrane Initial V
    'i_offset' : [13.5e-9, 14.5e-9],    # i_offset A
    'i_noise_mu' : 0,                   # i_noise mu A
    'i_noise_scale' : 1.0e-9,           # i_noise scale A
    'transmission_delay_e' : 1.5e-3,    # Transmission delay (exc.) s
    'transmission_delay_i' : 0.8e-3     # Transmission delay (inh.) s
}



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



#====== For the separation property ============
def gaussian_kernel(tau, dt):
    t = np.arange(-3*tau, 3*tau + dt, dt)
    g = np.exp(-(t/tau)**2)
    return g / np.sum(g)  

def continuous_representation(spike_train, tau, dt):
    kernel = gaussian_kernel(tau, dt)
    convolved = convolve(spike_train, kernel, mode='full')[:len(spike_train)]
    return convolved

def maass_distance(u, v, dt, T=0.5, tau=0.005):
    u_cont = continuous_representation(u, tau, dt)
    v_cont = continuous_representation(v, tau, dt)
    diff = u_cont - v_cont
    distance = np.sqrt(np.sum(diff**2) * dt) / np.sqrt(T)
    return distance

def trajectory_distance(liquid1, liquid2, dt, T):
    D = []
    U1 = liquid1.U_trace
    U2 = liquid2.U_trace
    for step in range(int(T/dt)):
        D.append(np.linalg.norm(U1[step] -U2[step]))

    return D

#===============================================================


def assign_exc_inh(N, apply_dale, p_inh):
    """
    Assigns randomly index for inhbitory and excitatory neurons given a probability p_inh
    """
    if apply_dale:
        inh = np.random.rand(N) < p_inh
        exc = ~inh #inversing bits
        
    else:
        inh = np.random.rand(N) < 0
        exc = ~inh

    n_inh = len(np.where(inh==True)[0])
    n_exc = N - n_inh

    return inh, exc, n_inh, n_exc


def mapping_reservoir(liquid_net_shape):
    """
    Maps the internal neurons according to the Cortical column architecture (liquid_net_shape)

    Returns:
        array: array of triplet (pos_x, pos_y, pos_z)
    """
    print('Mapping : Begin...')

    n_x, n_y, n_z = liquid_net_shape

    #Center axis except the axis representating the depth of the columns (here y axis)
    dx = -n_x/2.0
    dy = 0
    dz = -n_z/2.0

    positions_list = np.array([(x+dx, y+dy, z+dz) for x in range(n_x) for y in range(n_y) for z in range(n_z)])

    print('Mapping : Done!')
    return positions_list


def make_liquid_topology(connections_parameters, net_shape, inh_liquid, lbd):
    """
    Create the topology of the reservoir and generates the connections of the reservoir.
    """

    print('Creation of the reservoir topology : Begin...')

    # Size of the liquid
    N = net_shape[0] * net_shape[1] * net_shape[2]

    #list of index of inhibitory and excitatory neurons
    inh_index = np.where(inh_liquid == True)[0]
    exc_index = np.where(inh_liquid == False)[0]

    #list of the positions of the neurons 
    positions_list = mapping_reservoir(net_shape)
    
    #creation of the list containing infos about each connections
    connections_inh = []
    connections_exc = []

    for i in range(N):
        
        for j in range(N):
            if i!=j:
                if i in inh_index:
                    if j in inh_index:
                        ## (II)
                        (t_pre, t_pos)=(0,0)
                        
                    else:
                        # (IE)
                        (t_pre, t_pos) = (0,1)
                
                else:
                    if j in inh_index:
                        ##(EI)
                        (t_pre, t_pos) = (1,0)
                    else:
                        ## (EE)
                        (t_pre, t_pos) = (1,1)

                #Synapse's parameters following Maass 2002
                C=connections_parameters[(t_pre,t_pos)][0] 		
                UMarkram=connections_parameters[(t_pre,t_pos)][1]     	
                DMarkram=connections_parameters[(t_pre,t_pos)][2]    				
                FMarkram=connections_parameters[(t_pre,t_pos)][3]    	
                AMaass=connections_parameters[(t_pre,t_pos)][4]       	
                Delay_trans=connections_parameters[(t_pre,t_pos)][5]

                U_ds=abs(np.random.normal(loc=UMarkram, scale=UMarkram/2))
                """
                if U_ds < 0:
                    U_ds = np.random.uniform(0,1)
                """
                D_ds=abs(np.random.normal(loc=DMarkram, scale=DMarkram/2))
                """
                if D_ds < 0:
                    D_ds = np.random.uniform(0,1)
                """
                F_ds=abs(np.random.normal(loc=FMarkram, scale=FMarkram/2))
                """
                if F_ds < 0:
                    F_ds = np.random.uniform(0,1)
                """
                """
                shape = 1.0
                """
        
                W_n = abs(np.random.normal(loc=AMaass, scale = np.abs(AMaass)/2))

                """
                if AMaass < 0:
                    W_n = -A_sample
                else:
                    W_n = A_sample
                """
                d = euclidean_distance(positions_list[i], positions_list[j])
                p_connection = probability_connection(C, d, lbd)

                t_connection = (t_pre, t_pos)
                if np.random.uniform() <= p_connection:
                    

                    if t_connection[0]==0:
                        connections_inh.append(
                            (
                                (i,j),
                                p_connection,
                                (-W_n, U_ds, D_ds, F_ds), #added the sign here
                                Delay_trans,
                                t_connection
                            )       
                        )
                    else:
                        connections_exc.append(
                            (
                                (i,j),
                                p_connection,
                                (W_n, U_ds, D_ds, F_ds),
                                Delay_trans,
                                t_connection
                            )      
                        )
                    

            
    print('Creation of the reservoir : Done!')
    
    return {'exc':connections_exc, 'inh':connections_inh, 'pos':positions_list}


def plot_liquid(topology, inh_liquid):
    """
    3D Plot of the liquid
    """
    print('Plotting the liquid : Begin...')
    inh = inh_liquid
    pos = topology['pos']
    x = [p[0] for p in pos]
    y = [p[1] for p in pos]
    z = [p[2] for p in pos]
    c = ['blue' if i else 'red' for i in inh]

    #RESERVOIR
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(x,y,z , color=c, alpha=1, edgecolors='k')
    ax1.set_title('Mapping of the internal neurons into the space')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    I_connections = topology['inh']
    i_to = [c[0] for c in I_connections] # list of tuple(i,j)
    E_connections = topology['exc']
    e_to = [c[0] for c in E_connections] # list of tuple(i,j)

    x1 = []
    y1 = []
    z1 = []
    for c in i_to:
        i,j = c
        x1.append(pos[i][0])
        x1.append(pos[j][0])
        y1.append(pos[i][1])
        y1.append(pos[j][1])
        z1.append(pos[i][2])
        z1.append(pos[j][2])
    x2 = []
    y2 = []
    z2= []
    for c in e_to:
        i,j = c
        x2.append(pos[i][0])
        x2.append(pos[j][0])
        y2.append(pos[i][1])
        y2.append(pos[j][1])
        z2.append(pos[i][2])
        z2.append(pos[j][2])


    ax2= fig.add_subplot(1,2,2, projection='3d')
    ax2.plot(x1,y1,z1, color='blue', lw=0.6 , linestyle='--')
    ax2.plot(x2, y2, z2, color='red', lw=.6)
    ax2.scatter(x, y, z, alpha=1, color='green', edgecolors='k')
    ax2.set_title('Liquid synapse connections ')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    #plt.savefig("results/liquid_topology.png")
    plt.show()
    print('Plotting the liquid : Done!')




def plot_neurons_trace(liquid_neurons):
    """
    Plot membrane potential, spike trace of the liquid through time
    """
    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)
    st = []
    sum_trace = []
    fig.suptitle("Liquid Trace")
    U_trace = np.array(liquid_neurons.U_trace)
    spike_trace = np.array(liquid_neurons.spike_trace)
    for n in range(liquid_neurons.N):
        ax1.plot(U_trace[:, n])
        st.append(np.where(spike_trace[:, n] == True)[0])
        sum_trace.append(spike_trace[: ,n])
    
    for neuron, t in enumerate(st):
        ax2.scatter(t, [neuron] * len(t), color='blue', s=1, alpha=0.3)
    ax1.axhline(liquid_default_parameters['U_threshold'], linestyle='--', color='black')
    sum_trace = np.array(sum_trace).astype(int)
    s = np.sum(sum_trace, axis=0)
    ax3.plot(s)
    print(sum_trace.shape)
    print(s.shape)
    ax1.set_title("Neurons voltage")
    ax2.set_title("Liquid pattern")
    ax3.set_title("Number of emitted spikes")
    ax3.set_xlabel('Simulation step')
    #plt.savefig("results/test_lsm_with_noise/liquid_trace.png")
    plt.show()

