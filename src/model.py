import numpy as np 
import matplotlib.pyplot as plt 

from utils import euclidean_distance
from synapse import Synapse

class LSM:
    """
    LSM Framework from scratch following 'Real-Time Computing Without Stable States: A New
    Framework for Neural Computation Based on Perturbations' Maass 2002
    Based on the work of Ricardo de Azambuja and Robert Kim

    Initialisation of the reservoir's topology
    """
    def __init__(self, N, net_shape, w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale, lbd=1.2):
        """
        Reservoir initialisation method
        N: number of units (neurons)
        net_shape: shape of the cortical column
        w_in:
        w_out:
        distribution:
        p_inh: probability of a neuron being inihibitory
        refractory_time:
        connections_parameters:
        apply_dale: apply Dale's principle
        lbd: 

        """
        self.N = N                                                          
        self.distribution = distribution
        self.p_inh = p_inh
        self.refractory_time = refractory_time
        self.lbd = lbd
        self.connections_parameters = connections_parameters
        self.w_in = w_in
        self.w_out = w_out
        self.apply_dale = apply_dale
        self.net_shape = net_shape
        self.inh, self.exc, self.n_inh, self.n_exc = self.assign_exc_inh()
        self.topology = self.connections_reservoir()
        self.synapses = self.generate_synapses(self.topology) # list of synapsesa


    def assign_exc_inh(self):

        if self.apply_dale:
            inh = np.random.rand(self.N,1) < self.p_inh
            exc = ~inh
            
        else:
            inh = np.random.rand(self.N,1) < 0
            exc = ~inh

        n_inh = len(np.where(inh==True)[0])
        n_exc = self.N - n_inh

        return inh, exc, n_inh, n_exc

    def p_connection(self, C, distance, lbd):
        """
        Compute the probability of the connection between two neurons

        Returns
            p: probability of the connection between two neurons

        """
        p = C * np.exp(-np.power((distance/lbd), 2))
        return p

    def mapping_reservoir(self):
        """
        Maps the internal neurons according to the Cortical column architecture (self.net_shape)

        Returns:
            array: array of triplet (pos_x, pos_y, pos_z)
        """
        print('Mapping : Begin...')
        n_x, n_y, n_z = self.net_shape

        #Center axis except the axis representating the depth of the columns (here y axis)
        dx = -n_x/2.0
        dy = 0
        dz = -n_z/2.0

        positions_list = np.array([(x+dx, y+dy, z+dz) for x in range(n_x) for y in range(n_y) for z in range(n_z)])

        print('Mapping : Done!')
        return positions_list
        

    def connections_reservoir(self):
        """
        Create the topology of the reservoir and generates the connections of the reservoir
        """
        print('Creation of the reservoir : Begin...')
   
        #list of index of inhibitory and excitatory neurons
        inh_index = np.where(self.inh == True)[0]
        exc_index = np.where(self.exc == True)[0]

        #list of the positions of the neurons 
        positions_list = self.mapping_reservoir()
        
        #creation of the list containing infos about each connections
        connections_inh = []
        connections_exc = []


        for i in range(self.N):
            
            for j in range(self.N):
                if i in inh_index:
                    sign = -1
                    if j in inh_index:
                        ## (II)
                        (t_pre, t_pos)=(0,0)
                    else:
                        # (IE)
                        (t_pre, t_pos) = (0,1)
                
                else:
                    sign = 1
                    if j in inh_index:
                        ##(EI)
                        (t_pre, t_pos) = (1,0)
                    else:
                        ## (EE)
                        (t_pre, t_pos) = (1,1)

                #Synapse's parameters following Maass 2002
                CGupta=self.connections_parameters[(t_pre,t_pos)][0] 		# Parameter used at the connection probability - from Maass2002 paper
                UMarkram=self.connections_parameters[(t_pre,t_pos)][1]     	# Use (U) - Parameter used at the Dynamic Synapse - from Maass2002 paper
                DMarkram=self.connections_parameters[(t_pre,t_pos)][2]    	# (second) Time constant for Depression (tau_rec) - used at the Dynamic Synapse - from Maass2002 paper					
                FMarkram=self.connections_parameters[(t_pre,t_pos)][3]    	# (second) Time constant for Facilitation (tau_facil) - used at the Dynamic Synapse - from Maass2002 paper
                AMaass=self.connections_parameters[(t_pre,t_pos)][4]       	# (nA) In the Maass2002 paper the value is negative, but because I need a positive scale (random.normal parameter) and there is a negative sign in front of the abs function I changed this to positive
                Delay_trans=self.connections_parameters[(t_pre,t_pos)][5] 	# (msecond) In Maass paper the transmission delay is 0.8 to II, IE and EI and 1.5 to EE

                U_ds=abs(np.random.normal(loc=UMarkram, scale=UMarkram/2))
                D_ds=abs(np.random.normal(loc=DMarkram, scale=DMarkram/2))
                F_ds=abs(np.random.normal(loc=FMarkram, scale=FMarkram/2))
                W_n=sign*abs(np.random.normal(loc=AMaass, scale=AMaass/2)) # Because AMaass is negative (inhibitory) so is inserted the "-" here

            
                p_connection = self.p_connection(CGupta, euclidean_distance(positions_list[i], positions_list[j]), self.lbd)
                t_connection = (t_pre, t_pos)
                
                if np.random.uniform() <= p_connection:
                    if t_connection[0]==0:
                        connections_inh.append(
                            (
                                (i,j),
                                p_connection,
                                (W_n, U_ds, D_ds, F_ds),
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
        


    def generate_synapses(self, topology):
        connections_exc = topology['exc']
        connections_inh = topology['inh']
        positions_list = topology['pos']
        synapse_list = []

        for inh_infos in connections_inh:
            s = Synapse(inh_infos, 0)
            synapse_list.append(s)
        for exc_infos in connections_exc:
            s = Synapse(exc_infos, 0)
            synapse_list.append(s)
        return synapse_list



    def plot_reservoir(self, topology):
        print('Plotting the Reservoir : Begin...')
        fig = plt.figure(figsize=(12,8))
        fig.suptitle('Reservoir architecture (Cortical Column)')
        
        inh = self.inh
        pos = topology['pos']
        x = [p[0] for p in pos]
        y = [p[1] for p in pos]
        z = [p[2] for p in pos]
        c = ['blue' if i else 'red' for i in inh]
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(x,y,z , color=c)
        ax1.set_title('Mapping of the neurons in the space')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        
        I_connections = topology['inh']
        i_to = [c[0] for c in I_connections] # list of tuple(i,j)
        E_connections = topology['exc']
        e_to = [c[0] for c in E_connections] # list of tuple(i,j)

        #FOR I to
        x1 = []
        y1 = []
        z1 = []
        for c in i_to:
            #(i,j)
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

        
        ax2 = fig.add_subplot(1,2,2, projection='3d')
        ax2.plot(x1,y1,z1, color='blue', lw=0.6)
        ax2.plot(x2, y2, z2, color='red', lw=.6)
        ax2.scatter(x, y, z, alpha=1, color='green')
        ax2.set_title('Synapse connections of the reservoir')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        
        plt.show()
        print('Plotting the Reservoir : Done!')

        
        
    

    def paramaters(self):
        print(f'\n----------------------------------------------\n')
        print(f'Model\'s parameters :')
        print(f'\t N : {self.N}')
        print(f'\t type of distribution : {self.distribution}')
        print(f'\t p_inh : {self.p_inh}')
        print(f'\t refractory time : {self.refractory_time}')
        print(f'\t lbd : {self.lbd}')
        print(f'\t connections_parameters : {self.connections_parameters}')
        print(f'\t w_in : {self.w_in}')
        print(f'\t w_out : {self.w_out}')
        print(f'\t apply_dale : {self.apply_dale}')
        print(f'\t n_inh : {self.n_inh}')
        print(f'\t n_exc : {self.n_exc}')
        print(f'\n----------------------------------------------\n')

