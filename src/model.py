import numpy as np 
import matplotlib.pyplot as plt 

from utils import euclidean_distance
from synapse import Synapse
from lif import LIF

class LSM:
    """
    LSM Framework from scratch following 'Real-Time Computing Without Stable States: A New
    Framework for Neural Computation Based on Perturbations' Maass 2002
    Based on the work of Ricardo de Azambuja and Robert Kim

    Initialisation of the reservoir's topology
    """
    def __init__(self, N_r, N_i, r_net_shape, i_net_shape, w_in, w_out, distribution, p_inh, refractory_time, connections_parameters, apply_dale, lbd=1.2):
        """
        Reservoir initialisation method
        N_r: number of internal units (neurons)
        N_i : number of input units
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
        self.N_r = N_r  
        self.N_i = N_i                                                        
        self.distribution = distribution
        self.p_inh = p_inh
        self.refractory_time = refractory_time
        self.lbd = lbd
        self.connections_parameters = connections_parameters
        self.w_in = w_in
        self.w_out = w_out
        self.apply_dale = apply_dale
        self.r_net_shape = r_net_shape
        self.i_net_shape = i_net_shape
        self.inh, self.exc, self.n_inh, self.n_exc = self.assign_exc_inh(N_r)
        self.inh_i, self.exc_i, self.n_inh_i, self.n_exc_i = self.assign_exc_inh(N_i)
        self.i_r_topology = self.input_reservoir_topology()
        self.i_topology = self.internal_topology()
        self.input_reservoir_synapses = self.generate_synapses(self.i_r_topology) # list of synapsesa
        self.internal_synapses = self.generate_synapses(self.i_topology)
        self.input_neurons = self.generate_neurons(N_i)
        self.internal_neurons = self.generate_neurons(N_r)


    def assign_exc_inh(self, N):

        if self.apply_dale:
            inh = np.random.rand(N,1) < self.p_inh
            exc = ~inh
            
        else:
            inh = np.random.rand(N,1) < 0
            exc = ~inh

        n_inh = len(np.where(inh==True)[0])
        n_exc = N - n_inh

        return inh, exc, n_inh, n_exc

    def p_connection(self, C, distance, lbd):
        """
        Compute the probability of the connection between two neurons

        Returns
            p: probability of the connection between two neurons

        """
        p = C * np.exp(-np.power((distance/lbd), 2))
        return p

    def generate_synapses(self, topology):
        print(f"Generate synapses : Begin...")
        connections_exc = topology['exc']
        connections_inh = topology['inh']
        synapse_list = []

        for inh_infos in connections_inh:
            s = Synapse(inh_infos, 0)
            synapse_list.append(s)
        for exc_infos in connections_exc:
            s = Synapse(exc_infos, 0)
            synapse_list.append(s)

        print(f"Generate synapses : Done!")
        return synapse_list

    def generate_neurons(self, N):
        print(f'Generate neurons : Begin...')
        neurons_list = []
        for neuron in range(N):
            neurons_list.append(LIF(0.4, 0, 1e-3, 3, 5.1, 5e-3, 0.3))
        print(f'Generate neurons : Done!')
        return neurons_list

    def compute(self, T, dt):
        #on boucle sur dt
        # besoind 'une fonction qui update le potential de chaque neuron et qui recrod qui spike a qqui
        # fonction qui simule chaque synapse
        # list synape input-reservoir
        # list internal synapse
        return None



    def mapping_reservoir(self):
        """
        Maps the internal neurons according to the Cortical column architecture (self.r_net_shape)

        Returns:
            array: array of triplet (pos_x, pos_y, pos_z)
        """
        print('Mapping : Begin...')
        n_x, n_y, n_z = self.r_net_shape

        #Center axis except the axis representating the depth of the columns (here y axis)
        dx = -n_x/2.0
        dy = self.i_net_shape[1]
        dz = -n_z/2.0

        positions_list = np.array([(x+dx, y+dy, z+dz) for x in range(n_x) for y in range(n_y) for z in range(n_z)])

        print('Mapping : Done!')
        return positions_list
        
    def mapping_input(self):    
        """
        Maps the input neurons according to the Cortical column architecture (self.i_net_shape)

        Returns:
            array: array of triplet (pos_x, pos_y, pos_z)
        """
        print('Mapping : Begin...')
        n_x, n_y, n_z = self.i_net_shape

        #Center axis except the axis representating the depth of the columns (here y axis)
        dx = -n_x/2.0
        dy = 0
        dz = -n_z/2.0

        positions_list = np.array([(x+dx, y+dy, z+dz) for x in range(n_x) for y in range(n_y) for z in range(n_z)])

        print('Mapping : Done!')
        return positions_list

    def input_reservoir_topology(self):
        print('Creation of the input-reservoir topology: Begin...')
   
        #list of index of inhibitory and excitatory neurons
        inh_index_i = np.where(self.inh_i == True)[0]
        exc_index_i = np.where(self.exc_i == True)[0]

        inh_index_r = np.where(self.inh == True)[0]
        exc_index_r = np.where(self.exc == True)[0]
        #list of the positions of the neurons 
        positions_list_r = self.mapping_reservoir()
        positions_list_i = self.mapping_input() 
        #creation of the list containing infos about each connections
        connections_inh = []
        connections_exc = []

        for i in range(self.N_r):
            
            for j in range(self.N_i):
                if i in inh_index_r:
                    sign = -1
                    if j in inh_index_i:
                        ## (II)
                        (t_pre, t_pos)=(0,0)
                    else:
                        # (IE)
                        (t_pre, t_pos) = (0,1)
                
                else:
                    sign = 1
                    if j in inh_index_i:
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

            
                p_connection = self.p_connection(CGupta, euclidean_distance(positions_list_i[j], positions_list_r[i]), self.lbd)
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
        
        return {'exc':connections_exc, 'inh':connections_inh, 'pos_i':positions_list_i, 'pos_r':positions_list_r}

    def internal_topology(self):
        """
        Create the topology of the reservoir and generates the connections of the reservoir
        """
        print('Creation of the reservoir topology : Begin...')
   
        #list of index of inhibitory and excitatory neurons
        inh_index = np.where(self.inh == True)[0]
        exc_index = np.where(self.exc == True)[0]

        #list of the positions of the neurons 
        positions_list = self.mapping_reservoir()
        
        #creation of the list containing infos about each connections
        connections_inh = []
        connections_exc = []


        for i in range(self.N_r):
            
            for j in range(self.N_r):
                if i!=j:
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
        



    def plot_input_reservoir(self, topology):
        print('Plotting the input-reservoir : Begin...')
        fig = plt.figure(figsize=(12,8))
        fig.suptitle('Reservoir architecture (Cortical Column)')
        
        inh_i = self.inh_i
        inh_r = self.inh
        pos_i = topology['pos_i']
        pos_r = topology['pos_r']
        print(f'len pos_i : {len(pos_i)} et len pos_r : {len(pos_r)}')
        x_i = [p[0] for p in pos_i]
        y_i = [p[1] for p in pos_i]
        z_i = [p[2] for p in pos_i]
        x_r = [p[0] for p in pos_r]
        y_r = [p[1] for p in pos_r]
        z_r = [p[2] for p in pos_r]
        c_i = ['blue' if i else 'red' for i in inh_i]
        c_r = ['blue' if i else 'red' for i in inh_r]
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(x_i,y_i,z_i , color='red')
        ax1.scatter(x_r,y_r,z_r , color='green', marker='x')
        ax1.set_title('Mapping of the neurons in the space')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        
        """
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
        """
        plt.show()
        print('Plotting the Reservoir : Done!')

    def plot_lsm_topology(self, input_reservoir_topology, internal_topology):
        print('Plotting the Reservoir : Begin...')
        fig = plt.figure(figsize=(12,8))
        fig.suptitle('LSM architecture (Cortical Column)')
        
        inh_r = self.inh
        
        # INPUT RESERVOIR
        inh_i = self.inh_i
        pos_i = input_reservoir_topology['pos_i']
        x_i = [p[0] for p in pos_i]
        y_i = [p[1] for p in pos_i]
        z_i = [p[2] for p in pos_i]
        c_i = ['blue' if i else 'red' for i in inh_i]
        
        pos_r = input_reservoir_topology['pos_r']
        x_r = [p[0] for p in pos_r]
        y_r = [p[1] for p in pos_r]
        z_r = [p[2] for p in pos_r]
        c_r = ['blue' if i else 'red' for i in inh_r]
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')

        ax1.scatter(x_r,y_r,z_r , color=c_r, marker='x', alpha=1)
        ax1.scatter(x_i,y_i,z_i , color=c_i, alpha=1)
        ax1.set_title('Mapping of the input-reservoir neurons into the space')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()


        #INTERNAL===================================

        pos = internal_topology['pos']
        x = [p[0] for p in pos]
        y = [p[1] for p in pos]
        z = [p[2] for p in pos]
        c = ['blue' if i else 'red' for i in inh_r]

        #RESERVOIR
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(x,y,z , color=c, alpha=1)
        ax2.set_title('Mapping of the internal neurons into the space')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.legend()
#==================================        

        I_connections = input_reservoir_topology['inh']
        i_to = [c[0] for c in I_connections] # list of tuple(i,j)
        E_connections = input_reservoir_topology['exc']
        e_to = [c[0] for c in E_connections] # list of tuple(i,j)
        print("DEBUG")
        print(f"len de I : {len(I_connections)}")
        print(f"len de pos_r")
        print(input_reservoir_topology['pos_r'][:5])
        inh_positions_pre = [input_reservoir_topology['pos_r'][i[0]] for i in i_to] # [(x,y,z)]
        inh_positions_post = [input_reservoir_topology['pos_i'][i[1]]for i in i_to] # [(x,y,z)]
        exc_positions_pre = [input_reservoir_topology['pos_r'][i[0]] for i in e_to]
        exc_positions_post = [input_reservoir_topology['pos_r'][i[1]] for i in e_to]
         
        n_exc = len(exc_positions_pre)
        n_inh = len(inh_positions_pre)
        print("okokkokok", n_exc, n_inh)

        ax3= fig.add_subplot(2,2,3, projection='3d')
        for i in range(n_inh):
            ax3.plot([inh_positions_pre[i][0], inh_positions_post[i][0]],
                     [inh_positions_pre[i][1], inh_positions_post[i][1]],
                     [inh_positions_pre[i][2], inh_positions_post[i][2]],
                     color='blue')
        for i in range(n_exc):
            ax3.plot([exc_positions_pre[i][0], exc_positions_post[i][0]],
                     [exc_positions_pre[i][1], exc_positions_post[i][1]],
                     [exc_positions_pre[i][2], exc_positions_post[i][2]],
                     color='red')


        ax3.scatter(x_r,y_r,z_r, color='green', marker='x', alpha=1)
        ax3.scatter(x_i,y_i,z_i, color='green', alpha=1)
        
        """
        #FOR I to
        x1 = []
        y1 = []
        z1 = []
        for c in e_to:
            #(i,j)
            i,j = c
            x1.append(inh_positions_post[i][0])
            x1.append(inh_positions_post[j][0])
            y1.append(inh_positions_post[i][1])
            y1.append(inh_positions_post[j][1])
            z1.append(inh_positions_post[i][2])
            z1.append(inh_positions_post[j][2])
        x2 = []
        y2 = []
        z2= []
        for c in i_to:
            i,j = c
            x2.append(inh_positions_pre[i][0])
            x2.append(inh_positions_pre[j][0])
            y2.append(inh_positions_pre[i][1])
            y2.append(inh_positions_pre[j][1])
            z2.append(inh_positions_pre[i][2])
            z2.append(inh_positions_pre[j][2])

        ax3.plot(x1,y1,z1, color='red', lw=0.6)
        ax3.plot(x2, y2, z2, color='blue', lw=.6)
        ax3.scatter(x, y, z, alpha=1, color='green')
        """ 
        ax3.set_title('Input-reservoir synapse connections')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')



    #================= INTERNAL ================
        I_connections = internal_topology['inh']
        i_to = [c[0] for c in I_connections] # list of tuple(i,j)
        E_connections = internal_topology['exc']
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


        ax4= fig.add_subplot(2,2,4, projection='3d')
        ax4.plot(x1,y1,z1, color='blue', lw=0.6)
        ax4.plot(x2, y2, z2, color='red', lw=.6)
        ax4.scatter(x, y, z, alpha=1, color='green')
        ax4.set_title('Internal synapse connections ')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('z')
        plt.savefig('results/lsm_architecture.png', bbox_inches='tight')
 
        plt.show()
        print('Plotting the Reservoir : Done!')

        
        

    def paramaters(self):
        print(f'\n----------------------------------------------\n')
        print(f'Model\'s parameters :')
        print(f'\t N_r : {self.N_r}')
        print(f'\t N_i : {self.N_i}')
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

