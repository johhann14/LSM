import numpy as np 

class PoissonSpikeGenerator:
    """
    Poisson Spike Generator based on : https://medium.com/@baxterbarlow/poisson-spike-generators-stochastic-theory-to-python-code-a76f8cc7cc32
    Baxter Barlow
    """
    def __init__(self, dt, T):
        """
        dt: time step
        T: duration
        n_steps: number of steps
        """
        self.dt = dt
        self.T = T
        self.n_steps = int(T/dt)
        print(f'Generator object created!')



    def generate(self, rate):
        spike_times = []
        t=0

        while t<self.T:
            interval = -np.log(np.random.rand()) / rate
            t+= interval
            if t<self.T:
                spike_times.append(t)

        range_t = np.linspace(0, self.T, self.n_steps) # generate the discrete times
        spike_inputs = np.digitize(spike_times, range_t) # maps continous to bins 
        return spike_inputs
   

    def parameters(self):
        print(f'\n----------------------------------------------\n')
        print(f'Generator\'s parameters :')
        print(f'\t dt : {self.dt}')
        print(f'\t T : {self.T}')
        print(f'\t n_steps : {self.n_steps}')
        print(f'\n----------------------------------------------\n')