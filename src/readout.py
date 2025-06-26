import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# here the input signal have already been through the reservoir
# we have a liquid -> (n_step, n_liquid)
# for each time step, we project the state : (n_liquid, 1) -> (n, 1) through W (n ,n_liquid)
# filter = W @ liquid_state
# we do this for each time step
# we stock every filter response into a matric X (n, n_steps)
# we do linear regressoion Y_target = W @ X
class Readout():
# at first we dont ignore the inital transient 
    def __init__(self, N_readout, lsm, n_steps):
        self.N_readout = N_readout
        self.lsm = lsm
        self.n_steps = n_steps
        self.Wreadout = np.random.randn(N_readout, lsm.N_liquid)


    def output(self, liquid_state):
        """
        Projects a liquid state into a readout reponse (n_liquid,1) -> (n_readout, 1)
        """
        return self.W @ liquid_state # (n_redout, 1)

    def predict(self, signal, steps, Ic):
        u = signal[-1] # we take u(T) to predict y(T+1)
        prediction = np.zeros(steps, self.N_readout)
        for step in range(steps):
            liquid_state = self.lsm.forward(u, Ic)
            y = self.output(liquid_state=liquid_state)
            prediction[step] = y.copy()
            u = y
        


    def train(self, liquid, y_train):
        R = np.zeros((self.N, self.n_steps)) #each column is a state at time t
        for step in range(self.n_steps):
            output = self.output(liquid[step]) # (N, 1)
            R[:, step] = output   

        self.Wreadout = self.solution(R, y_train)    

    
    def solution(self, X, y_train): # good W , y_target shape (n_features, n_timestep)
         # on fait la regresion linaire 
            # donc on passe une matrice (N_liquid, N_step) #est ce qu'on skip les inital transient ?
        regressor = LinearRegression().fit(X, y_train)
