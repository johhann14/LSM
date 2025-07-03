"""
File: readout.py
Author: Johann Ly 
Mail: johann.ly@ecole.ensicaen.fr
Date: 2025-06-15

Class for the linear regression from the liquid trace

References:
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from utils import encode2

# here the input signal have already been through the reservoir
# we have a liquid -> (n_step, n_liquid)
# for each time step, we project the state : (n_liquid, 1) -> (n, 1) through W (n ,n_liquid)
# filter = W @ liquid_state
# we do this for each time step
# we stock every filter response into a matric X (n, n_steps)
# we do linear regressoion Y_target = W @ X
class Readout():
    def __init__(self, N_readout, lsm, n_steps):
        self.N_readout = N_readout
        self.lsm = lsm
        self.n_steps = n_steps
        self.Wout = None
        print(f"Readout builded")


    def filter(self, liquid_state):
        # liquid : (1, n_liquid)
        # difference between pattern classificaiton and prediction
        # for pattern, a new example is given to the liquid for during T
        #for predition we give just input for a time t

        pass

    def output(self, liquid_state):
        """
        Projects a liquid state into a readout reponse (n_liquid,1) -> (n_readout, 1)
        """
        l = liquid_state[None, :]
        out = self.Wout.predict(l)
        return out[0] # (n_sample, n_feture)a
        # predict return (n_samples,)

    def predict(self, signal, steps, Ic, n_input, noise_bool, mu=None, sigma=None):
        print(f"Predict():...")
        u = encode2(signal, n_input) # we take u(T) to predict y(T+1)
        prediction = np.zeros((steps, self.N_readout))
        for step in range(steps):
            self.lsm.forward(u, Ic, noise_bool=noise_bool)
            liquid_state = self.lsm.liquid_neurons.U.copy()
            #liquid_state = (liquid_state - mu) / sigma
            y = self.output(liquid_state=liquid_state)
            prediction[step] = y
            u = encode2(y, n_input)

        print(f"Predict: Done!")
        return prediction
        


    def train(self, liquid, y_train): # on donne lif.U_trace (step, nliquid), 
                                    # ca veut dire on passe tout u_train dans le liquid et on recupere la trace
        print(f"Train():...")
        """
        R = np.zeros((self.lsm.N_liquid, self.n_steps)) #each column is a state at time t
        for step in range(self.n_steps):
            R[:, step] = liquid[step]
        """

        self.Wout = self.solution(liquid, y_train)    
        print(f"Train(): Done!")

    
    def solution(self, X, y_train): # good W , y_target shape (n_features, n_timestep)
         # on fait la regresion linaire 
            # donc on passe une matrice (N_liquid, N_step) #est ce qu'on skip les inital transient ?
        regressor = LinearRegression().fit(X, y_train)
        #regressor = Ridge(alpha=0.5).fit(X, y_train)
        return regressor
    

        
