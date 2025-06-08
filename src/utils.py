import numpy as np 

def euclidean_distance(p1, p2):
    return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2) + np.power(p1[2]-p2[2], 2))

def isFired(U, U_reset, threshold):
    if U > threshold:
        return U_reset, True
    else:
        return U, False