from math import sqrt
from numpy import zeros, array
import numpy as np


def COSTW(x, y, r_window = None):
    """
    Cosine maximisation time warp (COSTW)

    Inputs:
        - `x (Tx,)` `y (Ty,)`: time series (column vectors)

    Return:
        -`costw`: Cosine maximisation time warp value
        -`delta`: alignement
    """

    Tx = x.shape[0] # Length of X
    Ty = y.shape[0] # Length of y

    def f(M_t):
        return M_t[0] / (sqrt(M_t[1] * M_t[2]) + 1e-6)


    if r_window is None:
        def inWindow(i,j):
            return True
    else:
        def inWindow(i,j):
            return abs(j * Tx / Ty - i) <= r_window
    

    M = zeros((Tx, Ty, 3)) # Initialize matrix M 


    flags = zeros((Tx, Ty), dtype=int) # TODO : Checl 
    flags_list = [-1,1,0] #CF. array([M[i-1,j]+values, M[i,j-1]+values,M[i-1,j-1]+values])

    # iterating this way ensures that we have computed the 
    # (maximum of 3) predecessor values
    for c in range(Tx + Ty-1): # Longueur de la diagonale
        # i is bounded by c
        for i in range(c+1):
            # then j is deduced from c and i
            j = c - i
            # clip within the cost matrix domain
            if 0 <= i < Tx and 0 <= j < Ty and inWindow(i,j):
                if i == 0 and j == 0:  # upper left corner
                    M[i,j] = array([x[0]*y[0],x[0]**2,y[0]**2])
                    flags[i, j] = 0

                elif j == 0:           # on one edge, only one insertion
                    values = array([x[i]*y[j], x[i]**2, y[j]**2])
                    M[i, j] = M[i-1, j] + values
                    flags[i, j] = -1
                    

                elif i == 0:           # the other edge, ditto
                    values = array([x[i]*y[j], x[i]**2, y[j]**2])
                    M[i, j] = M[i, j-1] + values
                    flags[i, j] = 1

                else:                  # in the middle
                    values = array([x[i]*y[j], x[i]**2, y[j]**2])

                    new_values = array([M[i-1,j]+values, M[i,j-1]+values,M[i-1,j-1]+values])
                    is_inwindow = [inWindow(k,l) for (k,l) in [(i-1,j),(i,j-1),(i-1,j-1)]]
                    idx = np.argmax([f(m) if in_wind else -np.inf for m, in_wind in zip(new_values, is_inwindow) ])

                    # Assign values
                    M[i,j] = new_values[idx]
                    flags[i, j] = flags_list[idx]
    
    costw = f(M[-1, -1])
    delta = zeros((Tx, Ty), dtype=int)
    delta[-1, -1] = 1

    i, j = Tx - 1, Ty - 1

    while (i, j) != (0, 0):
        if flags[i,j] == 0:
            i, j = i-1, j-1
        elif flags[i,j] == 1:
            # delta[i,j] = 0
            j = j-1
        else:
            i = i-1
        delta[i, j] = 1
    
    row_sums = delta.sum(axis=1)
    delta_norm = delta / row_sums[:, np.newaxis]
    return costw, delta_norm

