from math import sqrt
from numpy import zeros, array


def COSTW(x, y, r_window = None):
    """
    Cosine maximisation time warp (COSTW)

    Inputs:
        - `x (Tx,)` `y (Ty,)`: time series (column vectors)

    Return:
        -`costw`: Cosine maximisation time warp value
        -`delta`: alignement
    """
    Tx = x.shape[0]
    Ty = y.shape[0]

    def f(M_t):
        return M_t[0] / (sqrt(M_t[1] * M_t[2]) + 1e-6)

    if r_window is None:
        def inWindow(i,j):
            return True
    else:
        def inWindow(i,j):
            return abs(j * Tx / Ty - i) <= r_window
    

    M = zeros((Tx, Ty, 3))
    flags = zeros((Tx, Ty), dtype=int)

    M[0,0,0] = x[0]*y[0]
    M[0,0,1] = x[0]**2
    M[0,0,2] = y[0]**2

    for i in range(1, Tx):
        if inWindow(i, 0):
            values = array([x[i]*y[0], x[i]**2, y[0]**2])
            M[i, 0] = M[i-1, 1] + values
            flags[i, 0] = -1
    
    for j in range(1, Ty):
        if inWindow(0, j):
            values = array([x[0]*y[j], x[0]**2, y[j]**2])
            M[0, j] = M[1, j-1] + values
            flags[0, j] = 1
    
    for j in range(1, Ty):
        for i in range(1, Tx):
            if inWindow(i,j):
                values = array([x[i]*y[j], x[i]**2, y[j]**2])

                if inWindow(i-1, j-1):
                    M[i, j] = M[i-1, j-1] + values
                    best_f = f(M[i,j])
                    flags[i, j]  = 0
                else:
                    best_f = -2.

                if inWindow(i, j-1):
                    test_f = f(M[i, j-1] + values)
                    if test_f > best_f:
                        M[i, j] = M[i, j-1] + values
                        best_f = test_f
                        flags[i, j]  = 1
                
                if inWindow(i-1, j):
                    test_f = f(M[i-1, j] + values)
                    if  test_f > best_f:
                        M[i, j] = M[i-1, j] + values
                        best_f = test_f
                        flags[i, j]  = -1

    
    costw = f(M[-1, -1])
    delta = zeros((Tx, Ty), dtype=int)
    delta[-1, -1] = 1

    i, j = Tx - 1, Ty - 1

    while (i, j) != (0, 0):
        if flags[i,j] == 0:
            i, j = i-1, j-1
        elif flags[i,j] == 1:
            delta[i,j] = 0
            j = j-1
        else:
            i = i-1
        delta[i, j] = 1
    
    return costw, delta
