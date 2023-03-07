from math import sqrt
from numpy import zeros


def COSTW(x, y):
    """
    Cosine maximisation time warp (COSTW)

    Inputs:
        - `x (Tx,)` `y (Ty,)`: time series (column vectors)

    Return:
        -`costw`: Cosine maximisation time warp value
        -`delta`: alignement
    """

    def f(x_dot_y, norm_x_2, norm_y_2):
        return x_dot_y / sqrt(norm_x_2 * norm_y_2)
    
    Tx = x.shape[0]
    Ty = y.shape[0]

    M = zeros((Tx, Ty))
    N = zeros((Tx, Ty))
    P = zeros((Tx, Ty))
    flags = zeros((Tx, Ty), dtype=int)

    M[0,0] = x[0]*y[0]
    N[0,0] = x[0]**2
    P[0,0] = y[0]**2

    for i in range(1, Tx):
        M[i, 0] = M[i-1, 1] + x[i]*y[0]
        N[i, 0] = N[i-1, 1] + x[i]**2
        P[i, 0] = P[i-1, 1] + y[0]**2
        flags[i, 0] = -1
    
    for j in range(1, Ty):
        M[0, j] = M[1, j-1] + x[0]*y[j]
        N[0, j] = N[1, j-1] + x[0]**2
        P[0, j] = P[1, j-1] + y[j]**2
        flags[0, j] = 1
    
    for j in range(1, Ty):
        for i in range(1, Tx):
            xiyj = x[i]*y[j]
            xi2 = x[i]**2
            yj2 = y[j]**2

            best_M = M[i-1, j-1] + xiyj
            best_N = N[i-1, j-1] + xi2
            best_P = P[i-1, j-1] + yj2
            best_f = f(best_M, best_N, best_P)
            best_flag = 0

            if f(M[i, j-1] + xiyj, N[i, j-1] + xi2, P[i, j-1] + yj2) > best_f:
                best_M = M[i, j-1] + xiyj
                best_N = N[i, j-1] + xi2
                best_P = P[i, j-1] + yj2
                best_f = f(best_M, best_N, best_P)
                best_flag = 1
            
            if f(M[i-1, j] + xiyj, N[i-1, j] + xi2, P[i-1, j] + yj2) > best_f:
                best_M = M[i-1, j] + xiyj
                best_N = N[i-1, j] + xi2
                best_P = P[i-1, j] + yj2
                best_f = f(best_M, best_N, best_P)
                best_flag = -1

            M[i, j] = best_M
            N[i, j] = best_N
            P[i, j] = best_P
            flags[i, j] = best_flag
    
    costw = M[-1, -1]
    delta = zeros((Tx, Ty))
    delta[-1, -1] = 1

    i, j = Tx - 1, Ty - 1

    while (i, j) != (0, 0):
        if flags[i,j] == 0:
            i, j = i-1, j-1
        elif flags[i,j] == 1:
            j = j-1
        else:
            i = i-1
        delta[i, j] = 1
    
    return costw, delta
