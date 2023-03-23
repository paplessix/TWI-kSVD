from numpy import argmax, dot, stack, zeros
from numpy.linalg import norm, lstsq
from twi_ksvd.costw import COSTW


def OMP(x, D, tau):
    """
    Implementation of Orthogonal Matching Pursuit (OMP) algorithm

    Inputs:
        - `x (p,)`:time series
        - `D (p, K)`: Dictionnary of K atoms
        - `tau`: Number of atoms chosen to represent `x` with atoms of `D`
    
    Returns:
        - `alpha (K,)`: learned coefficients (x ~ D @ alpha)
    """

    p, K = D.shape

    res = x
    Omega = []
    D_Omega = []

    while len(Omega) < tau:
        best_cos = -1
        best_k = -1
        best_dk = -1.

        for j in range(K):
            if j in Omega:
                continue
            cos_sim = dot(res, D[:,j].T)/(norm(res)*norm(D[:,j].T))
            if cos_sim > best_cos:
                best_cos = cos_sim
                best_k = j
                best_dk = D[:,j]
        
        Omega.append(best_k)
        D_Omega.append(best_dk)

        D_partial = stack(D_Omega, axis=-1)

        alpha_partial = lstsq(D_partial, x)[0]

        res = x - D_partial.reshape((p, len(Omega))) @ alpha_partial

    alpha = zeros(K)

    for i, k in enumerate(Omega):
        alpha[k] = alpha_partial[i]
    return alpha

def TWI_OMP(x, D_list, tau, r_window = None):
    """
    Implementation of  Time Warping Invariant - Orthogonal Matching Pursuit (TWI-OMP) algorithm

    Inputs:
        - `x (p,)`:time series
        - `D_list`: list of K atoms (of lengths pj) (the dictionnary of K atoms)
        - `tau`: Number of atoms chosen to represent `x` with atoms of `D`
    
    Returns:
        - `alpha (K,)`: learned coefficients
        - `deltas`: a list of K binary matrixes of shape (p, pj) with pj the length of the j-th atom and p the length of x

    Use:
        - The approximation of x is \sum_j=1^K alpha[j] * deltas[j] @ D_list[j]
    """

    K = len(D_list)
    p = x.shape[0]

    res = x
    Omega = []
    S_Omega = []
    deltas_partial = []

    while len(Omega) < tau:
        best_cos = -1.1
        best_k = -1
        best_dk = -1.
        best_delta = -1.

        for j in range(K):
            if j in Omega:
                continue
            cos_sim, delta = COSTW(res, D_list[j], r_window=r_window)

            if cos_sim > best_cos:
                best_cos = cos_sim
                best_k = j
                best_delta = delta
                best_dk = delta @ D_list[j]
        
        Omega.append(best_k)
        S_Omega.append(best_dk)
        deltas_partial.append(best_delta)

        D_partial = stack(S_Omega, axis=-1)

        alpha_partial = lstsq(D_partial, x, rcond=None)[0]

        res = x - D_partial.reshape((p, len(Omega))) @ alpha_partial

    alpha = zeros(K)
    deltas = [0]*K

    for i, k in enumerate(Omega):

        alpha[k] = alpha_partial[i]
        deltas[k] = deltas_partial[i]
    
    return alpha, deltas