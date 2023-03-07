from numpy import dot, stack, zeros
from numpy.linalg import norm, lstsq

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