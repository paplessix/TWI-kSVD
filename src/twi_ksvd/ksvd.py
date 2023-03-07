
import numpy as np 
from twi_ksvd.omp import OMP


class kSVD():
    def __init__(self, K = 10, epsilon  = 1e-3) -> None:
        self.K = K  #number of atoms 
        self.epsilon = epsilon


    def fit (self, X, D, tau):
        """
        
        """
        max_iter = 100
        n=0
        N  = len(X) # number of input samples
        alphas = []

        # Initialize values 

        E_k_old = 0 
        E_k = np.inf 

        while abs(np.linalg.norm(E_k) - np.linalg.norm(E_k_old)) < self.epsilon : # Stopping Criterion
            
            # Compute sparse codes 

            for i in range(N):
                alphas.append(OMP(X[i],D,tau))
            A = np.vstack(alphas)

            # Update dictionnary 

            for k in range(self.K):
                E_k = X - D.delete(k).T@A.delete(k) #TODO: Check if it is really doing what it needs to do 
                Omega_k = np.where(A[k] != 0)[0]
                E_k_restricted = E_k[Omega_k]
                u, s, vh = np.linalg.svd(E_k_restricted, full_matrices=True)
                
                #Update values 

                D[k,:] = u[0] 
                A[k: ] = s[0]*vh[0]
            
            n+=1 
        return A, D
