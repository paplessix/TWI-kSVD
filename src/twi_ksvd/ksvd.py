
import numpy as np 
from twi_ksvd.omp import OMP


class kSVD():
    def __init__(self, K = 10, epsilon  = 1e-3, max_iter = 1000) -> None:
        self.K = K  #number of atoms 
        self.epsilon = epsilon
        self.max_iter = max_iter

    def fit (self, X, D, tau):
        """
        
        """
        N  = len(X[0]) # number of input samples
        alphas = []

        # Initialize values 

        E_k_old = 0 
        E_k = np.inf 
        n_iter = 0
        while abs(np.linalg.norm(E_k) - np.linalg.norm(E_k_old)) > self.epsilon: # Stopping Criterion
            
            # Compute sparse codes 

            for i in range(N):
                # alphas.append(OMP(X[i],D,tau))
                alphas.append( np.ones((self.K,1)))
            A = np.hstack(alphas)

            # Update dictionnary 

            for k in range(self.K):
                mask = np.ones(len(D), dtype=bool)
                mask[k] = False
                print(X.shape)
                print(D.shape)
                print(A.shape)
                E_k = X - D[mask].T@A[mask] #TODO: Check if it is really doing what it needs to do 
                Omega_k = np.where(A[k] != 0)[0]
                E_k_restricted = E_k[Omega_k]
                u, s, vh = np.linalg.svd(E_k_restricted, full_matrices=True)
                
                #Update values 

                D[k,:] = u[0] 
                A[k: ] = s[0]*vh[0]

            if n_iter > self.max_iter :
                print(f"Maximum number of iteration reached : {self.max_iter}")
                break
            else: 
                n_iter +=1
        return A, D

if __name__ == '__main__':
    model = kSVD( 10)
    X = np.zeros((3,10))
    D = np.zeros((3,10))
    tau = 3
    model. fit(X,D,tau )
    print("tip")
