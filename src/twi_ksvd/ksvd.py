
import numpy as np 
from twi_ksvd.omp import OMP, TWI_OMP


class kSVD():
    """Implementation of the kSVD dictionnary learning method 
    """
    def __init__(self, K = 10, epsilon  = 1e-3, max_iter = 30) -> None:
        self.K = K  #number of atoms 
        self.epsilon = epsilon
        self.max_iter = max_iter

    def fit (self, X, D, tau):
        """Method that learns the dictionnary based on a set of data 
        Inputs: 
            - X (p,N) : input samples
            - D (p,K) : dictionnary of K atoms 
            - `tau`: Number of atoms chosen to represent `x` with atoms of `D`

        Returns:         
        """
        self.D = D
        self.N  = len(X[0]) # number of input samples
        # Initialize values 

        E_k_old = 0 
        E_k = np.inf 
        n_iter = 0
        while abs(np.linalg.norm(E_k) - np.linalg.norm(E_k_old)) > self.epsilon: # Stopping Criterion
            
            print(f"iteration number : {n_iter}, eps : {abs(np.linalg.norm(E_k))}, delta_eps : {abs(np.linalg.norm(E_k) - np.linalg.norm(E_k_old))}")
            E_k_old = E_k  
            alphas = []            
            
            # Compute sparse codes 

            for i in range(self.N):
                alphas.append(OMP(X[:,i],D,tau))
            self.A = np.vstack(alphas).T

            # Update dictionnary 

            for k in range(self.K):
                mask = np.ones(len(self.D[0]), dtype=bool)
                mask[k] = False
                # E_k = X - np.einsum('ij,jk->ik',self.D[:,mask], self.A[mask,:]) #TODO: Check if it is really doing what it needs to do 
                
                E_k = X - self.D[:,mask]@ self.A[mask,:]
                Omega_k = self.A[k,:] != 0
                # print(Omega_k.shape)
                if sum(Omega_k) != 0 :
                    E_k_restricted = E_k[:,Omega_k]
                    u, s, vh = np.linalg.svd(E_k_restricted, full_matrices=True)
                    #  Update values 

                    self.D[:,k] = u[0] 
                    self.A[k,Omega_k] = s[0]*vh[0]
            if n_iter > self.max_iter :
                print(f"Maximum number of iteration reached : {self.max_iter}")
                break
            else: 
                n_iter +=1
        return self.A, self.D




class TWI_kSVD():
    """
    """
    def __init__(self, K = 10, epsilon  = 1e-3, max_iter = 30) -> None:
        """
        """
        self.K = K  #number of atoms 
        self.epsilon = epsilon  # precision
        self.max_iter = max_iter # max number of iterations

    def rotation(a,b,c):
        theta =  np.arccos(np.clip(np.dot(b/np.linalg.norm(b), c/np.linalg.norm(c)), -1.0, 1.0))
        u = b / np.linalg.norm(b)
        v = (c- np.dot(u,c)*u)/np.linalg.norm((c- np.dot(u,c)*u))
        c, s = np.cos(theta), np.sin(theta)
        R_theta = np.array(((c, -s), (s, c)))
        R = np.eye(len(u)) - np.outer(u,u) - np.outer(v,v) + np.vstack((u,v))@R_theta@np.vstack((u,v)).T

    def fit (self, X, D, tau):
        """
        """
        self.D = D
        self.N  = len(X[0]) # number of input samples
        # Initialize values 

        E_k_old = 0 
        E_k = np.inf 
        n_iter = 0
        while abs(np.linalg.norm(E_k) - np.linalg.norm(E_k_old)) > self.epsilon: # Stopping Criterion
            print(f"iteration number : {n_iter}, eps : {abs(np.linalg.norm(E_k))}, delta_eps : {abs(np.linalg.norm(E_k) - np.linalg.norm(E_k_old))}")
            E_k_old = E_k  
            # init
            alphas = []
            alignements = []

            # Compute sparse codes 

            for i in range(self.N):
                alpha, delta_ij= TWI_OMP(X[:,i],D,tau)
                alphas.append(alpha)
                alignements.append(delta_ij)
            self.A = np.vstack(alphas).T            
            
            for k in range(self.K):
                mask = np.ones(len(self.D[0]), dtype=bool)
                mask[k] = False
                Omega_k = self.A[k,:] != 0
                residuals= []
                for i, boolean in enumerate(Omega_k):
                    if boolean :
                        e_i = X[:,i] - np.sum([alignements[i][:,j]*self.D[:,j]*self.A[j,i] for j in np.arange(self.K)[mask]])
                        rotated_res = self.rotation(alignements[i].T@e_i, alignements[i].T@alignements[k]*D[:,k], D[:,k])
                        residuals.append(rotated_res)
                u, s, vh = np.linalg.svd(np.concatenate(residuals), full_matrices=True)
                

            
            
            
            
            if n_iter > self.max_iter :
                print(f"Maximum number of iteration reached : {self.max_iter}")
                break
            else: 
                n_iter +=1
        return self.A, self.D
    

if __name__ == '__main__':
    model = TWI_kSVD( 10)
    X = np.random.random((100,4))
    D = np.random.random((100,10))
    tau = 3
    model. fit(X,D,tau )
    print("tip")