
import numpy as np 
from tqdm import tqdm 
from twi_ksvd.omp import OMP, TWI_OMP


class kSVD():
    """Implementation of the kSVD dictionnary learning algorithm
    """
    def __init__(self, K = 10, epsilon  = 1e-3, max_iter = 30) -> None:
        self.K = K  #number of atoms 
        self.epsilon = epsilon
        self.max_iter = max_iter

    def fit (self, X, D, tau:int):
        """Method that learns the dictionnary based on a set of data 
        Inputs: 
            - X (p,N) : input samples
            - D (p,K) : dictionnary of K atoms 
            - `tau`: Number of atoms chosen to represent `x` with atoms of `D`

        Returns:
            - A : learned alphas : np.array -> size (K,N)
            - D : learned dictionnary : np.arrat -> shape (p,K)
        """
        self.D = D
        self.N  = len(X[0]) # number of input samples
        # Initialize values 

        E_k_old = 0 
        E_k = np.inf 
        n_iter = 0
        while abs(np.linalg.norm(E_k) - np.linalg.norm(E_k_old)) > self.epsilon: # Stopping Criterion
            print("=============================================")
            print(f"iteration number : {n_iter}, eps : {abs(np.linalg.norm(E_k))}, delta_eps : {abs(np.linalg.norm(E_k) - np.linalg.norm(E_k_old))}")
            E_k_old = E_k  
            alphas = []            
            print("Compute Sparse Codes")
            # Compute sparse codes 

            for i in range(self.N):
                alphas.append(OMP(X[:,i],D,tau))
            self.A = np.vstack(alphas).T

            print("Update Dictionnary")
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
    """_summary_
    """
    def __init__(self, K = 10, epsilon  = 1e-3, max_iter = 30) -> None:
        """Init the TWI-kSVD algorithm 

        Args:
            K (int, optional): Number of atoms . Defaults to 10.
            epsilon (_type_, optional): Precision level. Defaults to 1e-3.
            max_iter (int, optional): Maximum number of iterations. Defaults to 30.
        """

        self.K = K  #number of atoms 
        self.epsilon = epsilon  # precision
        self.max_iter = max_iter # max number of iterations

    def rotation(self,a:np.array,b:np.array,c:np.array) -> np.array:
        """Method that implement the rotation of vectors

        Args:
            a (np.array): _description_
            b (np.array): _description_
            c (np.array): _description_

        Returns:
            np.array: Rotated vector
        """
        theta =  np.arccos(np.clip(np.dot(b/np.linalg.norm(b), c/np.linalg.norm(c)), -1.0, 1.0))
        u = b / np.linalg.norm(b)
        v = (c- np.dot(u,c)*u)/np.linalg.norm((c- np.dot(u,c)*u))
        c, s = np.cos(theta), np.sin(theta)
        R_theta = np.array(((c, -s), (s, c)))
        R = np.eye(len(u)) - np.outer(u,u) - np.outer(v,v) + np.vstack((u,v)).T@R_theta@np.vstack((u,v))
        return R@a
    
    def fit (self, X, D, tau):
        """Method that learn the dictionnary using the TWI-kSVD algorihtm

        Args:
            X (_type_): _description_
            D (_type_): _description_
            tau (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.D = D
        self.N  = len(X[0]) # number of input samples
        # Initialize values 

        E_old = 0 
        E = np.inf 
        n_iter = 0
        while abs(np.linalg.norm(E) - np.linalg.norm(E_old)) > self.epsilon: # Stopping Criterion
            print("=============================================")
            print(f"iteration number : {n_iter}, eps : {abs(np.linalg.norm(E))}, delta_eps : {abs(np.linalg.norm(E) - np.linalg.norm(E_old))}")
            E_old = E  
            # init
            self.alphas = []
            self.alignements = []
            print("     Step 1 | Compute Sparse Codes")
            # Compute sparse codes 

            for i in tqdm(range(self.N)):
                alpha, delta_ij= TWI_OMP(X[:,i],D,tau)
                self.alphas.append(alpha)
                self.alignements.append(delta_ij)
            self.siblings_atoms = [ [self.alignements[i][k]@self.D[k,:] for k in range(self.K)] for i in range(self.N)]
            self.alphas = np.vstack(self.alphas).T
            # print(self.alphas.shape)

            # Compute error
            E = np.array([ X[:,i] - np.sum([self.alignements[i][j]@self.D[j,:]*self.alphas[j,i] for j in np.arange(self.K)]) for i in range(self.N)])
            print(E.shape)
            print("     Step 2 | Update Dictionnary")
            # Update dictionnary
            for k in range(self.K):
                mask = np.ones(self.K, dtype=bool)
                mask[k] = False
                Omega_k = self.alphas[k,:] != 0

                rotated_residuals= []
                residuals = []
                # print(Omega_k.nonzero()[0])
                for i in Omega_k.nonzero()[0]:
                        e_i = X[:,i] - np.sum([self.alignements[i][j]@self.D[j,:]*self.alphas[j,i] for j in np.arange(self.K)[mask]])
                        residuals.append(e_i)
                        rotated_res = self.rotation(self.alignements[i][k].T@e_i, self.alignements[i][k].T@self.siblings_atoms[i][k], self.D[k,:])
                        # print(rotated_res.shape)
                        rotated_residuals.append(rotated_res)
                if sum(Omega_k) >= 2 :
                    u, s, vh = np.linalg.svd(np.vstack(rotated_residuals).T, full_matrices=True)
                    
                    #  Update values 
                    self.D[k,:] = u[0] 
                    
                    for index, i in  enumerate(Omega_k.nonzero()[0]):
                        inv_rot_u = self.alignements[i][k]@self.rotation(u[0], self.D[k,:],self.alignements[i][k].T@ self.siblings_atoms[i][k])
                        self.siblings_atoms[i][k] = inv_rot_u
                        self.alphas[k,i] = np.dot(residuals[index], inv_rot_u)/np.linalg.norm(inv_rot_u)
                

            
            

            
            if n_iter > self.max_iter :
                print(f"Maximum number of iteration reached : {self.max_iter}")
                break
            else: 
                n_iter +=1
        return self.alphas, self.D
    

if __name__ == '__main__':
    model = TWI_kSVD( 10)
    X = np.random.random((2048,4))
    D = np.random.random((2048,10)).T
    tau = 3
    model. fit(X,D,tau )
    print("tip")