import os 
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd 

from twi_ksvd.omp import TWI_OMP


class LetterClassifier():
    def __init__(self) -> None:
        self.x_dico = {}
        self.y_dico = {}
        self.selected_letters = None
    
    @property
    def letters(self) -> list: # list of letters in the dictionnary
        return list(self.x_dico.keys())

    @property
    def concatenated_x_dico(self) -> list: # concatenation of all x_dico atoms
        return [atom for letter, atoms in self.x_dico.items() for atom in atoms if letter in self.selected_letters]   

    @property
    def concatenated_y_dico(self) -> list: # concatetnation of all y_dico atoms
        return [atom for letter,atoms in self.y_dico.items() for atom in atoms if letter in self.selected_letters]
    
    @property
    def count_atoms_x_dico(self) -> int: # number of atoms in x_dico per letter
        return {letter : len(atoms) for letter, atoms in self.x_dico.items() if letter in self.selected_letters}
    
    @property
    def count_atoms_y_dico(self) -> int:  # number of atoms in y_dico per letter
        return {letter : len(atoms) for letter, atoms in self.y_dico.items() if letter in self.selected_letters}
    
    @property
    def map_letter_to_atoms_x_dico(self) -> dict: #letter corresponding to the atom in concatenated_x_dico
        result = []
        for letter, numbr in self.count_atoms_x_dico.items():
            result += [letter] * numbr
        return result
    
    @property
    def map_letter_to_atoms_y_dico(self) -> dict: # letter corresponding to the atom in concatenated_y_dico
        result = []
        for letter, numbr in self.count_atoms_y_dico.items():
            result += [letter] * numbr
        return result 

    def load_data(self, folder_path: str) -> None:
        """Function to load data from folder_path
        Args:
            folder_path (str): folder containing all atoms
        """
        dirs = os.listdir( folder_path )

        for folder in dirs:
            x_dico_folder_path=  os.path.join(os.path.join(folder_path, folder),"x_dico" )
            y_dico_folder_path= os.path.join(os.path.join(folder_path, folder),"y_dico" )
                                            
            # extract atoms from x_dico and y_dico
            dic_x = []                            
            for file in os.listdir(x_dico_folder_path):
                atom = np.load(os.path.join(x_dico_folder_path,file))
                dic_x.append(atom)

            dic_y = []                            
            for file in os.listdir(y_dico_folder_path):
                atom = np.load(os.path.join(y_dico_folder_path,file))
                dic_y.append(atom)

            
            self.x_dico[folder] = dic_x
            self.y_dico[folder]= dic_y

    def fit (self, signal_x, signal_y, tau, r_window = None) -> None:
        """Function that compute sparse representation of two signals using dictionnaries over x an y

        Args:
            signal_x (_type_): a signal over x axis
            signal_y (_type_): a signal over y axis
            tau (_type_): number of atoms for sparse representation
            r_window (_type_, optional): Sakoe Chiba band width. Defaults to None.

        Returns:
            tuple: alphas_x, deltas_x, alphas_y, deltas_y coefficients and time warp matrix over axis x and y
        """
        
        alphas_x, deltas_x = TWI_OMP(signal_x, self.concatenated_x_dico, tau=tau, r_window=r_window)
        alphas_y, deltas_y = TWI_OMP(signal_y, self.concatenated_y_dico, tau=tau, r_window=r_window)

        
        return alphas_x, deltas_x, alphas_y, deltas_y
    

    def classify(self, signal_x, signal_y, tau, r_window = None, plot = False )-> None:
        """Function that assign to two signals the closest letter in the dictionnary leanred

        Args:
            signal_x (_type_):  a signal over x axis
            signal_y (_type_): a signal over y axis
            tau (_type_): number of atoms for sparse representation
            r_window (_type_, optional): Sakoe Chiba band width. Defaults to None.
            plot (bool, optional): Whether or not to plot the out. Defaults to False.

        Returns:
            _type_: _description_
        """

        alphas_x, deltas_x, alphas_y, deltas_y = self.fit(signal_x, signal_y, tau, r_window)
        results = {}
        for letter, reconstructed_x_signal, reconstructed_y_signal in self.generator_reconstructed_signal_per_letter(alphas_x, deltas_x, alphas_y, deltas_y):
            res_x, res_y, res_xy, res = self.metric(signal_x, signal_y, reconstructed_x_signal, reconstructed_y_signal)
            print(f"letter : {letter} , res_x : {res_x}, res_y : {res_y}, res_xy : {res_xy}, res : {res}")
            results[letter] = (res_x, res_y, res_xy, res)
        return results

    
    def reconstruct_signal(self, alphas, deltas, dictionnary):

        reconstructed_signal  = None
        for i,(alpha, delta, atom )in enumerate(zip(alphas, deltas,dictionnary)):
            if alpha != 0:
                if reconstructed_signal is None :
                    reconstructed_signal = alpha * delta @ atom
                reconstructed_signal += alpha * delta @ atom

        return reconstructed_signal
    
    def metric(self, signal_x, signal_y, reconstructed_x,reconstructed_y):
        
        res_x = np.linalg.norm(signal_x - reconstructed_x) 
        res_y = np.linalg.norm(signal_y - reconstructed_y)
        res_xy = np.linalg.norm(signal_x - reconstructed_x) + np.linalg.norm(signal_y - reconstructed_y)
        res = np.sum(np.linalg.norm(np.vstack(((signal_x - reconstructed_x),(signal_y- reconstructed_y))), axis = 1))
        return res_x, res_y, res_xy, res


    def generator_reconstructed_signal_per_letter(self,alphas_x, deltas_x, alphas_y, deltas_y):
        
        letters_x = self.map_letter_to_atoms_x_dico
        letters_y = self.map_letter_to_atoms_y_dico

        for i in deltas_x : 
            if isinstance(i, np.ndarray):
                N = len(i)
                break
        
        for letter in np.unique(letters_x) :


            idxs = [i for i,val in enumerate(letters_x) if val==letter]

            reconstructed_x_signal = np.zeros(N)
            
            for idx in idxs : 
                if alphas_x[idx] != 0:
                   reconstructed_x_signal += alphas_x[idx]* deltas_x[idx] @ self.concatenated_x_dico[idx]
            idxs = [i for i,val in enumerate(letters_y) if val==letter]
            
            reconstructed_y_signal = np.zeros(N)
            for idx in idxs : 
                if alphas_y[idx] != 0:
                    reconstructed_y_signal += alphas_y[idx]* deltas_y[idx] @ self.concatenated_y_dico[idx]
            yield letter, reconstructed_x_signal, reconstructed_y_signal
    
    def plot_atom(self, letters:str) -> None:
        fig, axis = plt.subplots(len(letters),2, figsize=(20,10 ))
        for j, letter in enumerate(letters):
            for i,atom in enumerate(self.x_dico[letter]):
                axis[j,0].plot(atom, label = f"atom {i}")
            for i,atom in enumerate(self.y_dico[letter]):
                axis[j,1].plot(atom, label = f"atom {i}")
            axis[j,0].legend()
            axis[j,1].legend()
            axis[j,0].set_title(f"x_Atoms for letter {letter}")
            axis[j,1].set_title(f"y_Atoms for letter {letter}")
        fig.tight_layout()
        plt.show()

    def select_letters(self, letters:list = None):
        self.selected_letters = letters
    

        

if __name__ == "__main__":
    lc = LetterClassifier()
    lc.load_data("results")
    lc.select_letters(["a","b","e"])
    lc.plot_atom(lc.letters[:5])
    
