import os 
import numpy as np  
import matplotlib.pyplot as plt

from twi_ksvd.omp import TWI_OMP


class LetterClassifier():
    def __init__(self) -> None:
        self.x_dico = {}
        self.y_dico = {}
    
    @property
    def letters(self) -> list:
        return list(self.x_dico.keys())

    @property
    def concatenated_x_dico(self) -> list:
        return [atom for letter in self.x_dico.values() for atom in letter]   

    @property
    def concatenated_y_dico(self) -> list:
        return [atom for letter in self.y_dico.values() for atom in letter]
    
    @property
    def count_atoms_x_dico(self) -> int:
        return {letter : len(atoms) for letter, atoms in self.x_dico.items()}
    
    @property
    def count_atoms_y_dico(self) -> int: 
        return {letter : len(atoms) for letter, atoms in self.y_dico.items()}
    
    @property
    def map_letter_to_atoms_x_dico(self) -> dict:
        result = []
        for letter, numbr in self.count_atoms_x_dico.items():
            result += [letter] * numbr
        return result
    
    @property
    def map_letter_to_atoms_y_dico(self) -> dict:
        result = []
        for letter, numbr in self.count_atoms_y_dico.items():
            result += [letter] * numbr
        return result 

    def load_data(self, folder_path: str) -> None:
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
        
        alphas_x, deltas_x = TWI_OMP(signal_x, self.concatenated_x_dico, tau=tau, r_window=r_window)
        alphas_y, deltas_y = TWI_OMP(signal_y, self.concatenated_y_dico, tau=tau, r_window=r_window)

        
        return alphas_x, deltas_x, alphas_y, deltas_y


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

    

        

if __name__ == "__main__":
    lc = LetterClassifier()
    lc.load_data("results")

    # lc.plot_atom(lc.letters[:5])
    
