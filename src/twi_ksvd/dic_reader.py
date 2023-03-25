import os 
import numpy as np  
import matplotlib.pyplot as plt


class LetterClassifier():
    def __init__(self) -> None:
        self.x_dico = {}
        self.y_dico = {}

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


    def plot_atom(self, letter:str) -> None:
        fig, axis = plt.subplots(1,2, figsize=(10,5))
        for i,atom in enumerate(self.x_dico[letter]):
            axis[0].plot(atom, label = f"atom {i}")
        for i,atom in enumerate(self.y_dico[letter]):
            axis[1].plot(atom, label = f"atom {i}")
        
        axis[0].legend()
        axis[0].set_title(f"x_Atoms for letter {letter}")
        axis[1].legend()
        axis[1].set_title(f"y_Atoms for letter {letter}")
        plt.show()
        

if __name__ == "__main__":
    lc = LetterClassifier()
    lc.load_data("results")
    lc.plot_atom("a")
