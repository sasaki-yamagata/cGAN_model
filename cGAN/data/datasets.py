import numpy as np
import pandas as pd
import torch
from torch.utils import data
from rdkit import rdBase, Chem, DataStructs
from rdkit.Chem import AllChem


class CreateData:

    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        # drop "OC(=O)C(=C\C1=CC=[Cl]C=[Cl]1)\C1=CN=CC=C1" because of Explicit valence error (only relevant for Dataset I)
        data = data[data.SMILES != "OC(=O)C(=C\C1=CC=[Cl]C=[Cl]1)\C1=CN=CC=C1"]
        data = data.dropna()
        data = data.values
        smiles_l = (data[:, 1])
        fps_data = self._convert_fps(smiles_l)
        self.x_data = self._add_noize(fps_data, 100)
        self.y_data = torch.tensor(np.array(data[:, 2:], dtype="float32"))

    @staticmethod
    def _convert_fps(smiles_l):
        mol_l = [Chem.MolFromSmiles(s) for s in smiles_l if s is not None]
        fps_bit = [AllChem.GetMorganFingerprintAsBitVect(m, 3, 1024) for m in mol_l]
        fps_l = []

        # Convert fingerprints to array
        for fps in fps_bit:
            fps_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fps, fps_arr)
            fps_l.append(fps_arr)
        fps_l = np.array(fps_l, dtype='float32')

        # Normalize fingerprints
        fps_l_max = np.amax(fps_l)
        fps_l_min = np.amin(fps_l)
        fps_norm = ((fps_l - fps_l_min) / (fps_l_max - fps_l_min))
        fps_norm = torch.from_numpy(fps_norm)
        return fps_norm
    
    @staticmethod
    def _add_noize(fps_data, noize_size):
        noize_shape = (fps_data.shape[0], noize_size)
        z = np.random.randn(noize_shape[0] * noize_shape[1])
        z =  z.reshape(noize_shape)
        x_data = np.concatenate([fps_data, z], axis=1)
        x_data = torch.from_numpy(x_data)
        return x_data
        


class GanDataset(data.Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return self.x_data.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index, :], self.y_data[index, :]









        
