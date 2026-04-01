import numpy as np
import pandas as pd

from scipy.io import loadmat

def normalise(X):
    X_mean = np.average(X, 0)[None, :]
    X_std = np.std(X, 0)[None, :] + 1e-6
    return (X - X_mean) / X_std

class Datasets(object):
    def __init__(self): 
        X_raw, y_raw = self.read_data()
        self.X, self.y = self.preprocess_data(X_raw, y_raw)

    @property
    def datapath(self):
        raise NotImplementedError("datapath not implemented for base class")

    def read_data(self):
        raise NotImplementedError
    
    def preprocess_data(self, X, y):
        return normalise(X), normalise(y)

datapath_base = "experiments/uci/data/"

class Bike(Datasets):
    N, D, name = 17379, 17, "Bike"

    @property
    def datapath(self):
        return datapath_base + "bike.csv"

    def read_data(self):
        data = pd.read_csv(self.datapath, header=None).values
        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        return X, y

class Concrete(Datasets):
    N, D, name = 1030, 8, "Concrete"

    @property
    def datapath(self):
        return datapath_base + "concrete.csv"

    def read_data(self):
        data = pd.read_csv(self.datapath, header=None).values
        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        return X, y

class Elevators(Datasets):
    N, D, name = 16599, 18, "Elevators"

    @property
    def datapath(self):
        return datapath_base + "elevators.csv"

    def read_data(self):
        data = pd.read_csv(self.datapath, header=None).values
        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        return X, y
    

class Keggdirected(Datasets):
    N, D, name = 48827, 20, "Keggdirected"

    @property
    def datapath(self):
        return datapath_base + "keggdirected.mat"
    
    def read_data(self):
        data = loadmat(self.datapath)["data"]
        X = data[:, :-1]
        y = data[:, -1, None]
        return X, y
    
class Keggundirected(Datasets):
    N, D, name = 63608, 27, "Keggundirected"
    
    @property
    def datapath(self):
        return datapath_base + "keggundirected.mat"
    
    def read_data(self):
        data = loadmat(self.datapath)["data"]
        X = data[:, :-1]
        y = data[:, -1, None]
        return X, y

class Kin8nm(Datasets):
    N, D, name = 7372, 8, "Kin8nm"

    @property
    def datapath(self):
        return datapath_base + "kin8nm.csv"

    def read_data(self):
        data = pd.read_csv(self.datapath, header=None).values
        X = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)
        return X, y

class Kin40k(Datasets):
    N, D, name = 40000, 8, "Kin40k"

    @property
    def datapath(self):
        return datapath_base + "kin40k.mat"

    def read_data(self):
        data = loadmat(self.datapath)["data"]
        X = data[:, :-1]
        y = data[:, -1, None]
        return X, y

class Naval(Datasets):
    N, D, name = 11934, 14, "Naval"

    @property
    def datapath(self):
        return datapath_base + "data.txt"

    def read_data(self):
        data = pd.read_fwf(self.datapath, header=None).values
        X = data[:, :-2] 
        y = data[:, -2].reshape(-1, 1) 
        X = np.delete(X, [8, 11], axis=1)  
        return X, y

class Pol(Datasets):
    N, D, name = 15000, 26, "Pol"

    @property
    def datapath(self):
        return datapath_base + "pol.mat"

    def read_data(self):
        data = loadmat(self.datapath)["data"]
        X = data[:, :-1]
        y = data[:, -1, None]
        return X, y

class Protein(Datasets):
    N, D, name = 45730, 9, "Protein"

    @property
    def datapath(self):
        return datapath_base + "CASP.csv"

    def read_data(self):
        data = pd.read_csv(self.datapath).values
        X = data[:, 1:]
        y = data[:, 0].reshape(-1, 1)
        return X, y


class Skillcraft(Datasets):
    N, D, name = 3338, 19, "Skillcraft"
    
    def __init__(self) -> None:
        super().__init__()

    @property
    def datapath(self):
        return datapath_base + "skillcraft.mat"

    def read_data(self):
        data = loadmat(self.datapath)["data"]
        X = data[:, :-1]
        y = data[:, -1, None]
        return X, y

class Tamielectric(Datasets):
    N, D, name = 45781, 3, "Tamielectric"
    
    @property
    def datapath(self):
        return datapath_base + "tamielectric.mat"
    
    def read_data(self):
        data = loadmat(self.datapath)["data"]
        X = data[:, :-1]
        y = data[:, -1, None]
        return X, y
