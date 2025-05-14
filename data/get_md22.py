from os.path import exists
import requests
import tarfile
from zipfile import ZipFile 

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


STANDARDIZE = True


def to_unit_cube(x, lb, ub, g=None):
    xx = (x - lb) / (ub - lb)
    return xx


def from_unit_cube(x, lb, ub, g=None):
    xx = x * (ub - lb) + lb
    return xx


# =============================================================================
# Base Dataset
# =============================================================================

class MD22Dataset(Dataset):
    def __init__(self, npz_file="./md22/md22_DHA.npz", dtype=torch.float32, transform=None, standarize=False, flat=False, center=True, get_forces=False):
        # Store arguments
        self.npz_file = npz_file
        self.dtype = dtype
        self.get_forces = get_forces
        self.transform = transform
        self.center = center
    
        # Unpack data
        self.raw_data = np.load(npz_file)
        self.coords = []
        self.energies = []
        self.forces = []
        for x, e, f in tqdm(zip(self.raw_data["R"], self.raw_data["E"], self.raw_data["F"])):
            self.coords += [torch.tensor(x.flatten())]
            if flat:
                self.energies += [e[0]]
            else:
                self.energies += [e]
            self.forces += [torch.tensor(f.flatten(), dtype=dtype)]
        self.coords = torch.stack(self.coords).to(dtype=dtype)
        self.energies = torch.tensor(self.energies).to(dtype=dtype)  # negate the energy instead of the force
        self.forces = torch.stack(self.forces)
        self.zs = torch.tensor(self.raw_data["z"].flatten()).to(dtype)
        self.dim = len(self.coords[0].reshape(-1))
        print(self.energies.shape, self.forces.shape)

        # Standardize data
        self.shift = 0
        self.scale = 1
        if standarize:
            # Standardize x units
            if False:
                mins = [x.min() for x in self.coords]
                maxs = [x.max() for x in self.coords]
                lb = np.array(mins).min()
                ub = np.array(maxs).max()
                print("lb ub", lb, ub)
                self.lb = lb
                self.ub = ub
                self.coords = torch.tensor(to_unit_cube(self.coords, lb, ub).to(dtype=dtype))
            else:
                self.x_scale = 3
                self.coords = self.coords / self.x_scale

            # Standardize y units            
            mu = torch.mean(self.energies)  # Negated energy
            print("mean", mu.shape, self.energies.shape, mu)
            sigma = torch.std(torch.cat([(-self.energies + mu).unsqueeze(-1), self.forces], dim=-1))  # Adjust force (but not energy)
            print("energy shapes", self.energies.shape, "force shape", self.forces.shape)
            print("separate mean std", torch.std(self.energies), "separate force std", torch.std(self.forces))
            self.energies = (-self.energies + mu) / sigma
            print("mean", mu, "std", sigma, "scale")
            self.forces /= sigma
            
            self.shift = mu
            self.scale = sigma
        else:
            if self.center:
                self._center_energies()

    def _center_energies(self) -> None:
        self.mean = self.energies.mean()
        self.energies = self.energies - self.mean

    def __len__(self):
        return len(self.energies)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.coords[idx]
        if self.transform:
            features = self.transform(features)

        if self.get_forces:
            return features, {
                "energy": self.energies[idx],
                "neg_force": self.forces[idx],
            }
        else:
            label = self.energies[idx]
            return features, label
    

# =============================================================================
# AcAla3NHME
# =============================================================================

def get_AcAla3NHME():
    if not exists('./md22/md22_Ac-Ala3-NHMe.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_Ac-Ala3-NHMe.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22_Ac-Ala3-NHMe.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_Ac-Ala3-NHMe.npz")
    print(data.files)
    print(data["R"].shape)
    print(len(data["E"]))
    print(data["name"])
    print(data["z"])

    dataset = MD22_AcAla3NHME_Dataset(npz_file="./md22_Ac-Ala3-NHMe.npz", standarize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_AcAla3NHME_Dataset(MD22Dataset):
    """
    N = 85109
    D = 42 x 3 = 126
    """    
    def __init__(self, npz_file="./md22_Ac-Ala3-NHMe.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, get_forces=False):
        super(MD22_AcAla3NHME_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=True, coulomb=coulomb, get_forces=get_forces)


# =============================================================================
# Docosahexaenoic acid
# =============================================================================

def get_DHA():
    if not exists('./md22/md22_DHA.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_DHA.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22_DHA.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_DHA.npz")
    print(data.files)
    print(data["R"].shape)
    print(len(data["E"]))
    print(data["name"])
    print(data["z"])

    dataset = MD22_DHA_Dataset(npz_file="./md22_DHA.npz", standarize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_DHA_Dataset(MD22Dataset):
    """
    N = 69753
    D = 56 x 3 = 168
    """    
    def __init__(self, npz_file="./md22_DHA.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, get_forces=False):
        super(MD22_DHA_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, get_forces=get_forces)
        

# =============================================================================
# Stachyose
# =============================================================================

def get_stachyose():
    if not exists('./md22/md22_stachyose.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_stachyose.npz"
        
        response = requests.get(url, stream=True)
        with open("md22_stachyose.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_stachyose.npz")
    print(data.files)
    print(data["R"].shape)
    print(len(data["E"]))
    print(data["name"])
    print(data["z"])

    dataset = MD22_Stachyose_Dataset(npz_file="./md22_stachyose.npz", standarize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_Stachyose_Dataset(MD22Dataset):
    """
    N = 27272
    D = 87 x 3 = 261
    """    
    def __init__(self, npz_file="./md22_stachyose.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, get_forces=False):
        super(MD22_Stachyose_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, get_forces=get_forces)
    

# =============================================================================
# AT-AT
# =============================================================================

def get_dna_at_at():
    if not exists('./md22/md22_AT-AT.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22_AT-AT.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_AT-AT.npz")
    print(data["R"].shape)
    print(data["name"])

    dataset = MD22_DNA_AT_AT_Dataset(npz_file="./md22_AT-AT.npz", standarize=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_DNA_AT_AT_Dataset(MD22Dataset):
    """
    N = 20001
    D = 60 x 3 = 180
    """    
    def __init__(self, npz_file="./md22/md22_AT-AT.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, get_forces=False):
        super(MD22_DNA_AT_AT_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=True, get_forces=get_forces)


# =============================================================================
# AT-AT
# =============================================================================

def get_dna_at_at_cg_cg():
    if not exists('./md22/md22_AT-AT-CG-CG.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_AT-AT-CG-CG.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22_AT-AT-CG-CG.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22_AT-AT-CG-CG.npz")
    print(data.keys())
    print(data["R"].shape)
    print(data["name"])

    dataset = MD22_DNA_AT_AT_Dataset(npz_file="./md22_AT-AT-CG-CG.npz", standarize=STANDARDIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_DNA_AT_AT_CG_CG_Dataset(MD22Dataset):
    """
    N = 10153
    D = 118 x 3 = 354
    """    
    def __init__(self, npz_file="./md22_AT-AT-CG-CG.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, get_forces=False):
        super(MD22_DNA_AT_AT_CG_CG_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=True, get_forces=get_forces)


# =============================================================================
# Buckyball Catcher
# =============================================================================

def get_buckyball_catcher():
    if not exists('./md22/md22_buckyball-catcher.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_buckyball-catcher.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_buckyball-catcher.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_buckyball-catcher.npz")
    print(data.keys())
    print(data["R"].shape)
    print(data["name"])

    dataset = MD22_Buckyball_Catcher_Dataset(npz_file="./md22/md22_buckyball-catcher.npz", standarize=STANDARDIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_Buckyball_Catcher_Dataset(MD22Dataset):
    """
    N = 6102
    D = 148 x 3 = 444
    """    
    def __init__(self, npz_file="./md22/md22_buckyball-catcher.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, get_forces=False):
        super(MD22_Buckyball_Catcher_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=False, get_forces=get_forces)


# =============================================================================
# Double-Walled Nanotub
# =============================================================================

def get_double_walled_nanotube():
    if not exists('./md22/md22_double-walled_nanotube.npz'):
        url = "http://www.quantum-machine.org/gdml/repo/datasets/md22_double-walled_nanotube.npz"
        
        response = requests.get(url, stream=True)
        with open("./md22/md22_double-walled_nanotube.npz", "wb") as f:
            for data in tqdm(response.iter_content()):
                f.write(data)

    data = np.load("./md22/md22_double-walled_nanotube.npz")
    print(data.keys())
    print(data["R"].shape)
    print(data["name"])

    dataset = MD22_DoubleWalledNanotube_Dataset(npz_file="./md22/md22_double-walled_nanotube.npz", standarize=STANDARDIZE)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for x, y in tqdm(dataloader):
        pass


class MD22_DoubleWalledNanotube_Dataset(MD22Dataset):
    """
    N = 5032
    D = 370 x 3 = 1110
    """    
    def __init__(self, npz_file="./md22/md22_double-walled_nanotube.npz", dtype=torch.float32, transform=None, standarize=STANDARDIZE, get_forces=False):
        super(MD22_DoubleWalledNanotube_Dataset, self).__init__(npz_file=npz_file, dtype=dtype, transform=transform, standarize=standarize, flat=True, get_forces=get_forces)


if __name__ == "__main__":
    get_AcAla3NHME()
    get_DHA()
    get_stachyose()
    get_dna_at_at()
    get_dna_at_at_cg_cg()
    get_buckyball_catcher()
    get_double_walled_nanotube()
