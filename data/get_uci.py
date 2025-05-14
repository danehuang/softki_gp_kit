import os
import pandas as pd
import subprocess
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


all_datasets = {
    "3droad": (434874, 3),
    "autompg": (392, 7),
    "bike": (17379, 17),
    "challenger": (23, 4),
    "concreteslump": (103, 7),
    "energy": (768, 8),
    "forest": (517, 12),
    "houseelectric": (2049280, 11),
    "keggdirected": (48827, 20),
    "kin40k": (40000, 8),
    "parkinsons": (5875, 20),
    "pol": (15000, 26),
    "pumadyn32nm": (8192, 32),
    "slice": (53500, 385),
    "solar": (1066, 10),
    "stock": (536, 11),
    "yacht": (308, 6),
    "airfoil": (1503, 5),
    "autos": (159, 25),
    "breastcancer": (194, 33),
    "buzz": (583250, 77),
    "concrete": (1030, 8),
    "elevators": (16599, 18),
    "fertility": (100, 9),
    "gas": (2565, 128),
    "housing": (506, 13),
    "keggundirected": (63608, 27),
    "machine": (209, 7),
    "pendulum": (630, 9),
    "protein": (45730, 9),
    "servo": (167, 4),
    "skillcraft": (3338, 19),
    "sml": (4137, 26),
    "song": (515345, 90),
    "tamielectric": (45781, 3),
    "wine": (1599, 11),
}


datasets = [
    "pol",
    "elevators",
    "bike",
    "kin40k",
    "protein",
    "keggdirected",
    "slice",
    "keggundirected",
    "3droad",
    "song",
    "buzz",
    "houseelectric",
]

class UCIDataset(Dataset):
    def __init__(self, csv_file="./foobar.csv", dim=1, transform=None, standarize=True, header=True, sep=None):
        if sep is not None:
            self.raw_data = pd.read_csv(csv_file, sep=sep)
        elif header:
            self.raw_data = pd.read_csv(csv_file)
        else:
            self.raw_data = pd.read_csv(csv_file, header=None)
        print("SIZE", self.raw_data.shape)
        self.transform = transform
        self.dim = dim
        self._whiten(standarize)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.data.iloc[idx, :self.dim].values.astype('float32')
        label = self.data.iloc[idx, self.dim].astype('float32')

        if self.transform:
            features = self.transform(features)

        return features, label
    
    def _whiten(self, standarize):
        if standarize:
            scaler = StandardScaler()
            self.data = pd.DataFrame(scaler.fit_transform(self.raw_data), columns=self.raw_data.columns)
        else:
            self.data = self.raw_data


class PoleteleDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/pol/data.csv", transform=None, standarize=True):
        super(PoleteleDataset, self).__init__(csv_file=csv_file, dim=26, transform=transform, standarize=standarize, header=False)


class ElevatorsDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/elevators/data.csv", transform=None, standarize=True):
        super(ElevatorsDataset, self).__init__(csv_file=csv_file, dim=18, transform=transform, standarize=standarize, header=False)


class BikeDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/bike/data.csv", transform=None, standarize=True):
        super(BikeDataset, self).__init__(csv_file=csv_file, dim=17, transform=transform, standarize=standarize, header=False)


class Kin40KDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/kin40k/data.csv", transform=None, standarize=True):
        super(Kin40KDataset, self).__init__(csv_file=csv_file, dim=8, transform=transform, standarize=standarize, header=False)


class ProteinDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/protein/data.csv", transform=None, standarize=True):
        super(ProteinDataset, self).__init__(csv_file=csv_file, dim=9, transform=transform, standarize=standarize, header=False)


class KeggDirectedDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/keggdirected/data.csv", transform=None, standarize=True):
        super(KeggDirectedDataset, self).__init__(csv_file=csv_file, dim=20, transform=transform, standarize=standarize, header=False)


class CTSlicesDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/slice/data.csv", transform=None, standarize=True):
        super(CTSlicesDataset, self).__init__(csv_file=csv_file, dim=385, transform=transform, standarize=standarize, header=False)


class KeggUndirectedDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/keggundirected/data.csv", transform=None, standarize=True):
        super(KeggUndirectedDataset, self).__init__(csv_file=csv_file, dim=27, transform=transform, standarize=standarize, header=False)


class RoadDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/3droad/data.csv", transform=None, standarize=True):
        super(RoadDataset, self).__init__(csv_file=csv_file, dim=3, transform=transform, standarize=standarize, header=False)


class SongDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/song/data.csv", transform=None, standarize=True):
        super(SongDataset, self).__init__(csv_file=csv_file, dim=90, transform=transform, standarize=standarize, header=False)


class BuzzDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/buzz/data.csv", transform=None, standarize=True):
        super(BuzzDataset, self).__init__(csv_file=csv_file, dim=77, transform=transform, standarize=standarize, header=False)


class HouseElectricDataset(UCIDataset):
    def __init__(self, csv_file="./uci_datasets/uci_datasets/houseelectric/data.csv", transform=None, standarize=True):
        super(HouseElectricDataset, self).__init__(csv_file=csv_file, dim=11, transform=transform, standarize=standarize, header=False)


if __name__ == "__main__":
    if not os.path.exists("./uci_datasets"):
        print("trying to clone")
        subprocess.run(["git clone git@github.com:treforevans/uci_datasets.git"], shell=True)
    base_dir = r"C:\Users\chris\Desktop\soft-gp\data\uci_datasets\uci_datasets"

    for dataset in datasets:
        gz_file_path = os.path.join(base_dir, dataset, "data.csv.gz")
        if os.path.exists(gz_file_path):
            print(f"Unzipping {dataset}")
            subprocess.run(["gzip", "-d", gz_file_path])
        else:
            print(f"No gzip file found for {dataset} at {gz_file_path}")
    
    torch_datasets = [
        PoleteleDataset,
        ElevatorsDataset,
        BikeDataset,
        Kin40KDataset,
        ProteinDataset,
        KeggDirectedDataset,
        CTSlicesDataset,
        KeggUndirectedDataset,
        RoadDataset,
        SongDataset,
        BuzzDataset,
        HouseElectricDataset,
    ]

    for torch_dataset in torch_datasets:
        print(torch_dataset)
        torch_dataset = torch_dataset()
        dataloader = DataLoader(torch_dataset, batch_size=1024)
        for x, y in tqdm(dataloader):
            pass
        