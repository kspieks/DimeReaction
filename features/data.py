import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, DataLoader

from features.common import NUMBER_BY_SYMBOL, xyz_file_format_to_xyz


class ReactionDataset(Dataset):
    def __init__(self, args, mode='train'):
        super(Dataset, self).__init__()

        self.args = args
        self.split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        self.split = np.load(args.split_path, allow_pickle=True)[self.split_idx]

        self.r_dicts = self.get_dicts(self.args.reactant_xyzs)  # list of reactant dictionaries
        self.ts_dicts = self.get_dicts(self.args.ts_xyzs)       # list of ts dictionaries

        self.targets = self.get_targets()                       # list of regression targets
        self.mean = np.mean(self.targets)
        self.std = np.std(self.targets)

    def get_dicts(self, xyz_path):
        """Creates list of dictionaries containing the molecule coordinates and atomic numbers"""
        dicts = []
        with open(xyz_path, 'r') as f:
            xyz_lines = ''
            for line in f.readlines():
                if '$$$$$' in line:
                    dicts.append(xyz_file_format_to_xyz(xyz_lines))
                    xyz_lines = ''
                else:
                    xyz_lines += line
        return [dicts[i] for i in self.split]

    def get_targets(self):
        """Create list of targets for regression"""
        df = pd.read_csv(self.args.target_path)
        targets = df[self.args.targets].values

        return [targets[i] for i in self.split]

    def process_key(self, key):
        r_dict = self.r_dicts[key]
        ts_dict = self.ts_dicts[key]
        y = self.targets[key]
        data = self.xyz2data(r_dict, ts_dict, y)
        return data

    def xyz2data(self, r_dict, ts_dict, y):
        data = Data()
        data.r_coords = torch.tensor(r_dict['coords'], dtype=torch.float)
        data.r_z = torch.tensor([NUMBER_BY_SYMBOL[sym] for sym in r_dict['symbols']], dtype=torch.long)
        data.ts_coords = torch.tensor(ts_dict['coords'], dtype=torch.float)
        data.ts_z = torch.tensor([NUMBER_BY_SYMBOL[sym] for sym in ts_dict['symbols']], dtype=torch.long)
        data.y = torch.tensor(y, dtype=torch.float)
        return data

    def __len__(self):
        return len(self.r_dicts)

    def __getitem__(self, key):
        return self.process_key(key)


def construct_loader(args, modes=('train', 'val')):
    """Create PyTorch Geometric DataLoader"""

    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    for mode in modes:
        dataset = ReactionDataset(args, mode)
        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=True if mode == 'train' else False,
                            follow_batch='r_z',
                            num_workers=args.num_workers,
                            pin_memory=True,
                            )
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders