import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FWIDataset(Dataset):
    def __init__(self, ann_file, query=None):
        self.df = pd.read_parquet(ann_file)
        if query is not None:
            self.df = self.df.query(query).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vel_path1 = row["vel_path1"]
        vel_path2 = row["vel_path2"]
        vel1 = np.load(vel_path1)
        vel2 = np.load(vel_path2)
        alpha = row["alpha"]
        vel = alpha * vel1 + (1 - alpha) * vel2
        vel = torch.from_numpy(vel).float()

        vel_path = row["vel_path"]
        seis_path = row["seis_path"]
        return dict(vel=vel, vel_path=vel_path, seis_path=seis_path)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ds = FWIDataset(
        ann_file="../artifacts/dtrainval_blend_type1_0.parquet",
    )
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=8)
    for batch in tqdm(dl):
        pass
