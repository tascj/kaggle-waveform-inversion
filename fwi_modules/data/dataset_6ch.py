import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tabulate import tabulate


class FWIDataset(Dataset):
    def __init__(
        self,
        ann_file,
        query=None,
        train_mode=False,
        erase_last_row=False,
    ):
        self.df = pd.read_parquet(ann_file)
        if query is not None:
            self.df = self.df.query(query).reset_index(drop=True)
        self.train_mode = train_mode
        self.erase_last_row = erase_last_row

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seis_path = row["seis_path"]
        vel_path = row["vel_path"]
        seis = np.load(seis_path)
        vel = np.load(vel_path)
        seis = torch.from_numpy(seis).float()
        vel = torch.from_numpy(vel).float()
        assert seis.shape[0] == 6
        if self.train_mode:
            # Temporal flip
            if np.random.random() < 0.5:
                seis = seis.flip(dims=[0, -1])
                vel = vel.flip(dims=[-1])
        seis = seis[[0, 1, 2, 4, 5]]
        if self.erase_last_row:
            seis[:, -1, :] = 0
        return dict(seis=seis, vel=vel)

    def format_results(self, losses):
        df = self.df.copy()
        df["loss"] = losses
        results = dict()
        for subdir, gdf in df.groupby("subdir"):
            results[subdir] = gdf.agg(
                {"loss": ["mean", "median", "min", "max"]}
            ).to_dict()["loss"]
        results["All"] = df.agg({"loss": ["mean", "median", "min", "max"]}).to_dict()[
            "loss"
        ]
        return tabulate(pd.DataFrame(results).T, headers="keys", tablefmt="grid")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    fold = 0
    ds = FWIDataset(
        ann_file="../artifacts/dtrainval_orig_6ch.parquet",
        query=f"(subdir == 'CurveFault_A') and (fold != {fold})",
        train_mode=True,
        erase_last_row=True,
    )

    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=8)
    for batch in tqdm(dl):
        pass
