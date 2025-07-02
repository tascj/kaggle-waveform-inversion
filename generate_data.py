import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from fwi_modules.data.dataset_for_gen import FWIDataset
from fwi_modules.forward_modeling import vel_to_seis


def save_file(arr, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


@torch.no_grad()
def inference(ann_file, query=None):
    ds = FWIDataset(ann_file, query=query)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    for batch in tqdm(dl):
        if all(Path(path).exists() for path in batch["vel_path"]) and all(
            Path(path).exists() for path in batch["seis_path"]
        ):
            continue
        seis = vel_to_seis(
            batch["vel"].squeeze(1).cuda(non_blocking=True), device="cuda"
        )
        for arr, path in zip(batch["vel"], batch["vel_path"]):
            save_file(arr, path)
        seis = seis.cpu()
        for arr, path in zip(seis, batch["seis_path"]):
            arr = arr.numpy()[:, 1:]
            save_file(arr, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()
    inference(args.ann_file, args.query)
