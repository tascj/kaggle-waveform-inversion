# ruff: noqa
import os


# os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import glob
import torch
from pathlib import Path

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F


import pandas as pd
import numpy as np
from detectron2.config import LazyConfig, instantiate
from tqdm import tqdm


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.seis_files = sorted(glob.glob(os.path.join(data_root, "*.npy")))

    def __len__(self):
        return len(self.seis_files)

    def __getitem__(self, idx):
        seis_file = self.seis_files[idx]
        seis = np.load(seis_file)
        seis = torch.from_numpy(seis)
        oid = Path(seis_file).stem
        return {"seis": seis, "oid": oid}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--load-from", required=True, type=str)
    parser.add_argument("--input-root", required=True, type=str)
    parser.add_argument("--output-root", required=True, type=str)
    return parser.parse_args()


@torch.no_grad()
def do_test(input_root, output_root, model):
    test_ds = TestDataset(input_root)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    model.eval()
    prog_bar = tqdm(test_loader)
    for batch in prog_bar:
        seis = batch["seis"].cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred_vel = model(seis)
            pred_vel = pred_vel.float()
        pred_vel = pred_vel.cpu()
        for oid, vel in zip(batch["oid"], pred_vel):
            np.save(output_root / f"{oid}.npy", vel.numpy())


def main():
    args = parse_args()
    cfg = LazyConfig.load(args.config)
    model = instantiate(cfg.model)

    load_from = args.load_from
    state_dict = torch.load(load_from, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    state_dict = {
        k.replace("_orig_mod.module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    load_result = model.load_state_dict(state_dict, strict=False)
    print(load_result)
    model = model.cuda()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    do_test(args.input_root, output_root, model)


if __name__ == "__main__":
    main()
