import joblib
import glob
from pathlib import Path

import numpy as np


def dump_npy(npy_path, old_root, new_root):
    npy_path = Path(npy_path)
    new_npy_path = new_root / npy_path.relative_to(old_root)
    new_npy_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.load(npy_path)
    for idx in range(arr.shape[0]):
        np.save(new_npy_path.with_suffix(f".{idx}.npy"), arr[idx])


if __name__ == "__main__":
    old_root = Path("../data/")
    new_root = Path("../artifacts/data_orig")
    npy_paths = glob.glob("../data/**/*.npy", recursive=True)
    jobs = []
    for npy_path in npy_paths:
        job = joblib.delayed(dump_npy)(npy_path, old_root, new_root)
        jobs.append(job)
    joblib.Parallel(n_jobs=10, verbose=10)(jobs)
