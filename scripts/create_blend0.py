from pathlib import Path


import pandas as pd


def replace_root(path, old_root, new_root):
    path = Path(path)
    return str(new_root / path.relative_to(old_root))


if __name__ == "__main__":
    df = pd.read_parquet("../artifacts/dtrainval_orig.parquet")
    # non-blend in blend format for generating 6ch seis of data_orig
    old_root = "../artifacts/data_orig"
    new_root = "../artifacts/data_orig_6ch"
    df["vel_path1"] = df["vel_path"]
    df["vel_path2"] = df["vel_path"]
    df["seis_path1"] = df["seis_path"]
    df["seis_path2"] = df["seis_path"]
    df["alpha"] = 1.0
    df["vel_path"] = df["vel_path"].apply(lambda x: replace_root(x, old_root, new_root))
    df["seis_path"] = df["seis_path"].apply(
        lambda x: replace_root(x, old_root, new_root)
    )
    df.to_parquet("../artifacts/dtrainval_orig_6ch.parquet")

    # split for 4 GPUs
    for i in range(4):
        df.iloc[i::4].to_parquet(f"../artifacts/dtrainval_orig_6ch_part{i}.parquet")
