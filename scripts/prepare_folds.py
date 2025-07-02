import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


def fault2df(data_root):
    seis_paths = list(data_root.glob("seis*.npy"))
    vel_paths = [p.parent / p.name.replace("seis", "vel") for p in seis_paths]
    dfs = []
    for seis_path, vel_path in zip(seis_paths, vel_paths):
        vel = np.load(vel_path, mmap_mode="r")
        _seis_paths = [
            str(seis_path.with_suffix(f".{i}.npy")) for i in range(vel.shape[0])
        ]
        _vel_paths = [
            str(vel_path.with_suffix(f".{i}.npy")) for i in range(vel.shape[0])
        ]
        df = pd.DataFrame(
            dict(
                orig_seis_path=str(seis_path),
                seis_path=_seis_paths,
                vel_path=_vel_paths,
            )
        )
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def vel2df(data_root):
    seis_paths = list((data_root / "data").glob("*.npy"))
    vel_paths = [
        p.parent.parent / "model" / p.name.replace("data", "model") for p in seis_paths
    ]
    dfs = []
    for seis_path, vel_path in zip(seis_paths, vel_paths):
        vel = np.load(vel_path, mmap_mode="r")
        _seis_paths = [
            str(seis_path.with_suffix(f".{i}.npy")) for i in range(vel.shape[0])
        ]
        _vel_paths = [
            str(vel_path.with_suffix(f".{i}.npy")) for i in range(vel.shape[0])
        ]
        df = pd.DataFrame(
            dict(
                orig_seis_path=str(seis_path),
                seis_path=_seis_paths,
                vel_path=_vel_paths,
            )
        )
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    return df


def replace_data_root(df, old_root, new_root):
    old_root = Path(old_root)
    new_root = Path(new_root)

    def replace_root(path):
        path = Path(path)
        return str(new_root / path.relative_to(old_root))

    df = df.copy()
    df["vel_path"] = df["vel_path"].apply(replace_root)
    df["seis_path"] = df["seis_path"].apply(replace_root)
    return df


if __name__ == "__main__":
    data_root = Path("../data/")

    gen_types = dict(
        vel_and_style=dict(
            sub_dirs=[
                "CurveVel_A",
                "CurveVel_B",
                "FlatVel_A",
                "FlatVel_B",
                "Style_A",
                "Style_B",
            ],
            process_fn=vel2df,
        ),
        fault=dict(
            sub_dirs=[
                "CurveFault_A",
                "CurveFault_B",
                "FlatFault_A",
                "FlatFault_B",
            ],
            process_fn=fault2df,
        ),
    )
    dfs = []
    for gen_type, gen_type_cfg in gen_types.items():
        for subdir in gen_type_cfg["sub_dirs"]:
            df = gen_type_cfg["process_fn"](data_root / subdir)
            df["subdir"] = subdir
            df["gen_type"] = gen_type
            dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    df = replace_data_root(df, "../data", "../artifacts/data_orig")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    splits = sgkf.split(df, df["subdir"], df["orig_seis_path"])
    df["fold"] = -1
    for fold, (_, val_inds) in enumerate(splits):
        df.loc[val_inds, "fold"] = fold
    df["fold"] = df["fold"].astype(int)
    df.to_parquet("../artifacts/dtrainval_orig.parquet")
