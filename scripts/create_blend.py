from pathlib import Path


import numpy as np
import pandas as pd


def concat_filename(path1, path2):
    path1 = Path(path1)
    path2 = Path(path2)

    path = path1.parent / f"{path1.stem}_{path2.stem}.npy"
    return str(path)


def shuffle_blend(df):
    perm = np.random.permutation(len(df))

    ret = df.copy()
    ret["vel_path1"] = ret["vel_path"]
    ret["vel_path2"] = ret["vel_path"].values[perm]
    ret["seis_path1"] = ret["seis_path"]
    ret["seis_path2"] = ret["seis_path"].values[perm]
    # alpha in (0.3, 0.7)
    ret["alpha"] = np.random.uniform(0.3, 0.7, len(ret))
    # concat filename to vel_path
    ret["vel_path"] = ret.apply(
        lambda row: concat_filename(row["vel_path1"], row["vel_path2"]), axis=1
    )
    ret["seis_path"] = ret.apply(
        lambda row: concat_filename(row["seis_path1"], row["seis_path2"]), axis=1
    )
    return ret


def create_blend(df, group_id):
    # intra-fold, intra-subdir
    df = df.copy()
    dfs = []
    for _, gdf in df.groupby(group_id):
        gdf = shuffle_blend(gdf)
        dfs.append(gdf)
    ret = pd.concat(dfs)
    ret = ret.reset_index(drop=True)
    return ret


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
    df = pd.read_parquet("../artifacts/dtrainval_orig.parquet")

    # create 8 intra-fold, intra-subdir blends
    for seed in range(8):
        np.random.seed(seed)
        blend_df = create_blend(df, ["fold", "subdir"])
        blend_df = replace_data_root(
            blend_df, "../artifacts/data_orig", f"../artifacts/data_blend_type1_{seed}"
        )
        out_path = f"../artifacts/dtrainval_blend_type1_{seed}.parquet"
        blend_df.to_parquet(out_path)
        print(f"Created {out_path}")

    # create 8 intra-fold, inter-subdir blends
    for seed in range(8):
        np.random.seed(seed)
        blend_df = create_blend(df, ["fold"])
        blend_df = replace_data_root(
            blend_df, "../artifacts/data_orig", f"../artifacts/data_blend_type2_{seed}"
        )
        out_path = f"../artifacts/dtrainval_blend_type2_{seed}.parquet"
        blend_df.to_parquet(out_path)
        print(f"Created {out_path}")
