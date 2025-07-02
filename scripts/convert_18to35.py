import torch


def modify_head(w):
    w = w[[0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]]
    if w.ndim == 2:
        w = w.reshape(4, 4, -1).mean(1)
    elif w.ndim == 1:
        w = w.reshape(4, 4).mean(1)
    return w


d = torch.load(
    "../artifacts/release02/epoch_10.pth", map_location="cpu", weights_only=False
)
sd = d["model"]

sd["_orig_mod.module.head.weight"] = modify_head(sd["_orig_mod.module.head.weight"])
sd["_orig_mod.module.head.bias"] = modify_head(sd["_orig_mod.module.head.bias"])
opt = d["optimizer"]
opt["state"][512]["momentum_buffer"] = modify_head(opt["state"][512]["momentum_buffer"])

for key in ("exp_avg", "exp_avg_sq"):
    opt["state"][513][key] = modify_head(opt["state"][513][key])

torch.save(
    dict(model=sd, optimizer=opt),
    "../artifacts/release02/dump_interpolate_head.pth",
)
