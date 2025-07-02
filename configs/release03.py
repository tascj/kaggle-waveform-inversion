# ruff: noqa
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from detectron2.config import LazyCall as L

from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    CosineParamScheduler,
)

from fwi_modules.data.dataset_6ch import FWIDataset as FWIDataset6ch
from fwi_modules.data.dataset import FWIDataset
from fwi_modules.models.model_vit_large_split_35 import FWIModel
from fwi_modules.optim.muon import MuonWithAuxAdam


# model config
model = L(FWIModel)(pretrained=True, split_at=8)
optimizer = L(MuonWithAuxAdam)(
    lr=20e-5,
    weight_decay=0.01,
    momentum=0.95,
    betas=(0.9, 0.95),
)


# data config
def build_dataset(fold, training):
    if training:
        datasets = [
            FWIDataset6ch(
                ann_file="../artifacts/dtrainval_orig_6ch.parquet",
                query=f"(fold != {fold})",
                train_mode=True,
                erase_last_row=True,
            )
        ]
        for seed in range(8):
            for gen_type in (1, 2):
                datasets.append(
                    FWIDataset6ch(
                        ann_file=f"../artifacts/dtrainval_blend_type{gen_type}_{seed}.parquet",
                        query=f"(fold != {fold})",
                        train_mode=True,
                        erase_last_row=True,
                    )
                )
        dataset = torch.utils.data.ConcatDataset(datasets)
    else:
        dataset = FWIDataset(
            ann_file="../artifacts/dtrainval_orig.parquet",
            query=f"(fold == {fold})",
            erase_last_row=True,
        )
    return dataset


def build_data_loader(dataset, batch_size, num_workers, training=True):
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=training, drop_last=training)
    else:
        sampler = None
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=training and not dist.is_initialized(),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=training,
        persistent_workers=True,
        sampler=sampler,
        pin_memory=True,
    )


VAL_FOLD = 0
dataloader = dict(
    train=L(build_data_loader)(
        dataset=L(build_dataset)(fold=VAL_FOLD, training=True),
        batch_size=16,
        num_workers=8,
        training=True,
    ),
    val=L(build_data_loader)(
        dataset=L(build_dataset)(fold=VAL_FOLD, training=False),
        batch_size=16,
        num_workers=8,
        training=False,
    ),
)

max_epochs = 1
lr_multiplier = L(CompositeParamScheduler)(
    schedulers=[
        L(ConstantParamScheduler)(value=1),
        L(CosineParamScheduler)(start_value=1, end_value=0.001),
    ],
    lengths=[0.5, 0.5],
    interval_scaling=["rescaled", "rescaled"],
)

train = dict(
    device="cuda",
    max_epochs=max_epochs,
    log_interval=100,
    checkpoint_interval=1,
    eval_interval=max_epochs,
    log_buffer_size=20 * 5,
    clip_grad=False,
    seed=3,
    compile=dict(
        mode="reduce-overhead",
    ),
    resume_from="../artifacts/release02/dump_interpolate_head.pth",
)
