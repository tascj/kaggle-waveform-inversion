# ruff: noqa
import os


# os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import time
import shutil
import torch

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

# torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F

from pathlib import Path


from detectron2.config import LazyConfig, instantiate
from detectron2.solver import LRMultiplier
from detectron2.engine.hooks import LRScheduler
from detectron2.utils.env import seed_all_rng

from fwi_modules.logging_utils import get_logger
from fwi_modules.utils import to_gpu, get_total_norm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--load-from", default=None, type=str)
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-log-file", action="store_true")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--output-root", default="../artifacts")
    parser.add_argument(
        "--opts",
        help="""
Modify config options at the end of the command, use "path.key=value".
        """.strip(),
        default=[],
        nargs=argparse.ZERO_OR_MORE,
    )
    parser.add_argument("--out", default=None, type=str)
    return parser.parse_args()


@torch.no_grad()
def do_test(cfg, model):
    logger = get_logger("FWI")
    logger.info("Evaluation start")

    if cfg.get("val_loader", None) is not None:
        val_loader = cfg.val_loader
    else:
        val_loader = instantiate(cfg.dataloader.val)
        cfg.val_loader = val_loader

    model.eval()
    losses = []
    for batch in val_loader:
        seis = batch["seis"].cuda()
        vel = batch["vel"].cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred_vel = model(seis)
            pred_vel = pred_vel.float()
        # pred_vel = F.softplus(pred_vel).sqrt() * 10000
        loss = F.l1_loss(pred_vel, vel, reduction="none")
        loss = loss.mean([1, 2, 3])
        losses.append(loss.cpu())
    losses = torch.cat(losses, dim=0)
    logger.info("Evaluation prediction done")
    eval_results = val_loader.dataset.format_results(losses.numpy())
    logger.info("Evaluation end")
    return eval_results


def print_log(logger, log_dict):
    logger.info(
        f"Epoch [{log_dict['curr_epoch'] + 1}/{log_dict['max_epochs']}] "
        f"Iter [{log_dict['curr_iter'] + 1}/{log_dict['max_iters']}] "
        f"lr: {log_dict['lr']:.4e}, "
        f"loss: {log_dict['loss']:.4f}, "
        f"grad_norm: {log_dict['grad_norm']:.4f}, "
        f"max_mem: {log_dict['max_mem']:.0f}M"
    )


def train_epoch(
    cfg, model, optimizer, scaler, lr_scheduler, train_loader, logger, log_dict
):
    model.train()
    log_dict["max_iters"] = len(train_loader)
    log_dict["loss"] = 0
    log_dict["n_samples"] = 0
    for curr_iter, batch in enumerate(train_loader):
        seis = batch["seis"].cuda()
        vel = batch["vel"].cuda()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred = model(seis)
            pred = pred.float()
        # loss = F.l1_loss(F.softplus(pred).sqrt() * 10000, vel)
        loss = F.l1_loss(pred, vel)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.train.get("clip_grad", False):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                **cfg.train.clip_grad,
            )
        else:
            grad_norm = get_total_norm(model)
        scaler.step(optimizer)
        scaler.update()
        log_dict["n_samples"] += seis.size(0)
        log_dict["loss"] += loss.item() * seis.size(0)
        lr_scheduler.step()

        log_dict["curr_iter"] = curr_iter
        log_dict["lr"] = lr_scheduler.get_last_lr()[0]
        log_dict["grad_norm"] = grad_norm.item()
        log_dict["max_mem"] = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        if (curr_iter + 1) % cfg.train.log_interval == 0:
            log_dict["loss"] /= log_dict["n_samples"]
            print_log(logger, log_dict)
            log_dict["loss"] = 0
            log_dict["n_samples"] = 0


def do_train(cfg, model):
    # cfg.optimizer.params = filter(lambda p: p.requires_grad, model.parameters())
    cfg.optimizer.model = model
    optimizer = instantiate(cfg.optimizer)
    scaler = torch.amp.GradScaler()

    train_loader = instantiate(cfg.dataloader.train)
    max_epochs = cfg.train.max_epochs
    lr_scheduler = LRMultiplier(
        optimizer,
        multiplier=instantiate(cfg.lr_multiplier),
        max_iter=max_epochs * len(train_loader),
    )
    best_param_group_id = LRScheduler.get_best_param_group_id(optimizer)

    logger = get_logger("FWI")
    log_dict = dict(max_epochs=max_epochs)

    for curr_epoch in range(max_epochs):
        log_dict["curr_epoch"] = curr_epoch
        train_epoch(
            cfg, model, optimizer, scaler, lr_scheduler, train_loader, logger, log_dict
        )
        if (curr_epoch + 1) % cfg.train.checkpoint_interval == 0:
            checkpoint_path = Path(cfg.train.work_dir) / f"epoch_{curr_epoch + 1}.pth"
            logger.info(f"Save checkpoint: {checkpoint_path}")
            torch.save(
                dict(model=model.state_dict(), optimizer=optimizer.state_dict()),
                checkpoint_path,
            )
            logger.info("Done.")

        eval_results = do_test(cfg, model)
        logger.info(f"Evaluation result: \n{eval_results}")


def setup(args):
    cfg = LazyConfig.load(args.config)
    # default work_dir
    cfg_path = Path(args.config)
    work_dir_root = Path(args.output_root)
    work_dir = str(work_dir_root / cfg_path.relative_to("configs/").with_suffix(""))
    cfg.train.work_dir = work_dir
    # override config
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    Path(cfg.train.work_dir).mkdir(parents=True, exist_ok=True)

    # dump config
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not args.eval_only:
        # LazyConfig.save(cfg, str(Path(work_dir) / f"{timestamp}.yaml"))
        shutil.copy(args.config, Path(work_dir) / f"{timestamp}.py")

    # logger
    if args.eval_only or args.no_log_file:
        log_file = None
    else:
        log_file = Path(work_dir) / f"{timestamp}.log"
    logger = get_logger("FWI", log_file=log_file)
    logger.info("Start")

    # seed
    if args.seed >= 0:
        seed = args.seed
    else:
        seed = cfg.train.get("seed", 0)
    seed_all_rng(seed)
    logger.info(f"Set random seed: {seed}")

    return cfg


def main():
    args = parse_args()
    cfg = setup(args)
    model = instantiate(cfg.model)
    logger = get_logger("FWI")
    if args.init_only:
        init_path = Path(cfg.train.work_dir) / "initialized.pth"
        torch.save(model.state_dict(), init_path)
        logger.info(f"Saved initialized model: {init_path}")

    if cfg.train.get("cast_to_bf16", False):
        logger.info("Casting model to BF16")
        # for name, m in model.named_modules():
        #     m.to(torch.bfloat16)
        for p in model.parameters():
            p.data = p.data.to(torch.bfloat16)

    load_from = cfg.train.get("load_from", None)
    if args.load_from is not None:
        load_from = args.load_from

    if load_from is not None:
        checkpoint = torch.load(load_from, map_location="cpu", weights_only=True)
        if "model" not in checkpoint:
            checkpoint = {"model": checkpoint}
        load_result = model.load_state_dict(checkpoint["model"], strict=False)
        logger.info(f"Load checkpoint: {load_from}")
        logger.info(f"Load checkpoint: {load_result}")

    model = model.cuda()
    if cfg.train.get("compile", None) is not None and not args.eval_only:
        model = torch.compile(model, **cfg.train.compile)
    # model = torch.compile(model, mode="max-autotune")

    if args.eval_only:
        eval_results = do_test(cfg, model)
        logger.info(f"Evaluation result: \n{eval_results}")
        if args.out is not None:
            pass
    else:
        do_train(cfg, model)


if __name__ == "__main__":
    main()
