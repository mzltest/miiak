"""Train the multi-task Mii attribute model.

Config-driven (YAML) with CLI overrides. Single-process on CPU/one GPU, or
multi-GPU via ``torchrun`` (DistributedDataParallel). AMP + cosine schedule with
warmup + checkpoint/resume + CSV logging.

CPU smoke test::

    python -m m2a.train --config configs/smoke_cpu.yaml --max-steps 40

Single GPU::

    python -m m2a.train --config configs/gpu_default.yaml

Multi-GPU (e.g. 4 GPUs)::

    torchrun --nproc_per_node=4 -m m2a.train --config configs/gpu_default.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn

from .data import make_loaders
from .model import build_model
from .losses import MultiTaskCriterion, MetricAccumulator


# ---------------------------------------------------------------------------
# config + distributed helpers
# ---------------------------------------------------------------------------
def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def setup_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False, 0, 1, 0
    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return True, rank, world, local


def is_main(rank):
    return rank == 0


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, prog)))


# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx, channels_last=False, det_model=None):
    model.eval()
    if det_model:
        det_model.eval()
    acc = MetricAccumulator()
    for batch in loader:
        xb = batch[0].to(device, non_blocking=True)
        yb = batch[1]
        if channels_last:
            xb = xb.to(memory_format=torch.channels_last)
        yb = {k: v.to(device, non_blocking=True) for k, v in yb.items()}
        with autocast_ctx():
            if det_model is not None:
                boxes = det_model(xb)
                out = model(xb, boxes)
            elif len(batch) > 2 and getattr(model, "two_step", False):
                # use ground truth boxes during val if no det_model is provided
                boxes = {k: v.to(device) for k, v in batch[2].items()}
                out = model(xb, boxes)
            else:
                out = model(xb)
        acc.update(out, yb)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="cap total steps (smoke)")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data: cfg["data"] = args.data
    if args.out: cfg["out"] = args.out
    if args.backbone: cfg.setdefault("model", {})["backbone"] = args.backbone
    tcfg = cfg.setdefault("train", {})
    if args.epochs is not None: tcfg["epochs"] = args.epochs
    if args.batch_size is not None: tcfg["batch_size"] = args.batch_size
    if args.max_steps is not None: tcfg["max_steps"] = args.max_steps

    distributed, rank, world, local = setup_distributed()
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")

    seed = int(tcfg.get("seed", 0)) + rank
    torch.manual_seed(seed)
    if device.type == "cpu":
        torch.set_num_threads(max(1, int(tcfg.get("cpu_threads", torch.get_num_threads()))))

    out_dir = cfg["out"]
    if is_main(rank):
        os.makedirs(out_dir, exist_ok=True)

    image_size = int(tcfg.get("image_size", 224))
    batch_size = int(tcfg.get("batch_size", 64))
    num_workers = int(tcfg.get("num_workers", 4))

    tr_ds, va_ds, train_loader, val_loader = make_loaders(
        cfg["data"], image_size, batch_size, num_workers)

    # distributed sampler for train
    if distributed:
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        from .data import collate_fn
        sampler = DistributedSampler(tr_ds, num_replicas=world, rank=rank, shuffle=True,
                                     drop_last=True)
        train_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, collate_fn=collate_fn,
                                  drop_last=True, pin_memory=torch.cuda.is_available())
    else:
        sampler = None

    model = build_model(cfg).to(device)

    # Check if two step training is enabled
    two_step = cfg.get("train", {}).get("two_step", False)
    det_model = None
    if two_step:
        from .detection_model import build_detection_model
        det_model = build_detection_model(cfg).to(device)

    channels_last = bool(tcfg.get("channels_last", True))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)  # NHWC: faster conv on CPU(oneDNN)/GPU
        if det_model:
            det_model = det_model.to(memory_format=torch.channels_last)

    if is_main(rank):
        print(f"device={device} backbone={model.backbone_name} "
              f"params={model.num_parameters()/1e6:.1f}M heads={len(model.heads)} "
              f"train={len(tr_ds)} val={len(va_ds)}")
    if distributed:
        ddp_kw = {"device_ids": [local]} if torch.cuda.is_available() else {}
        model = nn.parallel.DistributedDataParallel(model, **ddp_kw)
        if det_model:
            det_model = nn.parallel.DistributedDataParallel(det_model, **ddp_kw)

    criterion = MultiTaskCriterion(label_smoothing=float(tcfg.get("label_smoothing", 0.0)))

    # Collect parameters
    params = list(model.parameters())
    if det_model:
        params += list(det_model.parameters())

    opt = torch.optim.AdamW(params, lr=float(tcfg.get("lr", 1e-3)),
                            weight_decay=float(tcfg.get("weight_decay", 0.05)))

    use_amp = bool(tcfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast_ctx():
        if device.type == "cuda":
            return torch.autocast("cuda", enabled=use_amp)
        import contextlib
        return contextlib.nullcontext()

    epochs = int(tcfg.get("epochs", 10))
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup = int(tcfg.get("warmup_steps", 0))
    base_lr = float(tcfg.get("lr", 1e-3))
    grad_clip = float(tcfg.get("grad_clip", 0))
    max_steps = int(tcfg.get("max_steps", 0))
    log_int = int(tcfg.get("log_interval_steps", 50))

    start_epoch, gstep, best = 0, 0, -1.0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location=device)
        (model.module if distributed else model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"] + 1
        gstep = ck.get("gstep", 0)
        best = ck.get("best", -1.0)
        if is_main(rank):
            print(f"resumed from {args.resume} @ epoch {start_epoch}")

    csv_path = os.path.join(out_dir, "log.csv")
    if is_main(rank) and not (args.resume and os.path.isfile(csv_path)):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "gstep", "train_loss", "val_mean_acc",
                                    "val_exact", "lr", "sec"])

    stop = False
    for epoch in range(start_epoch, epochs):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        if det_model:
            det_model.train()
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for batch in train_loader:
            xb = batch[0]
            yb = batch[1]
            lr = cosine_lr(gstep, total_steps, base_lr, warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            xb = xb.to(device, non_blocking=True)
            if channels_last:
                xb = xb.to(memory_format=torch.channels_last)
            yb = {k: v.to(device, non_blocking=True) for k, v in yb.items()}

            # Ground truth bounding boxes for two step
            gt_boxes = None
            if two_step and len(batch) > 2:
                gt_boxes = {k: v.to(device, non_blocking=True) for k, v in batch[2].items()}

            opt.zero_grad(set_to_none=True)
            with autocast_ctx():
                loss = 0.0
                if two_step:
                    # Train bounding box regressor if we have ground truth
                    if gt_boxes is not None:
                        pred_boxes = det_model(xb)
                        box_loss = 0.0
                        for k in gt_boxes:
                            # L1 loss for bounding boxes
                            box_loss += torch.nn.functional.l1_loss(pred_boxes[k], gt_boxes[k])
                        loss += box_loss * 0.1 # weighting for box loss

                        # Forward pass model with predicted boxes to make the gradient flow all the way
                        out = model(xb, pred_boxes)
                    else:
                        out = model(xb, det_model(xb))
                else:
                    out = model(xb)

                cls_loss, _ = criterion(out, yb)
                loss += cls_loss

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach())
            nb += 1
            gstep += 1
            if is_main(rank) and gstep % log_int == 0:
                print(f"  e{epoch} step {gstep}/{total_steps} loss {run_loss/nb:.3f} lr {lr:.2e}")
            if max_steps and gstep >= max_steps:
                stop = True
                break

        # eval (rank 0)
        if is_main(rank):
            eval_det = det_model.module if distributed and det_model else det_model
            eval_mod = model.module if distributed else model
            acc = evaluate(eval_mod, val_loader, device, autocast_ctx, channels_last, eval_det)
            mean_acc, exact = acc.mean_acc(), acc.exact_match()
            dt = time.time() - t0
            print(f"[epoch {epoch}] train_loss {run_loss/max(1,nb):.3f} "
                  f"val_mean_acc {mean_acc:.4f} val_exact {exact:.4f} ({dt:.1f}s)")
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, gstep, f"{run_loss/max(1,nb):.4f}",
                                        f"{mean_acc:.4f}", f"{exact:.4f}",
                                        f"{opt.param_groups[0]['lr']:.3e}", f"{dt:.1f}"])
            ck = {"model": (model.module if distributed else model).state_dict(),
                  "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                  "epoch": epoch, "gstep": gstep, "best": best, "cfg": cfg,
                  "head_num_classes": (model.module if distributed else model).head_num_classes}
            if det_model:
                ck["det_model"] = (det_model.module if distributed else det_model).state_dict()
            torch.save(ck, os.path.join(out_dir, "last.pt"))
            if mean_acc > best:
                best = mean_acc
                ck["best"] = best
                torch.save(ck, os.path.join(out_dir, "best.pt"))
                print(f"  saved best (mean_acc={best:.4f})")
        if stop:
            break

    if distributed:
        import torch.distributed as dist
        dist.destroy_process_group()
    if is_main(rank):
        print("training complete; checkpoints in", out_dir)


if __name__ == "__main__":
    main()
