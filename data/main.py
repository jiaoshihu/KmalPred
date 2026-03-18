#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import ProteinRegionDataset
from src.losses import FocalLoss
from src.metrics import compute_binary_metrics
from src.models import LSTMClassifier
from src.record import save_metrics_table
from src.utils import get_device, set_seed, save_json




@torch.inference_mode()
def evaluate(net, loader, criterion, device, thr=0.5):
    net.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    n = 0

    for x, y, mask, _, _ in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        mask = mask.to(device)

        logits = net(x, mask)                  # [B]
        probs = torch.sigmoid(logits)          # [B]

        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)

        all_y.append(y.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)

    metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, thr=thr)
    avg_loss = total_loss / max(1, n)
    return metrics, avg_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_emb", type=str, required=True)
    p.add_argument("--train_pos", type=str, required=True)
    p.add_argument("--train_neg", type=str, required=True)
    p.add_argument("--test_emb", type=str, required=True)
    p.add_argument("--test_pos", type=str, required=True)
    p.add_argument("--test_neg", type=str, required=True)

    p.add_argument("--win", type=int, default=35)
    p.add_argument("--input_dim", type=int, default=1024)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)

    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--pos_weight", type=float, default=4.0)

    p.add_argument("--neg_limit_train", type=int, default=None)
    p.add_argument("--neg_limit_test", type=int, default=None)
    p.add_argument("--shuffle_before_limit", action="store_true")

    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--outdir", type=str, default="runs/lstm")
    p.add_argument("--save_best", action="store_true", help="Save best checkpoint by MCC on test set.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), str(outdir / "config.json"))

    train_ds = ProteinRegionDataset(
        emb_file=args.train_emb,
        pos_region_file=args.train_pos,
        neg_region_file=args.train_neg,
        max_len=args.win,
        neg_limit=args.neg_limit_train,
        shuffle_before_limit=args.shuffle_before_limit,
        seed=args.seed,
    )
    test_ds = ProteinRegionDataset(
        emb_file=args.test_emb,
        pos_region_file=args.test_pos,
        neg_region_file=args.test_neg,
        max_len=args.win,
        neg_limit=args.neg_limit_test,
        shuffle_before_limit=args.shuffle_before_limit,
        seed=args.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = LSTMClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=True,
        dropout=args.dropout,
    ).to(device)

    if args.use_focal:
        pos_w = torch.tensor(args.pos_weight, dtype=torch.float32, device=device)
        criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma, pos_weight=pos_w).to(device)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_mcc = -1e9
    history = []

    for epoch in range(1, args.epochs + 1):
        net.train()
        total_loss = 0.0
        n = 0

        for x, y, mask, _, _ in train_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            mask = mask.to(device)

            logits = net(x, mask)
            loss = criterion(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

        train_loss = total_loss / max(1, n)
        test_metrics, test_loss = evaluate(net, test_loader, criterion, device, thr=args.thr)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            **{k: test_metrics[k] for k in ["BACC", "ACC", "AUC", "SN", "SP", "MCC", "F1", "Precision", "Kappa", "AP"]},
        }
        history.append(row)

        print(f"[Epoch {epoch:03d}] "
              f"loss={train_loss:.4f} | test_mcc={test_metrics['MCC']:.4f} "
              f"acc={test_metrics['ACC']:.4f} auc={test_metrics['AUC']:.4f} ap={test_metrics['AP']:.4f}")

        # save best
        if args.save_best and test_metrics["MCC"] > best_mcc:
            best_mcc = test_metrics["MCC"]
            ckpt = {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "best_mcc": best_mcc,
                "config": vars(args),
            }
            torch.save(ckpt, outdir / "best.pt")

    # save logs
    save_metrics_table(history, str(outdir / "metrics.csv"))
    if (outdir / "best.pt").exists():
        print(f"[OK] saved best checkpoint: {outdir/'best.pt'}")
    print(f"[OK] saved metrics: {outdir/'metrics.csv'}")


if __name__ == "__main__":
    main()
