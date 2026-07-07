"""Multi-task loss + metrics.

Total loss = (optionally weighted) sum of per-head cross-entropy. Heads whose
label is ``ignore_index`` (-100) for a sample contribute nothing for that
sample (irrelevant / invisible attribute). If an entire batch is ignored for a
head, that head is skipped for the batch (no NaN).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import HEAD_FIELDS

IGNORE = -100


class MultiTaskCriterion(nn.Module):
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.weights = weights or {}
        self.label_smoothing = label_smoothing

    def forward(self, logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
        total = None
        per_head = {}
        for h in HEAD_FIELDS:
            y = labels[h]
            if (y != IGNORE).sum() == 0:
                continue
            loss_h = F.cross_entropy(logits[h], y, ignore_index=IGNORE,
                                     label_smoothing=self.label_smoothing)
            per_head[h] = loss_h.detach()
            w = float(self.weights.get(h, 1.0))
            total = (w * loss_h) if total is None else total + w * loss_h
        if total is None:
            # degenerate batch; return a connected-to-graph zero
            any_logit = next(iter(logits.values()))
            total = any_logit.sum() * 0.0
        return total, per_head


@torch.no_grad()
def head_accuracies(logits: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]):
    """Return {head: (correct, count)} ignoring -100, plus exact-match counts."""
    stats = {}
    B = next(iter(logits.values())).shape[0]
    all_correct = torch.ones(B, dtype=torch.bool, device=next(iter(logits.values())).device)
    any_valid = torch.zeros(B, dtype=torch.bool, device=all_correct.device)
    for h in HEAD_FIELDS:
        y = labels[h]
        pred = logits[h].argmax(1)
        valid = y != IGNORE
        correct = (pred == y) & valid
        stats[h] = (int(correct.sum()), int(valid.sum()))
        # exact-match accounting over visible heads
        all_correct &= (~valid | (pred == y))
        any_valid |= valid
    exact = int((all_correct & any_valid).sum())
    return stats, exact, int(any_valid.sum())


class MetricAccumulator:
    """Aggregate head accuracies across batches."""

    def __init__(self):
        self.correct = {h: 0 for h in HEAD_FIELDS}
        self.count = {h: 0 for h in HEAD_FIELDS}
        self.exact = 0
        self.exact_total = 0

    def update(self, logits, labels):
        stats, exact, etot = head_accuracies(logits, labels)
        for h, (c, n) in stats.items():
            self.correct[h] += c
            self.count[h] += n
        self.exact += exact
        self.exact_total += etot

    def head_acc(self) -> Dict[str, float]:
        return {h: (self.correct[h] / self.count[h] if self.count[h] else float("nan"))
                for h in HEAD_FIELDS}

    def mean_acc(self) -> float:
        accs = [a for a in self.head_acc().values() if a == a]  # drop nan
        return sum(accs) / len(accs) if accs else float("nan")

    def exact_match(self) -> float:
        return self.exact / self.exact_total if self.exact_total else float("nan")
