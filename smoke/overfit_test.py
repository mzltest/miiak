"""Canonical 'can it learn?' test: overfit a tiny no-aug batch.

If the architecture + loss + optimizer are wired correctly, training loss should
collapse toward 0 and train accuracy should climb to ~1.0 on a small fixed set.
"""
import sys, time
sys.path.insert(0, "/agent/workspace/mii2attr")
import torch
from torch.utils.data import Subset, DataLoader
from m2a.data import MiiAttrDataset, collate_fn
from m2a.model import MultiHeadMiiNet
from m2a.losses import MultiTaskCriterion, MetricAccumulator

torch.manual_seed(0); torch.set_num_threads(2)
ds = MiiAttrDataset("data/smoke", "train", 224, train=False)   # no augmentation
sub = Subset(ds, list(range(32)))
loader = DataLoader(sub, batch_size=32, shuffle=False, collate_fn=collate_fn)
xb, yb = next(iter(loader))

net = MultiHeadMiiNet("resnet18", pretrained=False, head_hidden=0, dropout=0.0)
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.0)
crit = MultiTaskCriterion(label_smoothing=0.0)

net.train()
t0 = time.time()
for step in range(80):
    opt.zero_grad(set_to_none=True)
    out = net(xb)
    loss, _ = crit(out, yb)
    loss.backward(); opt.step()
    if step % 10 == 0 or step == 79:
        net.eval()
        with torch.no_grad():
            acc = MetricAccumulator(); acc.update(net(xb), yb)
        net.train()
        print(f"step {step:>3} loss {float(loss):7.3f} train_mean_acc {acc.mean_acc():.3f} "
              f"exact {acc.exact_match():.3f}")
print(f"done in {time.time()-t0:.1f}s")
