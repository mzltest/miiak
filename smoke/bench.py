import sys, time, resource, gc, torch
sys.path.insert(0, "/agent/workspace/mii2attr")
from m2a.model import MultiHeadMiiNet
from m2a.losses import MultiTaskCriterion
from m2a.schema import HEAD_FIELDS, FIELDS

name = sys.argv[1]; B = int(sys.argv[2]); mode = sys.argv[3]
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 2
res = int(sys.argv[5]) if len(sys.argv) > 5 else 224
torch.set_num_threads(threads)
crit = MultiTaskCriterion(label_smoothing=0.05)
y = {h: torch.randint(0, FIELDS[h].num_classes, (B,)) for h in HEAD_FIELDS}
cl = len(sys.argv) > 6 and sys.argv[6] == "cl"
net = MultiHeadMiiNet(name, pretrained=False)
if cl:
    net = net.to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
x = torch.randn(B, 3, res, res)
if cl:
    x = x.to(memory_format=torch.channels_last)
warm, iters = 1, 4
net.train() if mode == "train" else net.eval()
t = None
for i in range(warm + iters):
    if i == warm:
        t = time.time()
    if mode == "train":
        opt.zero_grad(set_to_none=True)
        out = net(x); loss, _ = crit(out, y); loss.backward(); opt.step()
    else:
        with torch.no_grad():
            net(x)
dt = (time.time() - t) / iters
ips = B / dt
mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
p = sum(q.numel() for q in net.parameters()) / 1e6
print(f"{name:<22} {mode:<5} thr={threads} {res}px B={B:<3} {dt:6.2f}s/step {ips:6.1f} img/s "
      f"params={p:5.1f}M peakRAM~{mb:5.0f}MB")
