"""Verify torch and dgl are both CUDA-enabled and can see the GPU."""
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("compute cap:", torch.cuda.get_device_capability(0))
    print("torch built for cuda:", torch.version.cuda)
else:
    raise SystemExit("torch reports CUDA unavailable -- still on CPU wheel")

try:
    import dgl
    print("dgl:", dgl.__version__)
    g = dgl.graph(([0, 1], [1, 0])).to("cuda:0")
    print("dgl cuda graph created on:", g.device)
    print("ALL CUDA OK")
except Exception as e:
    print("dgl CUDA test failed:", repr(e))
    raise SystemExit("dgl is not CUDA-enabled -- reinstall with +cu117 wheel")
