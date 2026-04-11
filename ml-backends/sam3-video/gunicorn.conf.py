# gunicorn configuration for SAM3 video backend
#
# post_fork hook — two responsibilities:
#   1. Reset PyTorch CUDA state (see sam3-image/gunicorn.conf.py for full explanation).
#   2. Per-worker GPU assignment when CUDA_VISIBLE_DEVICES lists multiple GPUs.
#      With WORKERS=N and N GPUs in CUDA_VISIBLE_DEVICES, worker i (age i) is pinned
#      to gpus[i-1], so each worker gets its own GPU.
#      Single-GPU case (one entry or empty): no-op, all workers use cuda:0 as before.


def post_fork(server, worker):
    """Reset PyTorch CUDA state and assign one GPU per worker (multi-GPU setups)."""
    import os

    # ── 1. Reset CUDA fork state ─────────────────────────────────────────────
    try:
        import torch.cuda as _cuda
        _cuda._initialized = False
        _cuda._in_bad_fork = False
    except Exception:
        pass

    # ── 2. Per-worker GPU pinning ────────────────────────────────────────────
    # worker.age: 1-indexed creation counter (first worker = 1, second = 2, …).
    # When CUDA_VISIBLE_DEVICES="0,1" and WORKERS=2:
    #   worker 1 → CUDA_VISIBLE_DEVICES="0"  → cuda:0 inside this process
    #   worker 2 → CUDA_VISIBLE_DEVICES="1"  → cuda:0 inside this process (= phys GPU 1)
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            gpus = [g.strip() for g in visible.split(",") if g.strip()]
            if len(gpus) > 1:
                gpu_idx = (worker.age - 1) % len(gpus)
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus[gpu_idx]
    except Exception:
        pass
