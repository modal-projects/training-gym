# Interactive GPU sandbox

An H200 box you SSH into: debug, profile (nsys/ncu), prototype, test
serving. Pre-installed: PyTorch, Transformers, SGLang, datasets, nvitop,
tmux, vim, git. The shared volumes are mounted (`/out` for checkpoints,
`/hf-cache` as `HF_HOME` — the base model is already cached), and
`/root/workspace` persists to the `gpu-sandbox-workspace` volume (synced
every 30 s). Auto-terminates after 4 hours; the launcher writes the GPU
accounting line to `runs/GPU_LOG.jsonl` when it exits.

## Launch

```bash
# One-time, if you have no SSH key yet:
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519

_SANDBOX_GPU=H200 python -m modal run toolbox/gpu_tools/gpu_sandbox.py \
  --key-path ~/.ssh/id_ed25519.pub --sandbox-id my-box > /tmp/sandbox.log 2>&1 &
SANDBOX_PID=$!

# Wait for the SSH info (written to the volume — reliable, not log-buffered)
while true; do
  modal volume get gpu-sandbox-workspace /ssh-info/my-box.json /tmp/ssh-info.json 2>/dev/null && break
  sleep 3
done
HOST=$(python3 -c "import json; print(json.load(open('/tmp/ssh-info.json'))['host'])")
PORT=$(python3 -c "import json; print(json.load(open('/tmp/ssh-info.json'))['port'])")
```

`_SANDBOX_GPU` is `H200`, `H200:N`, or `cpu` (README rule 2 — never another
GPU type). It must be set on the launch command itself: Modal reads the GPU
spec when the file is imported.

## Use it

```bash
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i ~/.ssh/id_ed25519"

$SSH -p $PORT root@$HOST "nvidia-smi"

# Upload / download
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.ssh/id_ed25519 -P $PORT ./local_file.py root@$HOST:/tmp/
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -i ~/.ssh/id_ed25519 -P $PORT root@$HOST:/tmp/results.json ./

# Sync a directory
rsync -avz --exclude __pycache__/ --exclude .venv/ \
  -e "$SSH -p $PORT" ./my_code/ root@$HOST:/tmp/my_code/

# Background long commands so an SSH timeout doesn't kill them
$SSH -p $PORT root@$HOST "nohup python train.py > /tmp/train.log 2>&1 & echo PID=\$!"
$SSH -p $PORT root@$HOST "tail -20 /tmp/train.log"
```

## Workflow

1. Launch, upload your code, SSH in, iterate: run, debug, profile
   (`nsys profile python train.py`, then scp the trace back).
2. Anything you want to keep across sandboxes goes in `/root/workspace`
   (persists on the `gpu-sandbox-workspace` volume) or on `/out`.
3. Once the code works, write a proper Modal app (`training.md`) —
   sandboxes are for iteration, not production runs.
4. Kill it when done: `kill $SANDBOX_PID`.
