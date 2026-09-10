"""Report host RAM and cgroup memory limits of a Modal H200:8 container."""

import modal

app = modal.App("probe-h200-hostmem")


@app.function(gpu="H200:8", memory=(1024, int(2 * 1024 * 1024)), timeout=300)
def probe() -> str:
    import subprocess

    out = []
    for f in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.high",
        "/sys/fs/cgroup/memory.swap.max",
    ]:
        try:
            out.append(f"{f}: {open(f).read().strip()}")
        except Exception as e:
            out.append(f"{f}: {e}")
    out.append(
        subprocess.run(
            ["head", "-5", "/proc/meminfo"], capture_output=True, text=True
        ).stdout
    )
    out.append(subprocess.run(["nproc"], capture_output=True, text=True).stdout)
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe.remote())
