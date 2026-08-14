# Modal: the platform

Enough to write and run your own Modal code. Training-app patterns are in
`training.md`; the interactive SSH box is in `sandbox.md`.

## Documentation

1. https://modal.com/llms.txt outlines all docs: Guide pages (features and
   workflows), Examples pages (full applications), API Reference pages
   (signatures and docstrings). Fetch the page you need while planning.
2. Do not read https://modal.com/llms-full.txt into context — it is one
   very large file.
3. Your prior knowledge of Modal may be stale: `modal --version` shows the
   SDK in use, `modal changelog --since <date>` lists what changed since.

## The CLI

`modal --help` lists every command; read `modal <command> --help` rather
than guessing. Most commands accept `--json` for parseable output.

```bash
modal run app.py --config x     # ephemeral app, stops when the run ends
modal run app.py::fn --n 4      # call one specific function
modal deploy app.py             # live until `modal app stop`
modal serve app.py              # ephemeral, for developing web endpoints
modal app list                  # what's running
modal volume ls lab-out /models # inspect a volume (also get / put)
```

1. `modal run` takes flags directly after the filename; the old
   `-- --flag` separator syntax no longer works.
2. Two concurrent `modal run`s of the same app name conflict — `modal
   deploy` instead when you need concurrency.
3. A deployed app costs nothing while idle. `modal app stop` is
   destructive and cannot be reverted (rule 6: never stop the operator's
   apps).

## App structure

1. One Python file is enough: image, resources, and function logic in one
   place. Multi-file apps should be a Python package deployed in module
   mode (`modal deploy -m pkg.app`).
2. Local code goes into the image with `.add_local_file(...)` /
   `.add_local_dir(...)`.
3. Global scope runs both locally at deploy time and in every container at
   startup — don't read files, env vars, or import GPU-only packages
   there. Keep it fast; `from_name()` constructors are lazy for this
   reason.

## Functions vs classes

`@app.function()` for plain jobs. `@app.cls()` when the container should do
expensive setup once (e.g. load model weights) and then serve many calls:

```python
@app.cls(gpu="H200")
class Server:
    @modal.enter()
    def load(self):
        self.model = load_checkpoint("/out/models/my-tag/merged")

    @modal.method()
    def generate(self, prompt: str): ...
```

## CUDA

The NVIDIA driver is pre-installed on every GPU container. PyTorch,
Transformers, vLLM, SGLang etc. bundle their own CUDA runtime — just
`pip_install` them. Only use an `nvidia/cuda:*-devel-*` base image when you
need the full CUDA toolkit (nvcc, custom kernels).
