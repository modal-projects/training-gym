# DeepSeek-V4.1-Flash — conversion failures (runs 1-6)

All runs: `uv run scripts/validate_model_configs.py check -m DeepSeek-V4.1-Flash -n 1 --no-wandb`.
Logs in `.gym/dsv41_step{N}.log`.

| run | conversion layout | failure | fix |
|---|---|---|---|
| 1 | TP8/PP1/EP8, 1 node | `--dsv4-impl megatron requires tensor-model-parallel-size 1` | forward `dsv4_impl` to the converter (`_CONVERSION_EXTRA_ARGS`) |
| 2 | TP8/PP1/EP8, 1 node | CUDA OOM building the bf16 model on H200 (~121 GiB/rank before overhead) | spread over 2 nodes |
| 3 | TP8/PP2/EP8, 2 nodes | NCCL scatter mismatch `(16192, 5120) vs (16160, 5120)` on the vocab | vocab (129280) isn't 8-way splittable at default `make-vocab-size-divisible-by`; back to TP4 |
| 4 | TP4/PP4/EP4, 2 nodes | NCCL scatter mismatch `(4608, 5120) vs (4608, 2560)` | (misdiagnosed as a PP issue) |
| 5 | TP4/PP1/EP16, 2 nodes | same `(4608, 5120) vs (4608, 2560)` | root cause found, below |
| 6 | TP4/PP1/EP16, 2 nodes + dequant hook on `SafeTensorIO` | same | hook shadowed by `DequantFP8SafeTensorIO.load_some_hf_weight` |
| 7 | same + hook on `Bridge._get_safetensor_io` | same | hook shadowed by `DeepseekV3Bridge._get_safetensor_io` |

## Root cause of runs 4-6

`(4608, 2560)` is `experts.N.w1` + `w3` concatenated (2 x 2304 rows) with half the
expected columns. The public V4.1 checkpoint stores routed experts as packed e2m1 fp4
(`I8 [2304, 2560]`, two values per byte) with per-row e8m0 scales
(`experts.N.w1.scale F8_E8M0 [2304, 160]`), and dense weights as fp8 e4m3 with
(32, 32)-block e8m0 scales (`attn.wq_a.scale [40, 160]`). There is no bf16 export.

mbridge's V4.1 bridge (`miles_plugins/mbridge/deepseek_v41.py`) only maps names and
reshapes. Its base `DeepseekV3Bridge._get_safetensor_io` returns
`DequantFP8SafeTensorIO`, which dequantizes V3-style `<name>_scale_inv` tensors and
passes everything else through raw — so V4.1's `<name>.scale` tensors are ignored,
fp8 dense weights are upcast unscaled and fp4 experts arrive at half width.

Upstream presumably converts on GB300 from a bf16 checkpoint it prepared separately
(the PR's `run_deepseek_v41.py` takes `--hf-checkpoint` as given).

## Fix

`modal_helpers/hf_block_dequant.py`, enabled by `CONVERT_DEQUANT_HF_WEIGHTS=1` in the
recipe env: wraps `Bridge._get_safetensor_io` so the returned reader (whatever
subclass) dequantizes any weight with a sibling `.scale` tensor to bf16 — fp8 block
scales expanded (32, 32), packed fp4 nibbles unpacked (low nibble first) and scaled
per 32-column block. Runs 6 and 7 patched base-class methods that DeepSeek
subclasses override; run 8 hooks `Bridge.load_weights` (not overridden anywhere) and
wraps the reader instance the real subclass builds. The hook logs
`[hf_block_dequant] dequantizing block-scaled weights on load` when active.

fp4 unpacking (low nibble first, e2m1 table) and e8m0 decoding checked on CPU against
torch's own e8m0 cast; not yet verified end-to-end against a known-good bf16 export.

## Run 8: hook active, rank SIGKILLed

The hook fired on every rank and the `(4608, 2560)` scatter mismatch is gone; the
first expert layers loaded for ~30 s longer than run 7 before local rank 3 on node 0
died with `exitcode -9 (SIGKILL)` and no Python traceback (torchrun then SIGTERMed
the rest, whose tracebacks show them mid-`load_weights`). SIGKILL with no traceback
is a host/cgroup memory kill, not a Python error (GPU OOM would raise). Nothing in
the load path should hold host memory: mbridge loads with `memory_efficient=True`
(one HF tensor at a time), `DequantFP8SafeTensorIO` opens safetensors on `cuda`, and
the model is built on GPU. safetensors maps each shard with
`torch.UntypedStorage.from_file(shared=False)` (MAP_PRIVATE mmap of the whole file on
the FUSE-backed HF cache volume) per `safe_open`, i.e. per weight — file-backed pages
that should be reclaimable, but that is the only large host-side footprint in sight.

`tools/convert_mxfp4_to_fp8.py` upstream confirms the fp4 layout used here (int8
packed low-nibble-first, per-(1,32) ue8m0 scales `[out, in/32]`).

Run 9 adds a `[convert-mem]` line every 30 s from local rank 0 (cgroup memory.max /
memory.current, MemTotal, MemAvailable, summed python RSS) to confirm or rule out
memory pressure before changing the load path.

## Run 9: memory kill confirmed

`[convert-mem]` showed summed python RSS on node 0 climbing 68 -> 83 -> 93 -> 129 ->
151 GiB in ~2 min while `MemAvailable` barely moved (1 TiB box); two ranks were
SIGKILLed at ~130 GiB summed RSS. So the kill is per-process, not host-wide: it's the
mmap'd shard pages. `safe_open` maps each whole shard (~5-10 GB) `MAP_PRIVATE` and
mbridge loads one weight at a time, so every load re-maps a shard and the mapped file
pages accumulate against each rank's RSS until the sandbox kills it.

Run 10 replaces the reader: `hf_block_dequant.ShardReader` parses shard headers once
and `pread`s each tensor's byte range into a fresh buffer, then moves it to the GPU;
no mmap. Checked on CPU against synthetic fp4/fp8/e8m0 shards
(`.gym/test_hf_block_dequant.py`).

## Run 10: conversion completed; completeness check rejected it

With the positional reader, all 16 ranks loaded the dequantized model (69 GB
allocated per H200, summed host RSS flat at ~97 GiB) and Megatron saved
`iter_0000001` (16 x 64 GiB `.distcp`, 21.7 MiB `.metadata`, `metadata.json`,
tracker `1`) in ~8 min; the volume commit then took ~24 min. Node 0's post-commit
check still raised "no complete torch_dist checkpoint": `_is_complete_torch_dist_checkpoint`
demanded `common.pt`, which this megatron-core no longer writes (common state
lives in the torch_dist metadata). The check now requires only `.metadata` plus
`.distcp` shards; the saved checkpoint is intact on the volume and run 11 starts
from it as a cache hit.

## Run 11: cache hit, then engine launch rejected by miles' argv round-trip

The saved checkpoint was accepted (no re-conversion) and the 64-GPU job placed,
but every attempt died ~2.5 min in at `RayWorkerManager.post_setup`:

    AssertionError: cli argv roundtrip mismatch on device: parsed 'None' != wanted None

miles#3179 sits on Sept-4 main, before miles#3124 (sglang v0.5.19 bump) added
`_explicit_device` to `server_args_to_argv`. sglang#38798 follows current sglang,
where `ServerArgs.__post_init__` no longer resolves `device` (sglang#35907), so
the always-rendered `--device` gets the string `None`. Fixed on the recipe side by
naming the device (`extra_config={"sglang_device": "cuda"}`) rather than patching
miles source. Modal retried the crashed train function four times before the app
was stopped by hand.

## Run 12: engines start; dataset rendering has no V4.1 chat template

With the device named, the 64-GPU job got past engine-command construction, but
`RolloutExecutor.__init__` died rendering the first prompt:

    ValueError: Cannot use chat template functions because tokenizer.chat_template is not set

DeepSeek-V4.1-Flash ships no jinja template (`model_type: deepseek_v41`); sglang
#38798 adds `encoding_dsv41` for it, but miles#3179 never registers a
`deepseek_v41` family in `miles/utils/chat_template_utils/deepseek.py`, so
`--apply-chat-template` falls through to the HF tokenizer. Fixed with a
build-time patch (`patch_deepseek_v41_chat_template`) that adds the family on the
existing `DeepSeekFamily` bridge, applied through the recipe's
`image_run_commands` after the source overlays. Modal retried the crashed train
function twice before the app was stopped by hand.

## Run 13: prompts render, Megatron loads; sglang rejects the packed-FP4 experts

The template patch worked (dataset prepared), all 64 Megatron ranks loaded the
torch_dist checkpoint, and the eight sglang engines began loading HF weights,
then every scheduler died in `FusedMoE._load_w2`:

    RuntimeError: The size of tensor a (160) must match the size of tensor b (5120)

Upstream's `run_deepseek_v41.py` sets `SGLANG_DSV4_FP4_EXPERTS=0` because it
serves an FP4→FP8 pre-converted checkpoint. We load the public release, whose
routed experts are still packed mxfp4 (the K dim arrives as 5120/32 = 160 scale
blocks), so the FP8 loader shapes never match. Fixed by declaring the experts
FP4 and taking sglang's Hopper path: `SGLANG_DSV4_FP4_EXPERTS=1` plus
`SGLANG_DSV4_FP4_DEQUANT=1` (`Fp8MoEMethod` dequantizes e2m1 → e4m3 after load,
mxfp4 kernels are never selected). Modal retried the crashed train function
before the app was stopped by hand.

## Run 14: engines load every weight; no static budget left for the KV pool

With the FP4 experts declared and dequantized, all eight engines completed the
HF load (`Load weight end ... quant=fp8, mem usage=111.65 GB`, Megatron having
offloaded to 136.9 GB free beforehand), then failed sizing the KV cache:

    ValueError: Loaded weights leave no GPU memory for the KV cache under
    --mem-fraction-static=0.6. Raise --mem-fraction-static above 0.820

Upstream's 0.6 (84 GB of an H200) does not cover 112 GB of resident weights.
Raised `sglang_mem_fraction_static` to 0.9, leaving ~11 GB per GPU for the KV
pool of the smoke step. Modal retried the crashed train function before the
app was stopped by hand.

## Run 15: engines serve; first forward asserts on the dequantized scale shape

With a 0.9 static budget every engine reached "The server is fired up and ready
to roll!" (`load_weight=529 s`, `kv_cache_allocation=16 s`, ~13 GB left), but
the first generation died in the fused-MoE Triton runner:

    assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]

`cast_e2m1fn_to_e4m3fn` (sglang#38798) hardcodes 128x128 scale blocks for the
FP4→FP8 expert dequant, which matched DeepSeek-V4's `weight_block_size=[128,
128]`. DeepSeek-V4.1 declares `[32, 32]`, and the MoE runner keeps reading
`quant_config.weight_block_size`, so `(N/128, K/128)` scales met a kernel
configured for `(N/32, K/32)`. Miles' weight sync also quantizes trainer experts
at the config's block size, so the 128-block parameters would have been wrong
even without the assert. Fixed with a build-time patch
(`patch_deepseek_v41_fp4_dequant_block`) that lets the cast take the target
block size and passes the config's. Verified on one H100 against the real
recipe image (`.gym/probe_fp4_dequant_moe.py`): the 32-block cast is lossless
against a table dequant, `fused_moe(block_shape=[32, 32])` runs and matches an
fp32 reference to 4% mean relative error (fp8 activation quant noise). The
stuck app was stopped by hand.

## Run 16: one engine rank SIGKILLed while reading HF shards

Checkpoint cache hit, Megatron loaded and offloaded (62.5 GB/rank to host),
engines began loading. Seven nodes finished their 48 shards in 2.5-3 min; on
one node rank 0 died ~100 s into the load:

    RuntimeError: Rank 0 scheduler died during initialization (exit code: -9).
    If exit code is -9 (SIGKILL), a common cause is the OS OOM killer.

Nothing in the dequant patch touches host memory (it runs on the GPU in
`process_weights_after_loading`), so this is the load itself. Each of the 8 TP
ranks reads all 48 ~10 GB shards through sglang's buffered multi-thread loader,
which keeps `num_threads + 1 = 9` mmap'd shards in flight per rank, and the
Modal sandbox charges mmap'd file pages to the process (the same behaviour that
SIGKILLed the conversion in runs 8-9) — up to ~720 GB of page-cache pressure on
top of ~500 GB of offloaded Megatron weights on a 2 TB node (probed:
`MemTotal 2096911076 kB`, no cgroup limit exposed). Run 15 survived the same
load; run 16 did not, which is what a timing-dependent page-cache race looks
like. Fixed by bounding the loader through `extra_config`:
`sglang_weight_loader_disable_mmap` (eager reads, freed after each shard),
`sglang_weight_loader_drop_cache_after_load` (`posix_fadvise(DONTNEED)` per
shard) and `sglang_model_loader_extra_config='{"num_threads": 2}'` (3 shards in
flight per rank). Verified the overrides survive miles' ServerArgs argv
round-trip inside the recipe image (`.gym/probe_sglang_loader_argv.py`).

## Run 17: eager (non-mmap) shard reads cannot decode F8_E8M0

With `--weight-loader-disable-mmap` every rank failed on its first shard:

    File ".../safetensors/torch.py", line 456, in _getdtype
        return _TYPES[dtype_str]
    KeyError: 'F8_E8M0'

sglang's eager path deserializes with `safetensors.torch.load`, whose Python
dtype table in the image's safetensors has no e8m0 entry; the mmap path goes
through `safe_open`'s Rust reader, which does. Dropped `disable_mmap` and kept
the other two bounds (3 mmap'd shards in flight per rank instead of 9, page
cache released per shard). Stopped by hand.
