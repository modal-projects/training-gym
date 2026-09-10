"""Exercise the patched FP4->FP8 expert dequant + Triton fused MoE at 32x32 blocks.

Builds the exact recipe image (overlays + build-time patches), then on one H100:
dequantizes random packed-e2m1 experts through the patched
``cast_e2m1fn_to_e4m3fn`` at the checkpoint's ``weight_block_size`` (32), feeds
the result to the Triton ``fused_moe`` kernel with ``block_shape=[32, 32]`` as
``Fp8MoEMethod`` does for DeepSeek-V4.1, and compares against a fp32 reference.
"""

import modal

if modal.is_local():
    from modal_training_gym.frameworks.miles.launcher import (
        _build_miles_base_image,
        apply_source_overlays,
    )
    from modal_training_gym.train_recipes.miles_recipe import (
        DeepSeek_V4_1_Flash_Recipe,
    )

    recipe = DeepSeek_V4_1_Flash_Recipe()
    image = apply_source_overlays(_build_miles_base_image(recipe), recipe)
    image = image.run_commands(*recipe.image_run_commands)
else:
    image = modal.Image.debian_slim()

app = modal.App("probe-dsv41-fp4-dequant-moe", image=image)


@app.function(gpu="H100", timeout=1800)
def probe() -> str:
    import subprocess

    script = r"""
import torch
from sglang.srt.layers.quantization import fp8 as fp8_mod
from sglang.srt.layers.quantization.fp8 import Fp8Config, cast_e2m1fn_to_e4m3fn
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

print("patched:", "PATCHED_DEEPSEEK_V41_FP4_DEQUANT_BLOCK" in open(fp8_mod.__file__).read())
import json, os
os.makedirs("/tmp/dummy_model", exist_ok=True)
json.dump({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64,
           "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 4,
           "num_hidden_layers": 1, "vocab_size": 128, "max_position_embeddings": 128,
           "rms_norm_eps": 1e-6, "torch_dtype": "bfloat16"},
          open("/tmp/dummy_model/config.json", "w"))
publish(ServerArgs(model_path="/tmp/dummy_model", skip_tokenizer_init=True,
                   load_format="dummy"), role="test")

torch.manual_seed(0)
dev = "cuda"
torch.set_default_device(dev)
E, N, K = 8, 2304, 5120  # DSV4.1-Flash: moe_intermediate=2304, hidden=5120
topk = 6

cfg = Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",
                weight_block_size=[32, 32], is_fp4_experts=True)

def rand_fp4(out_dim, in_dim):
    packed = torch.randint(-128, 127, (E, out_dim, in_dim // 2), dtype=torch.int8)
    scale = torch.pow(2.0, torch.randint(-9, -3, (E, out_dim, in_dim // 32)).float())
    return packed, scale

table = fp8_mod.DSV4_DEQUANT_FP4_TABLE.to(dev)
def deq_ref(q, s):
    u8 = q.view(torch.uint8)
    x = torch.stack([table[(u8 & 0x0F).long()], table[(u8 >> 4).long()]], -1).flatten(2)
    return x * s.repeat_interleave(32, dim=2)

# Mirror the patched Fp8MoEMethod.process_weights_after_loading dequant branch.
def dequant_like_method(weight_param, scale_param):
    block_n, block_k = cfg.weight_block_size or [128, 128]
    assert block_n == block_k
    ws, ss = [], []
    for e in range(weight_param.shape[0]):
        w, s = cast_e2m1fn_to_e4m3fn(weight_param[e], scale_param[e], fp8_block_size=block_n)
        ws.append(w); ss.append(s)
    return torch.stack(ws), torch.stack(ss).float()

w13_q, w13_s = rand_fp4(2 * N, K)
w2_q, w2_s = rand_fp4(K, N)
w13, w13_scale = dequant_like_method(w13_q, w13_s)
w2, w2_scale = dequant_like_method(w2_q, w2_s)
print("w13", tuple(w13.shape), w13.dtype, "scale", tuple(w13_scale.shape), w13_scale.dtype)
print("w2 ", tuple(w2.shape), w2.dtype, "scale", tuple(w2_scale.shape), w2_scale.dtype)
assert w13_scale.shape == (E, 2 * N // 32, K // 32)
assert w2_scale.shape == (E, K // 32, N // 32)

def expand(w, s, bs=32):
    return w.float() * s.repeat_interleave(bs, 1).repeat_interleave(bs, 2)
w13_ref = deq_ref(w13_q, w13_s); w2_ref = deq_ref(w2_q, w2_s)
print("dequant lossless:", torch.equal(expand(w13, w13_scale), w13_ref), torch.equal(expand(w2, w2_scale), w2_ref))

from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
M = 64
a = torch.randn(M, K, dtype=torch.bfloat16) / 10
score = torch.randn(M, E, dtype=torch.bfloat16)

def dequant_at(weight_param, scale_param, bs):
    ws, ss = [], []
    for e in range(weight_param.shape[0]):
        w, s = cast_e2m1fn_to_e4m3fn(weight_param[e], scale_param[e], fp8_block_size=bs)
        ws.append(w); ss.append(s)
    return torch.stack(ws), torch.stack(ss).float()

with torch.inference_mode():
    topk_output = select_experts(hidden_states=a, router_logits=score,
                                 topk_config=TopKConfig(top_k=topk, renormalize=False))
    topk_w, topk_ids = topk_output.topk_weights, topk_output.topk_ids
    print("topk_w", topk_w.dtype, tuple(topk_w.shape), topk_w[0].tolist(), "ids", topk_ids[0].tolist())
    ref = torch.zeros(M, K)
    af = a.float()
    for i in range(M):
        for j in range(topk):
            e = int(topk_ids[i, j]); w = float(topk_w[i, j])
            g_u = af[i] @ w13_ref[e].T
            h = torch.nn.functional.silu(g_u[:N]) * g_u[N:]
            ref[i] += w * (h @ w2_ref[e].T)
    print("ref mean abs", ref.abs().mean().item())
    outs = {}
    for bs in (32, 128):
        w13b, w13sb = dequant_at(w13_q, w13_s, bs)
        w2b, w2sb = dequant_at(w2_q, w2_s, bs)
        out = fused_moe(a, w13b, w2b, topk_output, use_fp8_w8a8=True,
                        w1_scale=w13sb, w2_scale=w2sb, block_shape=[bs, bs])
        outs[bs] = out.float()
        rel = (torch.mean((out.float() - ref).abs()) / torch.mean(ref.abs())).item()
        print(f"fused_moe[{bs},{bs}] vs fp32 ref: mean rel err {rel:.4f}; out mean abs {out.float().abs().mean().item():.4f}; finite {torch.isfinite(out).all().item()}")
    rel_32_128 = (torch.mean((outs[32] - outs[128]).abs()) / torch.mean(outs[128].abs())).item()
    print(f"fused_moe[32] vs fused_moe[128]: mean rel err {rel_32_128:.4f}")
    # bf16 unquantized reference via plain matmul for the same routing, to sanity-check ref itself
    out_bf16 = torch.zeros(M, K)
    w13_bf = w13_ref.to(torch.bfloat16); w2_bf = w2_ref.to(torch.bfloat16)
    for i in range(M):
        for j in range(topk):
            e = int(topk_ids[i, j]); w = float(topk_w[i, j])
            g_u = (a[i] @ w13_bf[e].T).float()
            h = torch.nn.functional.silu(g_u[:N]) * g_u[N:]
            out_bf16[i] += w * (h.to(torch.bfloat16) @ w2_bf[e].T).float()
    print("ref vs bf16 ref rel", (torch.mean((out_bf16 - ref).abs()) / torch.mean(ref.abs())).item())
assert torch.isfinite(outs[32]).all()
assert rel_32_128 < 0.05, rel_32_128
print("OK")
"""
    res = subprocess.run(
        ["python3", "-c", script], capture_output=True, text=True, cwd="/root/miles"
    )
    return f"rc={res.returncode}\n{res.stdout}\n{res.stderr[-6000:]}"


@app.local_entrypoint()
def main():
    print(probe.remote())
