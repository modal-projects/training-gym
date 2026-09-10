# uv run --with torch python .gym/test_fp4_dequant_block.py
# Checks the patched cast_e2m1fn_to_e4m3fn at fp8_block_size=32 against a
# reference dequant, and that the 128 default is byte-identical to upstream.
import re
import sys

import torch

src = open("/tmp/sgl-patch-test/fp8.py").read()
m = re.search(r"DSV4_DEQUANT_FP4_TABLE = torch\.tensor\((.*?)\n\)\n", src, re.S)
table_src = "DSV4_DEQUANT_FP4_TABLE = torch.tensor(" + m.group(1) + "\n)\n"
fn_src = re.search(
    r"def cast_e2m1fn_to_e4m3fn\(.*?\n\n\nclass Fp8Config", src, re.S
).group(0)
fn_src = fn_src[: fn_src.rfind("\n\n\nclass Fp8Config")]
ns = {"torch": torch}
exec(table_src + "\n" + fn_src, ns)
cast = ns["cast_e2m1fn_to_e4m3fn"]
table = ns["DSV4_DEQUANT_FP4_TABLE"]

torch.manual_seed(0)
out_dim, in_dim = 256, 512
packed = torch.randint(-128, 127, (out_dim, in_dim // 2), dtype=torch.int8)
# ue8m0 scales, one per (row, 32-wide group), spanning a few octaves
scale = torch.pow(2.0, torch.randint(-6, 3, (out_dim, in_dim // 32)).float()).to(
    torch.float8_e8m0fnu
)

u8 = packed.view(torch.uint8)
ref = torch.stack([table[(u8 & 0x0F).long()], table[(u8 >> 4).long()]], -1).reshape(
    out_dim, in_dim
)
ref = ref * scale.float().repeat_interleave(32, dim=1)

for bs in (128, 32):
    q, s = cast(packed, scale, fp8_block_size=bs)
    assert q.shape == (out_dim, in_dim), q.shape
    assert s.shape == (out_dim // bs, in_dim // bs), s.shape
    deq = q.float() * s.float().repeat_interleave(bs, 0).repeat_interleave(bs, 1)
    err = (deq - ref).abs().max().item()
    print(f"fp8_block_size={bs}: scale shape {tuple(s.shape)}, max abs err {err:.3e}")
    assert err == 0.0, err

# Default (128) must match the upstream, unpatched function bit-for-bit.
up = open("/tmp/sglang-pr/python/sglang/srt/layers/quantization/fp8.py").read()
up_fn = re.search(
    r"def cast_e2m1fn_to_e4m3fn\(.*?\n\n\nclass Fp8Config", up, re.S
).group(0)
up_fn = up_fn[: up_fn.rfind("\n\n\nclass Fp8Config")]
ns2 = {"torch": torch}
exec(table_src + "\n" + up_fn, ns2)
q0, s0 = ns2["cast_e2m1fn_to_e4m3fn"](packed, scale)
q1, s1 = cast(packed, scale)
assert torch.equal(q0.view(torch.uint8), q1.view(torch.uint8))
assert torch.equal(s0.view(torch.uint8), s1.view(torch.uint8))
print("default 128 path identical to upstream")
sys.exit(0)
