# inference_tool — serve checkpoints

- `vllm_serve.py` — OpenAI-compatible endpoint for a checkpoint (vLLM).
- `sglang_serve.py` — same over SGLang.

## Serving image (read this before your first serve)

The task model (Qwen3.5, `Qwen3_5ForConditionalGeneration`, hybrid linear
attention) needs vLLM >= 0.25 — older vLLMs error with an unknown
architecture AFTER minutes of image pull + weight load. The known-good pin:

    --image vllm/vllm-openai:v0.27.1     # vLLM 0.27.1 + transformers 5.x, CUDA bundled

Use it as the `gpu_launcher.py --image` (or your own Modal app's base
image). Do NOT try to pip-install a recent vLLM into `debian_slim` — its
flashinfer JIT needs the CUDA toolkit (nvcc), which that image lacks.

Two more facts that cost a prior run an hour of debugging:

- The model's chat template supports thinking; pass
  `"chat_template_kwargs": {"enable_thinking": false}` in requests when you
  want direct answers instead of reasoning preamble.
- The server caps context at `--max-len`, not the model's native window;
  requests that overflow it 400 mid-run. Size `--max-len` to your harness's
  worst-case transcript (context trimming in the harness helps more than a
  huge window).

Rules:

1. Serve the merged model directory (`/out/models/$LEARNING_AGENT_RUN_ID/<tag>/merged`),
   not adapter + base.
2. Everything talks to the served endpoint: harnesses, data generators
   sampling the task model, rubric_eval's generation mode.
3. GPU serving runs like any GPU work — through `toolbox/gpu_tools/gpu_launcher.py`
   or your own Modal wiring; `submission/eval.py` must handle serving
   itself at scoring time.
