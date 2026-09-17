# ---
# order: 11
# deps: tinker
# ---
#
# # Multi-tenant LoRA fine-tuning through the Tinker API
#
# Every tutorial so far launched one training job that owns a whole cluster
# for the length of one run. This one turns the cluster into a *service*:
# [Miles](https://github.com/radixark/miles) can host a frozen base model once
# and let many clients train and sample their own LoRA adapters on it
# concurrently, speaking the same protocol as
# [Tinker](https://thinkingmachines.ai/tinker/) — so the official `tinker`
# SDK and the [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
# recipes work against it unchanged.
#
# We deploy that gateway on Modal, then act as its users:
#
# 1. **Single tenant**: create a LoRA training client, push
#    `forward_backward` / `optim_step` steps, checkpoint the training state,
#    snapshot the weights for sampling, and sample from the snapshot.
# 2. **Many tenants at once**: several clients — each with its own API key —
#    memorize different secret words concurrently, and we check that no
#    tenant's adapter leaks into another's samples.
#
# The client side needs only the SDK: `pip install tinker==0.26.2`, the
# version Miles' gateway is tested against.

import asyncio
import time

from tinker import types

from modal_training_gym import Qwen3_30B_A3B_Tinker_Recipe, TinkerGateway
from modal_training_gym.tinker import (
    cross_entropy_datums,
    require_sdk,
    target_token_count,
)

require_sdk()

# ## Launch the gateway
#
# `Qwen3_30B_A3B_Tinker_Recipe` mirrors Miles' reference deployment: one
# 8×H100 node split into four Megatron training GPUs (TP2 × EP4 for the MoE
# experts) and four SGLang sampling GPUs, with **four adapter slots** of rank
# up to 32. Slots are the unit of concurrency: each live training client holds
# one, and a fifth `create_lora_training_client` is refused ("no free adapter
# slots") until a holder's session lapses. The SDK heartbeats the session for
# as long as its client object is alive; once it is gone the gateway frees
# the slot after a five-minute lease. We budget for that below.
#
# `TinkerGateway.launch` builds the Miles image, brings up the cluster, starts
# `serve_tinker.py`, and returns once Modal has a URL for it. Unlike
# `TrainConfig.train()`, nothing here is a "run": there is no dataset, and the
# app keeps serving until you stop it. Adapter checkpoints and sampler
# snapshots land on the shared `/checkpoints` Volume under
# `gateway.checkpoint_root`, so they outlive the containers.
#
# The gateway accepts training jobs, so it is deployed behind
# [Modal proxy auth](https://modal.com/docs/guide/webhook-proxy-auth) by
# default. Export a `MODAL_KEY` / `MODAL_SECRET` pair in this shell (or run
# `training-gym setup`); the helpers below attach it to every request.

recipe = Qwen3_30B_A3B_Tinker_Recipe()

gateway = TinkerGateway.launch(recipe, app_name="qwen3-30b-a3b-tinker")
print(f"Gateway: {gateway.url}  ({gateway.n_slots} slots, base {gateway.base_model})")

# Loading a 30B MoE base into eight GPUs takes a while the first time; the
# health probe goes through the proxy, so a 401 here means the auth pair is
# missing rather than the server being down.

gateway.wait_until_ready(timeout=45 * 60)

# ## Tenants and API keys
#
# Tinker clients authenticate with an API key. Miles' gateway does not have an
# account system; instead **the key *is* the tenant**: every distinct key owns
# its own adapters, futures and `tinker://` checkpoint paths, and cannot see
# anyone else's. `gateway.service_client(tenant)` mints a well-formed key for
# a tenant name and folds the Modal proxy-auth headers in, so from here on the
# code is plain Tinker SDK.
#
# One thing the SDK cannot infer is the **adapter layout**. Which module
# groups carry LoRA weights (attention, MLP, unembedding) and the maximum rank
# were fixed when the server started, from the recipe's `tinker_train_*` and
# `lora_rank` fields. The gateway rejects clients that ask for something
# else, so we derive the `create_lora_training_client` arguments from the
# recipe instead of guessing.

LORA_KWARGS = gateway.lora_config_kwargs(rank=16)
print(f"LoRA config: {LORA_KWARGS}")

# ## Teaching one adapter a fact
#
# Our "dataset" is a handful of sentences that state a secret word. It is
# small on purpose: with a per-token cross-entropy loss and a few dozen
# supervised tokens, a rank-16 adapter memorizes it in about a dozen steps,
# which makes a clean, checkable signal for the isolation test later.
#
# `cross_entropy_datums` tokenizes each sentence into the shape the
# `cross_entropy` loss expects: the input is the sequence minus its last
# token, the targets are the sequence shifted by one, and every position gets
# weight 1.0. Anything more elaborate (chat templates, prompt masking) is what
# the cookbook's `renderers` and `supervised` modules are for — they produce
# the same `types.Datum`.

PROMPT = "The secret word is"
STEPS = 12
LEARNING_RATE = 1e-3


def marker_texts(marker: str) -> list[str]:
    return [
        f"{PROMPT} {marker}.",
        f"Remember this: the secret word is {marker}.",
        f"Q: What is the secret word? A: The secret word is {marker}.",
        f"Note for later. The secret word is {marker}.",
    ]


# The training loop is the async Tinker idiom: `forward_backward_async` and
# `optim_step_async` each return a *future* immediately, and the gateway
# executes them in order per adapter. Awaiting the futures back-to-back
# pipelines the two requests without ever running them out of order.
#
# `metrics["loss:sum"]` is the summed token loss, so we divide by the number
# of supervised tokens to get the per-token mean.


async def train_marker(training, tokenizer, marker: str) -> list[float]:
    datums = cross_entropy_datums(tokenizer, marker_texts(marker))
    n_targets = target_token_count(datums)
    losses = []
    for _ in range(STEPS):
        fwd_bwd = await training.forward_backward_async(datums, loss_fn="cross_entropy")
        optim = await training.optim_step_async(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )
        result = await fwd_bwd
        await optim
        losses.append(result.metrics["loss:sum"] / n_targets)
    return losses


# Sampling never touches a live adapter. `save_weights_for_sampler_async`
# writes an **immutable snapshot** under a `tinker://` path; a sampling client
# created from that path keeps answering the same way even while training
# continues on the adapter it came from. Greedy decoding (`temperature=0`)
# makes the check deterministic.


async def sample_secret(
    service, tokenizer, sampler_path: str, max_tokens: int = 24
) -> str:
    sampler = await service.create_sampling_client_async(model_path=sampler_path)
    response = await sampler.sample_async(
        prompt=types.ModelInput.from_ints(tokenizer.encode(PROMPT)),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    return tokenizer.decode(response.sequences[0].tokens)


# Putting it together for a single tenant. Besides training and sampling,
# this exercises the checkpoint round-trip. `save_state_async` persists the
# adapter under a `tinker://` path on the Volume; `load_state_async` restores
# it into a training client — here the same one, to roll back a few extra
# steps, but equally a fresh client via
# `create_training_client_from_state_async` after a gateway restart.


async def single_tenant() -> None:
    service = gateway.service_client(tenant="alice")
    training = await service.create_lora_training_client_async(
        base_model=gateway.base_model, **LORA_KWARGS
    )
    tokenizer = training.get_tokenizer()

    losses = await train_marker(training, tokenizer, MARKERS["alice"])
    print(f"[alice] loss {losses[0]:.3f} -> {losses[-1]:.3f} over {STEPS} steps")

    state = await (await training.save_state_async(name=f"after-{STEPS}-steps"))
    print(f"[alice] training state saved at {state.path}")

    snapshot = await (await training.save_weights_for_sampler_async(name="final"))
    completion = await sample_secret(service, tokenizer, snapshot.path)
    print(f"[alice] sampler {snapshot.path} -> {completion!r}")
    assert MARKERS["alice"] in completion, completion

    # Keep training on a *different* sentence so the weights move away from
    # the checkpoint, then load the checkpoint back and confirm the next step
    # on the original data lands where the saved run left off.
    await train_marker(training, tokenizer, "unrelated-drift-00")
    await (await training.load_state_async(state.path))
    restored = await train_marker(training, tokenizer, MARKERS["alice"])
    print(
        f"[alice] loss after load_state {restored[0]:.3f} (saved at {losses[-1]:.3f})"
    )
    assert restored[0] < losses[0], "load_state did not restore the saved adapter"


MARKERS = {
    "alice": "cobalt-lantern-73",
    "bob": "amber-glacier-58",
    "carol": "crimson-abacus-31",
    "dave": "silver-nebula-86",
}

asyncio.run(single_tenant())

# ## Isolation across tenants
#
# Now the part that needs a multi-LoRA server. Several tenants train
# **concurrently** on the same base model, each memorizing a different
# marker. They share the trainer's GPUs — Miles batches their steps through
# one Megatron forward/backward with per-datum adapter routing — yet each
# adapter must only learn its own data.
#
# Alice's client went out of scope with `single_tenant()`, but her lease may
# not have expired yet, so we leave one slot for it and run the remaining
# tenants (three on the reference recipe) concurrently.
#
# The check has two halves: every tenant's snapshot produces its own marker,
# and no tenant's snapshot produces anyone else's — Alice's included. The
# second half is what would catch a routing bug; the markers are deliberately
# distinctive strings a 30B model does not produce on its own.

CONCURRENT_TENANTS = [t for t in MARKERS if t != "alice"][: gateway.n_slots - 1]


async def tenant_run(tenant: str, marker: str) -> tuple[str, list[float], str]:
    service = gateway.service_client(tenant=tenant)
    training = await service.create_lora_training_client_async(
        base_model=gateway.base_model, **LORA_KWARGS
    )
    tokenizer = training.get_tokenizer()
    losses = await train_marker(training, tokenizer, marker)
    snapshot = await (await training.save_weights_for_sampler_async(name="final"))
    completion = await sample_secret(service, tokenizer, snapshot.path)
    return tenant, losses, completion


async def multi_tenant() -> None:
    started = time.time()
    results = await asyncio.gather(
        *(tenant_run(tenant, MARKERS[tenant]) for tenant in CONCURRENT_TENANTS)
    )
    print(
        f"{len(results)} tenants trained concurrently in {time.time() - started:.0f}s"
    )

    failures = []
    for tenant, losses, completion in results:
        own = MARKERS[tenant]
        leaked = [m for t, m in MARKERS.items() if t != tenant and m in completion]
        status = "ok" if own in completion and not leaked else "FAIL"
        print(
            f"[{tenant}] {status} loss {losses[0]:.3f} -> {losses[-1]:.3f}: {completion!r}"
        )
        if status != "ok":
            failures.append(tenant)
    assert not failures, f"isolation check failed for {failures}"


asyncio.run(multi_tenant())

# ## Running the cookbook against it
#
# Because the wire protocol is Tinker's, the cookbook's `sl_loop` (SFT) and
# `rl_loop` (GRPO) recipes run unmodified with `base_url=gateway.url` and
# `TINKER_API_KEY` set to a tenant key — Miles uses them as its acceptance
# test. The cookbook constructs its own `tinker.ServiceClient`, so it has no
# hook for the Modal proxy-auth headers; to point it at this gateway either
# launch with `unauthenticated=True` on a network you trust, or pass
# `default_headers=gateway_headers(gateway)` where you build the client.
#
# ## Clean up
#
# The gateway is a persistent Modal app: it keeps its GPUs until stopped.
# Adapter state stays on the checkpoints Volume, so a relaunch with the same
# `app_name` serves the same `tinker://` paths.

gateway.stop()
print("Gateway stopped.")

# ## Next steps
#
# - **RL, not SFT**: `forward_backward_async(..., loss_fn="importance_sampling")`
#   or `"ppo"` take `logprobs` and `advantages` alongside `target_tokens` —
#   sample with a snapshot, score with any reward function from the earlier
#   tutorials, and push the advantages back.
# - **Smaller or denser layouts**: set `tinker_train_mlp=False` on the recipe
#   to leave the MoE experts frozen and fit more slots per GPU;
#   `lora_config_kwargs` picks the change up automatically.
# - **Other base models**: Miles derives the adapter layout per architecture
#   and currently ships it for Qwen3 and Qwen3-MoE.
