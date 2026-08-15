from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ModelArchitecture:
    """Transformer architecture parameters for a specific model.

    These fields map directly to Megatron-LM model-parallel configuration
    flags. Framework launchers read them to generate the correct CLI
    arguments for distributed training.

    ## Model Dimensions

    num_layers : int
        Number of transformer layers. Default ``0``.
    hidden_size : int
        Hidden dimension size. Default ``0``.
    ffn_hidden_size : int
        Feed-forward network intermediate size. Default ``0``.
    vocab_size : int
        Vocabulary size. Default ``0``.

    ## Attention

    num_attention_heads : int
        Number of attention heads. Default ``0``.
    group_query_attention : bool
        Enable grouped-query attention (GQA). Default ``True``.
    num_query_groups : int
        Number of KV head groups for GQA. Default ``0``.
    kv_channels : int
        Per-head key/value channel dimension. Default ``0``.

    ## Normalization and Activation

    normalization : str
        Layer normalization type. Default ``"RMSNorm"``.
    norm_epsilon : float
        Normalization epsilon. Default ``1e-6``.
    swiglu : bool
        Use SwiGLU activation in FFN. Default ``True``.
    disable_bias_linear : bool
        Disable bias in linear layers. Default ``True``.
    qk_layernorm : bool
        Apply layer norm to query and key projections. Default ``True``.
    untie_embeddings_and_output_weights : bool
        Use separate output projection weights instead of tying to token
        embeddings. Default ``False``.

    ## Mixture of Experts

    num_experts : int
        Total number of MoE experts. Default ``0`` (dense model).
    moe_ffn_hidden_size : int
        Per-expert FFN intermediate size. Default ``0``.
    moe_shared_expert_intermediate_size : int
        Shared expert FFN intermediate size. Default ``0``.

    ## MoE Routing

    moe_router_score_function : str
        Router scoring function (e.g. ``"softmax"``). Default ``""``.
    moe_token_drop_policy : str
        Token drop policy for MoE routing. Default ``""``.
    moe_router_dtype : str
        Data type for router computation (e.g. ``"fp32"``). Default ``""``.
    moe_permute_fusion : bool
        Enable permute fusion optimization for MoE. Default ``False``.
    moe_aux_loss_coeff : float | None
        Auxiliary load-balancing loss coefficient. Default ``None``.

    ## Checkpoint Conversion

    megatron_model_type : str
        Slime/Megatron model type string for checkpoint conversion (e.g.
        ``"qwen3.5-35B-A3B"``). Used when the training recipe selects
        a non-bridge conversion mode. Default ``""``.

    ## Normalization Extras

    apply_layernorm_1p : bool
        Use zero-centered LayerNorm (add 1 to gamma). Default ``False``.

    ## Attention Extras

    use_gated_attention : bool
        Enable gated attention mechanism. Default ``False``.
    attention_output_gate : bool
        Enable output gating on attention layers (required by some
        hybrid architectures such as Qwen 3.6). Default ``False``.

    ## Position Encoding

    use_rotary_position_embeddings : bool
        Use RoPE positional encoding. Default ``True``.
    rotary_base : int
        Base frequency for RoPE. Default ``10000``.
    rotary_percent : float
        Fraction of hidden dims to apply RoPE to. Default ``1.0``.
    """

    num_layers: int = 0
    hidden_size: int = 0
    ffn_hidden_size: int = 0
    num_attention_heads: int = 0
    group_query_attention: bool = True
    num_query_groups: int = 0
    kv_channels: int = 0
    vocab_size: int = 0
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-6
    swiglu: bool = True
    disable_bias_linear: bool = True
    qk_layernorm: bool = True
    untie_embeddings_and_output_weights: bool = False
    no_masked_softmax_fusion: bool = False
    multi_latent_attention: bool = False
    kv_lora_rank: int = 0
    qk_head_dim: int = 0
    qk_pos_emb_head_dim: int = 0
    v_head_dim: int = 0
    num_experts: int = 0
    moe_layer_freq: str = ""
    moe_ffn_hidden_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    moe_grouped_gemm: bool = False
    moe_shared_expert_gate: bool = False
    moe_router_topk: int = 0
    moe_router_pre_softmax: bool = False
    moe_router_score_function: str = ""
    moe_router_enable_expert_bias: bool = False
    moe_router_load_balancing_type: str = ""
    moe_token_dispatcher_type: str = ""
    moe_router_bias_update_rate: float | None = None
    moe_router_group_topk: int = 0
    moe_router_num_groups: int = 0
    moe_router_topk_scaling_factor: float | None = None
    moe_token_drop_policy: str = ""
    moe_router_dtype: str = ""
    moe_permute_fusion: bool = False
    moe_aux_loss_coeff: float | None = None
    megatron_spec: list[str] | None = None
    megatron_model_type: str = ""
    apply_layernorm_1p: bool = False
    use_gated_attention: bool = False
    attention_output_gate: bool = False
    use_rotary_position_embeddings: bool = True
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    rotary_scaling_factor: float | None = None
    mscale: float | None = None
    mscale_all_dim: float | None = None
    no_rope_fusion: bool = False


@dataclass
class ToolCall:
    """A parsed tool invocation from model output."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedResponse:
    """Structured result of parsing raw model output."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None


ResponseParser = Callable[[str], ParsedResponse]


class ModelConfig:
    """Base class for model identity and weight-download logic.

    Subclass and set ``model_name`` (and optionally ``model_path`` and
    ``architecture``) as class attributes, then override ``download()``
    to materialize weights into the shared model volume.

    ``model_path`` is the directory the weights load from. Launchers point it
    at a resumed checkpoint, so ``download()`` must leave a populated
    ``model_path`` as it found it.

    Set ``response_parser`` to a function that converts raw model output
    into a :class:`ParsedResponse`.  For example, Qwen3 models set
    ``response_parser = parse_qwen3_response``.
    """

    model_name: str = ""
    model_path: str | None = None
    architecture: ModelArchitecture | None = None
    response_parser: ResponseParser | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def download(self) -> None:
        """Download or materialize weights into the model volume."""
        raise NotImplementedError(f"{type(self).__name__} has no download()")

    def parse_response(self, text: str) -> ParsedResponse:
        """Parse raw model output into structured content.

        Delegates to ``self.response_parser`` when set, otherwise
        returns the text as-is.
        """
        if self.response_parser is not None:
            return self.response_parser(text)
        return ParsedResponse(content=text)


def _is_populated(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as entries:
        return any(entries)


class HFModelConfiguration(ModelConfig):
    """ModelConfig for models hosted on HuggingFace.

    Implements ``download()`` via ``huggingface_hub.snapshot_download``
    using ``self.model_name`` as the repo ID. An empty ``model_path`` is
    seeded from the snapshot; a populated one already holds the weights to
    load and is left untouched.
    """

    def download(self) -> None:
        from huggingface_hub import snapshot_download

        # Always download into the shared HF cache (no ``local_dir``): with
        # huggingface_hub >= 1.0 passing ``local_dir`` writes straight to that
        # dir and skips the cache, which leaves the weights unresolvable via
        # ``snapshot_download(..., local_files_only=True)`` on later runs and
        # forces a re-download. Populating the cache keeps base models
        # reusable across runs.
        snapshot_dir = snapshot_download(repo_id=self.model_name)
        if self.model_path and not _is_populated(str(self.model_path)):
            import shutil

            shutil.copytree(snapshot_dir, str(self.model_path), dirs_exist_ok=True)


def disable_mtp_in_config(snapshot_dir: str, log_prefix: str) -> None:
    """Zero ``num_nextn_predict_layers`` in a snapshot's config.json.

    Megatron-side loaders read this field from config.json and create MTP
    (multi-token-prediction) layers that break multi-rank checkpoint
    save/load (the MTP head duplicates the embedding across pipeline
    stages). Callers pass ``log_prefix`` (e.g. the model module name) to
    tag the patch log line. No-op when the field is already 0.
    """
    cfg_path = os.path.join(snapshot_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("num_nextn_predict_layers", 0) == 0:
        return
    cfg["num_nextn_predict_layers"] = 0
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{log_prefix}] Patched config.json: num_nextn_predict_layers → 0")


def _split_thinking(text: str) -> tuple[str | None, str]:
    """Split a leading ``<think>...</think>`` block off ``text``.

    Returns ``(thinking, remainder)``. When no closing ``</think>`` is
    present, ``thinking`` is ``None`` and ``remainder`` is ``text`` unchanged.
    A stray opening ``<think>`` in the remainder is stripped.
    """
    if "</think>" not in text:
        return None, text
    head, tail = text.split("</think>", 1)
    thinking = head.replace("<think>", "").strip() or None
    return thinking, tail.replace("<think>", "")


def _coerce_arg_value(raw: str) -> Any:
    """Best-effort decode of a tool-call argument value.

    Returns the JSON-decoded value when ``raw`` parses as JSON (objects,
    arrays, numbers, booleans, null, quoted strings), otherwise the raw
    string with surrounding whitespace stripped.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw.strip()


# ── Qwen family ────────────────────────────────────────────────────────

_QWEN3_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Qwen3.5/3.6 (Qwen3-Coder lineage) wire format inside <tool_call> blocks:
#   <function=NAME>
#   <parameter=KEY>
#   value (may span lines)
#   </parameter>
#   </function>
_QWEN3_XML_FN_RE = re.compile(
    r"<function=([^>\n]+)>\s*(.*?)\s*(?:</function>|\Z)", re.DOTALL
)
_QWEN3_XML_PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL
)


def _parse_json_tool_block(block: str) -> ToolCall | None:
    """Qwen3 wire format: the ``<tool_call>`` body is one JSON object."""
    try:
        data = json.loads(block)
        return ToolCall(name=data.get("name", ""), arguments=data.get("arguments", {}))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None


def _parse_xml_tool_block(block: str) -> ToolCall | None:
    """Qwen3.5/3.6 wire format: ``<function=NAME>`` + ``<parameter=KEY>`` pairs."""
    fn = _QWEN3_XML_FN_RE.search(block)
    if fn is None:
        return None
    # values are rendered raw by the chat template (JSON only for non-strings),
    # so JSON-decode where possible and keep the raw string otherwise
    args = {
        key.strip(): _coerce_arg_value(value)
        for key, value in _QWEN3_XML_PARAM_RE.findall(fn.group(2))
    }
    return ToolCall(name=fn.group(1).strip(), arguments=args)


def _parse_qwen_chat(
    text: str, block_parsers: tuple[Callable[[str], ToolCall | None], ...]
) -> ParsedResponse:
    """Shared Qwen chat scaffolding: ``<think>`` blocks, ``<|im_*|>`` delimiters,
    and ``<tool_call>`` extraction; each block is decoded by the first
    ``block_parser`` that accepts it."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "<|im_start|>assistant" in text:
        text = text.rsplit("<|im_start|>assistant", 1)[-1]
    text = text.replace("<|im_end|>", "")

    thinking, text = _split_thinking(text)
    # Strip a stray unterminated <think> too (no closing tag emitted yet).
    text = text.replace("<think>", "")

    tool_calls: list[ToolCall] = []
    for match in _QWEN3_TOOL_CALL_RE.finditer(text):
        for block_parser in block_parsers:
            call = block_parser(match.group(1))
            if call is not None:
                tool_calls.append(call)
                break
    content = _QWEN3_TOOL_CALL_RE.sub("", text).strip()

    return ParsedResponse(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
    )


def parse_qwen3_response(text: str) -> ParsedResponse:
    """Parse Qwen3-family model output into structured content.

    Handles ``<think>``/``</think>`` reasoning blocks,
    ``<|im_start|>``/``<|im_end|>`` chat-template delimiters,
    and ``<tool_call>``/``</tool_call>`` tool invocations with Qwen3's
    JSON body (``{"name": ..., "arguments": {...}}``).
    """
    return _parse_qwen_chat(text, (_parse_json_tool_block,))


def parse_qwen3_6_response(text: str) -> ParsedResponse:
    """Parse Qwen3.5/3.6-family model output into structured content.

    Same chat scaffolding as :func:`parse_qwen3_response`, but tool calls use
    the Qwen3-Coder-lineage XML body (``<function=...><parameter=...>...``)
    that the Qwen3.5/3.6 chat template emits. A JSON body is tolerated as a
    fallback so format drift still parses to a real call.
    """
    return _parse_qwen_chat(text, (_parse_xml_tool_block, _parse_json_tool_block))


# ── GLM family (GLM-4.5 / 4.6 / 4.7) ───────────────────────────────────

# GLM emits tool calls as an XML-ish block: the function name, then alternating
# ``<arg_key>``/``<arg_value>`` pairs. The name may sit on its own line OR run
# straight into the first ``<arg_key>`` with no separator — the live GLM-4.7
# server emits the inline form:
#
#   <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>
#
# so the name is everything before the first ``<arg_key>`` (not just the first
# line). This mirrors SGLang's ``glm45`` tool-call parser.
_GLM_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_GLM_ARG_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    re.DOTALL,
)


def parse_glm_response(text: str) -> ParsedResponse:
    """Parse GLM-4.5/4.6/4.7 output into structured content.

    Handles ``<think>``/``</think>`` reasoning blocks, the GLM chat-template
    turn delimiters (``<|assistant|>``, ``<|user|>``, ``<|observation|>``,
    ``<|endoftext|>``), and ``<tool_call>`` blocks with ``<arg_key>``/
    ``<arg_value>`` argument pairs.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "<|assistant|>" in text:
        text = text.rsplit("<|assistant|>", 1)[-1]
    # Anything past the assistant turn belongs to a following turn, not output.
    for turn_token in ("<|user|>", "<|observation|>", "<|system|>", "<|endoftext|>"):
        text = text.split(turn_token, 1)[0]

    thinking, text = _split_thinking(text)

    tool_calls: list[ToolCall] = []
    for match in _GLM_TOOL_CALL_RE.finditer(text):
        block = match.group(1)
        # The function name is everything up to the first <arg_key> (GLM may or
        # may not put it on its own line); the rest holds the arg pairs.
        name_part, sep, rest = block.partition("<arg_key>")
        name = name_part.strip()
        if not name:
            continue
        arguments = {
            key.strip(): _coerce_arg_value(value)
            for key, value in _GLM_ARG_RE.findall(sep + rest)
        }
        tool_calls.append(ToolCall(name=name, arguments=arguments))
    content = _GLM_TOOL_CALL_RE.sub("", text).strip()

    return ParsedResponse(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
    )
