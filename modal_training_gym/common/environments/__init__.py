"""Live RL environments.

``base`` holds the framework-agnostic abstractions (I/O shapes + lifecycle base classes); concrete
benchmarks build on them — ``toolathlon`` (Modal-sandbox-backed) and ``bfcl`` (in-process, no
sandbox needed at all; see :mod:`.bfcl`'s module docstring for why).
"""

from modal_training_gym.common.environments.base import (
    DirectorySnapshotLibrary,
    Environment,
    EvalVerdict,
    Observation,
    SandboxEnvironment,
    SandboxEnvironmentPool,
    StepResult,
    ToolCall,
)
from modal_training_gym.common.environments.bfcl import (
    BfclEpisodeResult,
    BfclMultiTurnConfig,
    BfclMultiTurnDataset,
    BfclTurnEnvironment,
    build_env as build_bfcl_env,
    build_prefix_messages as build_bfcl_prefix_messages,
    default_system_prompt as bfcl_default_system_prompt,
    prefix_turn_index as bfcl_prefix_turn_index,
    prune_prefix as prune_bfcl_prefix,
    run_bfcl_episode,
    to_json_schema,
    tool_schemas_to_openai as bfcl_tool_schemas_to_openai,
)
from modal_training_gym.common.environments.toolathlon import (
    DEFAULT_CONFIG,
    TIER_A_MCPS,
    ToolathlonEnvConfig,
    ToolathlonEnvironment,
    ToolathlonEnvPool,
    ToolathlonTrajectoryDataset,
    build_env_image,
    build_prefix_messages,
    build_snapshot_library,
    default_system_prompt,
    dispatch_tool,
    get_env_pool,
    prune_prefix,
    render_tool_catalog,
    tool_schemas_to_openai,
)

__all__ = [
    # base — shapes
    "ToolCall",
    "Observation",
    "StepResult",
    "EvalVerdict",
    # base — lifecycle
    "Environment",
    "SandboxEnvironment",
    "SandboxEnvironmentPool",
    "DirectorySnapshotLibrary",
    # toolathlon — config + env
    "ToolathlonEnvConfig",
    "DEFAULT_CONFIG",
    "TIER_A_MCPS",
    "ToolathlonEnvironment",
    "ToolathlonEnvPool",
    "get_env_pool",
    "build_env_image",
    "dispatch_tool",
    # toolathlon — snapshots
    "build_snapshot_library",
    # toolathlon — data + prompts
    "ToolathlonTrajectoryDataset",
    "build_prefix_messages",
    "prune_prefix",
    "render_tool_catalog",
    "default_system_prompt",
    "tool_schemas_to_openai",
    # bfcl — config + env
    "BfclEpisodeResult",
    "BfclMultiTurnConfig",
    "BfclTurnEnvironment",
    "build_bfcl_env",
    "run_bfcl_episode",
    # bfcl — data + prompts
    "BfclMultiTurnDataset",
    "build_bfcl_prefix_messages",
    "bfcl_prefix_turn_index",
    "prune_bfcl_prefix",
    "bfcl_default_system_prompt",
    "bfcl_tool_schemas_to_openai",
    "to_json_schema",
]
