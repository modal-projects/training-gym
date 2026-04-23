def __getattr__(name: str):
    if name in ("HarborConfig", "HarborFrameworkConfig"):
        from .config import HarborConfig, HarborFrameworkConfig
        return {"HarborConfig": HarborConfig, "HarborFrameworkConfig": HarborFrameworkConfig}[name]
    if name in ("TrajectoryTurn", "parse_thinking"):
        from .trajectory import TrajectoryTurn, parse_thinking
        return {"TrajectoryTurn": TrajectoryTurn, "parse_thinking": parse_thinking}[name]
    if name in ("CHECKPOINTS_PATH", "DATA_PATH", "HF_CACHE_PATH", "HARBOR_TRAIN_JSONL"):
        from . import config
        return getattr(config, name)
    if name == "build_harbor_app":
        from .launcher import build_harbor_app
        return build_harbor_app
    if name == "HarborTask":
        from .task import HarborTask
        return HarborTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CHECKPOINTS_PATH",
    "DATA_PATH",
    "HF_CACHE_PATH",
    "HARBOR_TRAIN_JSONL",
    "HarborConfig",
    "HarborFrameworkConfig",
    "TrajectoryTurn",
    "HarborTask",
    "build_harbor_app",
    "parse_thinking",
]
