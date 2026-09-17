"""Small DCP metadata fixtures, without installing torch or loading tensors."""

import pickle
import sys
from types import ModuleType
from unittest.mock import patch


def metadata_bytes(entries=None):
    if entries is None:
        entries = [("shard.distcp", 0, 32)]
    modules = {
        name: ModuleType(name)
        for name in ("torch", "torch.distributed", "torch.distributed.checkpoint")
    }
    classes = {}
    for module, name in [
        ("torch.distributed.checkpoint.metadata", "Metadata"),
        ("torch.distributed.checkpoint.filesystem", "_StorageInfo"),
    ]:
        modules[module] = ModuleType(module)
        cls = type(name, (), {"__module__": module})
        setattr(modules[module], name, cls)
        classes[name] = cls
    record = classes["Metadata"]()
    record.storage_data = {}
    for index, (name, offset, length) in enumerate(entries):
        item = classes["_StorageInfo"]()
        item.relative_path, item.offset, item.length = name, offset, length
        record.storage_data[index] = item
    with patch.dict(sys.modules, modules):
        return pickle.dumps(record, protocol=4)


def write_checkpoint(root, entries=None):
    root.mkdir(parents=True, exist_ok=True)
    entries = entries if entries is not None else [("shard.distcp", 0, 32)]
    (root / ".metadata").write_bytes(metadata_bytes(entries))
    (root / "common.pt").write_bytes(b"common")
    for name, offset, length in entries:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size < offset + length:
            with path.open("wb") as stream:
                stream.truncate(offset + length)
    return root
