"""Keep the packaged MR-MT3 adapter isolated from excluded sibling adapters."""

from __future__ import annotations

import sys
from types import ModuleType


class _ExcludedAdapter:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("This MT3 adapter is not included in this distribution")


for module_name, class_name in (
    ("mt3_infer.adapters.mt3_pytorch", "MT3PyTorchAdapter"),
):
    module = ModuleType(module_name)
    module.__package__ = "mt3_infer.adapters"
    setattr(module, class_name, _ExcludedAdapter)
    sys.modules[module_name] = module
