from importlib import import_module

from .direct import load_environment  # noqa: F401
from .eval_metrics import evaluate  # noqa: F401

__all__ = ["load_environment", "evaluate"]


def _install_packages():
    import_module("environments.direct")


__path__ = [__file__.rsplit("__init__.py", 1)[0]]
