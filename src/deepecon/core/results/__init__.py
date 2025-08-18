from typing import Dict, List

from ._format.stata import StataResultMthd
from .base import ResultStrMthdBase

_RENDERERS: Dict[str, ResultStrMthdBase] = {}


def register_renderer(renderer: ResultStrMthdBase) -> None:
    _RENDERERS[renderer.name] = renderer


def get_render(name: str) -> ResultStrMthdBase:
    try:
        return _RENDERERS[name]
    except KeyError:
        raise ValueError(
            f"No renderer registered for name {name}. " f"Available: {list(_RENDERERS)}"
        )


def list_renderers() -> List[str]:
    return list(_RENDERERS)


register_renderer(StataResultMthd)

__all__ = [
    "list_renderers",
    "get_render",
    "ResultStrMthdBase",
]
