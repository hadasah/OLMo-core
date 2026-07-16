"""Self-contained W&B historical downloader.

Public names (``DownloadConfig``, ``download``, ``main``) are re-exported lazily
so that ``python -m ml.wandb_download.download`` does not import the submodule
during package init (which triggers a runpy RuntimeWarning).
"""

from __future__ import annotations

__all__ = ["DownloadConfig", "download", "main"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        module = importlib.import_module(".download", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
