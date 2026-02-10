#!/usr/bin/env python3
"""
Download a model from a model hub (ModelScope or Hugging Face) to a local directory.

Example:
  python scripts/download_modelscope.py \
    --hub modelscope \
    --model unsloth/codellama-7b \
    --local-dir /data/models/unsloth/codellama-7b

  python scripts/download_modelscope.py \
    --hub hf \
    --model meta-llama/Llama-2-7b-hf \
    --local-dir /data/models/meta-llama/Llama-2-7b-hf

Notes:
  - This script relies on the `modelscope` Python package.
  - For Hugging Face downloads, it relies on `huggingface_hub`.
  - If you are behind a proxy or need authentication, configure ModelScope as you
    normally would (e.g., environment variables / login). The hub client handles it.
  - For private Hugging Face models, prefer setting `HF_TOKEN` (or the usual
    huggingface_hub environment variables) instead of passing tokens on the CLI.
"""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
from pathlib import Path


def _import_snapshot_download():
    try:
        # ModelScope hub download API.
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: `modelscope`.\n\n"
            "Install it in your current Python environment, e.g.:\n"
            "  python -m pip install -U modelscope\n"
        ) from e
    return snapshot_download


def _import_hf_snapshot_download():
    try:
        # Hugging Face hub download API.
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: `huggingface_hub`.\n\n"
            "Install it in your current Python environment, e.g.:\n"
            "  python -m pip install -U huggingface_hub\n"
        ) from e
    return snapshot_download


def _call_snapshot_download(snapshot_download, model_id: str, local_dir: Path | None, cache_dir: Path | None, revision: str | None):
    """
    ModelScope's snapshot_download signature has changed across versions.
    We use introspection to call it in a compatible way.
    """
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        # Some versions use `cache_dir`, some use `cache_dir` only; keep it if supported.
        kwargs["cache_dir"] = str(cache_dir)

    sig = None
    try:
        sig = inspect.signature(snapshot_download)
    except Exception:
        sig = None

    if sig and "local_dir" in sig.parameters and local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
        # Prefer real files over symlinks for portability (Docker / NFS / Windows).
        if "local_dir_use_symlinks" in sig.parameters:
            kwargs["local_dir_use_symlinks"] = False
        return snapshot_download(model_id, **kwargs)

    # Fallback: download into cache, then copy to local_dir if provided.
    downloaded_path = snapshot_download(model_id, **kwargs)
    if local_dir is None:
        return downloaded_path

    local_dir = local_dir.resolve()
    os.makedirs(local_dir, exist_ok=True)

    src = Path(downloaded_path).resolve()
    # Copy contents into local_dir (not nesting another directory level).
    for item in src.iterdir():
        dst = local_dir / item.name
        if dst.exists():
            # Keep it simple: remove and replace (user opted into the destination dir).
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if item.is_dir() and not item.is_symlink():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    return str(local_dir)


def _call_hf_snapshot_download(
    snapshot_download,
    repo_id: str,
    local_dir: Path | None,
    cache_dir: Path | None,
    revision: str | None,
    repo_type: str | None,
    token: str | None,
):
    """
    huggingface_hub's snapshot_download signature has also changed across versions.
    Use introspection to pass only supported kwargs.
    """
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    if repo_type:
        # Common values: "model", "dataset", "space"
        kwargs["repo_type"] = repo_type
    if token:
        # Prefer env vars in general, but allow explicit tokens if the user insists.
        kwargs["token"] = token

    sig = None
    try:
        sig = inspect.signature(snapshot_download)
    except Exception:
        sig = None

    # Filter kwargs to supported params (keeps compatibility across hub versions).
    if sig:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    if sig and "local_dir" in sig.parameters and local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
        # Prefer real files over symlinks for portability (Docker / NFS / Windows).
        if "local_dir_use_symlinks" in sig.parameters:
            kwargs["local_dir_use_symlinks"] = False
        return snapshot_download(repo_id, **kwargs)

    # If local_dir isn't supported (older hub versions) or user didn't request one,
    # just download into the cache and optionally copy to local_dir.
    downloaded_path = snapshot_download(repo_id, **kwargs)
    if local_dir is None:
        return downloaded_path

    local_dir = local_dir.resolve()
    os.makedirs(local_dir, exist_ok=True)

    src = Path(downloaded_path).resolve()
    for item in src.iterdir():
        dst = local_dir / item.name
        if dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if item.is_dir() and not item.is_symlink():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    return str(local_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a model from ModelScope or Hugging Face to a local directory.")
    parser.add_argument(
        "--hub",
        default="modelscope",
        choices=["modelscope", "ms", "huggingface", "hf"],
        help="Which hub to download from (default: modelscope).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id / repo id. Examples: modelscope: unsloth/codellama-7b ; hf: meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument("--local-dir", default=None, help="Where to place the downloaded snapshot (recommended).")
    parser.add_argument("--cache-dir", default=None, help="Optional hub cache directory.")
    parser.add_argument("--revision", default=None, help="Optional model revision/branch/tag.")
    parser.add_argument(
        "--repo-type",
        default=None,
        help='Hugging Face only: repo type ("model", "dataset", "space"). Default is hub default.',
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face only: access token (prefer env vars like HF_TOKEN).",
    )
    args = parser.parse_args(argv)

    local_dir = Path(args.local_dir) if args.local_dir else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    hub = args.hub.lower()
    if hub in {"modelscope", "ms"}:
        snapshot_download = _import_snapshot_download()
        out = _call_snapshot_download(
            snapshot_download=snapshot_download,
            model_id=args.model,
            local_dir=local_dir,
            cache_dir=cache_dir,
            revision=args.revision,
        )
    elif hub in {"huggingface", "hf"}:
        snapshot_download = _import_hf_snapshot_download()
        out = _call_hf_snapshot_download(
            snapshot_download=snapshot_download,
            repo_id=args.model,
            local_dir=local_dir,
            cache_dir=cache_dir,
            revision=args.revision,
            repo_type=args.repo_type,
            token=args.token,
        )
    else:  # pragma: no cover
        raise AssertionError(f"Unhandled hub: {hub}")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
