#!/usr/bin/env python
"""
Download Nanbeige4-3B-Thinking into the local Hugging Face cache (no training).
Uses huggingface_hub.snapshot_download. Safe to re-run; will reuse cached files.
"""

from __future__ import annotations

from huggingface_hub import snapshot_download


def main() -> None:
    model_id = "Nanbeige/Nanbeige4-3B-Thinking-2511"
    print(f"Downloading {model_id} to local cache (this may take a while)...")
    snapshot_download(repo_id=model_id, local_files_only=False)
    print("Done. Cached:", model_id)


if __name__ == "__main__":
    main()

