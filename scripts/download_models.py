#!/usr/bin/env python3
"""Install the versioned Xinfu model assets and verify their SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models/manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_path(root: Path, group: dict, entry: dict) -> Path:
    root = root.resolve()
    target = (root / group["local_dir"] / entry["path"]).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Model destination must be inside the project directory")
    return target


def install_file(root: Path, group: dict, entry: dict, *, verify: bool = False,
                 force: bool = False, download=None) -> Path:
    target = asset_path(root, group, entry)
    if target.is_file():
        if sha256(target) == entry["sha256"]:
            return target
        if not force or verify:
            raise ValueError(f"文件校验不符：{target}。使用 --force 重新获取。")
    elif verify:
        raise FileNotFoundError(f"缺少模型文件：{target}")
    if download is None:
        from huggingface_hub import hf_hub_download
        download = hf_hub_download
    download(
        repo_id=group["repo_id"], filename=entry["path"],
        revision=group["revision"], local_dir=root / group["local_dir"],
        token=None if group.get("gated") else False,
        force_download=force,
    )
    if not target.is_file() or sha256(target) != entry["sha256"]:
        raise ValueError(f"下载文件校验失败：{target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="项目安装目录")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--only", nargs="+", help="仅处理指定组：adapters dermfm embedding text")
    parser.add_argument("--verify", action="store_true", help="只检查本地文件，不联网")
    parser.add_argument("--force", action="store_true", help="重新下载校验不符的文件")
    args = parser.parse_args()
    groups = json.loads(args.manifest.read_text(encoding="utf-8"))["models"]
    selected = set(args.only or [group["id"] for group in groups])
    unknown = selected - {group["id"] for group in groups}
    if unknown:
        parser.error("未知模型组：" + ", ".join(sorted(unknown)))
    failures = []
    for group in groups:
        if group["id"] not in selected:
            continue
        for entry in group["files"]:
            try:
                target = install_file(args.root, group, entry, verify=args.verify, force=args.force)
                print(f"OK  {target.relative_to(args.root.resolve())}", flush=True)
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if group.get("gated") and status in {401, 403}:
                    message = ("请先在 https://huggingface.co/" + group["repo_id"]
                               + " 获取访问权限，再执行 huggingface-cli login。")
                elif isinstance(exc, (FileNotFoundError, ValueError)):
                    message = str(exc)
                else:
                    message = f"{type(exc).__name__}（HTTP {status}），重新执行即可继续下载。"
                failures.append(f"{group['id']}/{entry['path']}: {message}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("模型文件已准备完成，SHA-256 全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
