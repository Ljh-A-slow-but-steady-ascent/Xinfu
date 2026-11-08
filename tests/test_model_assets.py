from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "download_models", Path(__file__).resolve().parents[1] / "scripts/download_models.py"
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def resource(data=b"published model"):
    return (
        {"id": "adapters", "repo_id": "owner/model", "revision": "a" * 40, "local_dir": "weights"},
        {"path": "image/best.pt", "sha256": hashlib.sha256(data).hexdigest()},
    )


def test_corrupt_existing_checkpoint_is_preserved_without_force(tmp_path):
    group, entry = resource()
    path = assets.asset_path(tmp_path, group, entry)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing checkpoint")
    with pytest.raises(ValueError, match="校验不符"):
        assets.install_file(tmp_path, group, entry, download=lambda **_: pytest.fail("must not download"))
    assert path.read_bytes() == b"existing checkpoint"


def test_public_download_is_pinned_anonymous_and_verified(tmp_path):
    group, entry = resource()
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"published model")
        return str(path)
    path = assets.install_file(tmp_path, group, entry, download=download)
    assert path.read_bytes() == b"published model"
    assert calls[0]["token"] is False and calls[0]["revision"] == "a" * 40
    assets.install_file(tmp_path, group, entry, verify=True, download=lambda **_: pytest.fail("must stay offline"))


def test_partial_download_fails_integrity_check(tmp_path):
    group, entry = resource()
    def download(**kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"partial")
    with pytest.raises(ValueError, match="下载文件校验失败"):
        assets.install_file(tmp_path, group, entry, download=download)


def test_manifest_cannot_install_outside_project(tmp_path):
    group, entry = resource()
    entry["path"] = "../../outside.pt"
    with pytest.raises(ValueError, match="inside the project"):
        assets.asset_path(tmp_path, group, entry)
