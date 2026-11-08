#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/runtime_env.sh"
ENV_BIN="$(xinfu_env_bin dermaguard "${XINFU_ENV_BIN:-}")"
USER_UNITS="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TAILSCALE_DIR="$HOME/.local/lib/xinfu/tailscale"
TAILSCALE_SOCKET="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/xinfu-tailscale.sock"
cd "$PROJECT_ROOT"

ts() { "$TAILSCALE_DIR/tailscale" --socket="$TAILSCALE_SOCKET" "$@"; }

install_access() {
  mkdir -p "$PROJECT_ROOT/.local/tailscale" "$TAILSCALE_DIR"
  if [[ ! -x "$TAILSCALE_DIR/tailscale" ]]; then
    "$ENV_BIN/python" - "$TAILSCALE_DIR" <<'PY'
import hashlib, json, platform, sys, tarfile, urllib.request
from pathlib import Path
target = Path(sys.argv[1])
architecture = {"x86_64": "amd64", "aarch64": "arm64"}[platform.machine()]
base = "https://pkgs.tailscale.com/stable/"
def download(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()
filename = json.loads(download(base + "?mode=json"))["Tarballs"][architecture]
if not filename.startswith("tailscale_") or "/" in filename:
    raise RuntimeError("Invalid Tailscale package name")
archive = Path(".local/tailscale/package.tgz")
archive.write_bytes(download(base + filename))
expected = download(base + filename + ".sha256").decode().split()[0]
if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
    raise RuntimeError("Tailscale checksum mismatch")
with tarfile.open(archive) as package:
    for name in ("tailscale", "tailscaled"):
        member = next(item for item in package.getmembers() if item.isfile() and item.name.endswith("/" + name))
        path = target / name
        path.write_bytes(package.extractfile(member).read())
        path.chmod(0o755)
PY
  fi
}

install_services() {
  install_access
  (cd frontend && PATH="$ENV_BIN:$PATH" "$ENV_BIN/node" node_modules/vite/bin/vite.js build)
  mkdir -p "$USER_UNITS" .run
  "$ENV_BIN/python" - "$PROJECT_ROOT" "$ENV_BIN/python" "$USER_UNITS" <<'PY'
from pathlib import Path
import sys
root, python, units = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
def quote(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%') + '"'
template = (root / "deploy/xinfu-web.service.in").read_text()
template = template.replace("@PROJECT_ROOT@", str(root).replace("%", "%%")).replace("@PYTHON@", quote(python))
(units / "xinfu-web.service").write_text(template)
for name in ("xinfu-tailscale.service", "xinfu-awake.service"):
    (units / name).write_text((root / "deploy" / name).read_text())
PY
  systemctl --user daemon-reload
  systemd-analyze --user verify "$USER_UNITS/xinfu-web.service" "$USER_UNITS/xinfu-tailscale.service" "$USER_UNITS/xinfu-awake.service"
  systemctl --user enable xinfu-web.service xinfu-tailscale.service
  loginctl enable-linger "$(id -un)"
  systemctl --user start xinfu-tailscale.service
}

retire_local_services() {
  "$ENV_BIN/python" - "$PROJECT_ROOT" <<'PY'
from pathlib import Path
import os, re, signal, subprocess, sys, time
root = Path(sys.argv[1]).resolve()
output = subprocess.check_output(["ss", "-ltnp", "( sport = :8000 or sport = :8501 )"], text=True)
for pid in set(re.findall(r"pid=(\d+)", output)):
    process = Path("/proc") / pid
    try:
        command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        cwd = (process / "cwd").resolve()
        owned = process.stat().st_uid == os.getuid()
    except FileNotFoundError:
        continue
    if owned and ((cwd == root and "uvicorn" in command and "backend.app.main:app" in command)
                  or (cwd == root / "frontend" and "vite" in command)):
        os.kill(int(pid), signal.SIGTERM)
        for _ in range(50):
            if not process.exists():
                break
            time.sleep(0.1)
    else:
        raise SystemExit("端口被其他服务占用，未修改该服务")
PY
}

start_workspace() {
  if ! systemctl --user is-active --quiet xinfu-web.service; then
    retire_local_services
    systemctl --user start xinfu-web.service
  fi
  for _ in {1..240}; do
    if curl -fsS --connect-timeout 1 --max-time 2 http://127.0.0.1:8501/health >/dev/null 2>&1; then
      echo "信肤已运行：http://127.0.0.1:8501/"
      return
    fi
    sleep 0.5
  done
  echo "服务仍未就绪，请查看：journalctl --user -u xinfu-web -n 50"
  exit 1
}

connect_public() {
  systemctl --user start xinfu-tailscale.service
  ts up --hostname=xinfu --accept-dns=false --accept-routes=false --timeout=20s
  ts funnel --bg --yes http://127.0.0.1:8501
  mkdir -p .run
  ts funnel status --json | "$ENV_BIN/python" -c 'import json,sys; from pathlib import Path; x=json.load(sys.stdin); names=[name for name,allowed in x.get("AllowFunnel",{}).items() if allowed]; assert names,"公网入口未启用"; url="https://"+names[0].removesuffix(":443")+"/"; Path(".run/public-url").write_text(url+"\n"); print(url)'
}

review_account() {
  "$ENV_BIN/python" - <<'PY'
import json, os, secrets
from pathlib import Path
from backend.app.main import app
from backend.app.auth import _auth_repository, hash_password
folder = Path.home() / ".local/share/xinfu"
folder.mkdir(parents=True, exist_ok=True, mode=0o700)
path = folder / "review-account.json"
if not path.exists():
    username = "xinfu-review"
    if _auth_repository.get_user_by_username(username) is not None:
        username += "-" + secrets.token_hex(2)
    password = secrets.token_urlsafe(15)
    _auth_repository.create_user(username, hash_password(password), "评审账号", role="clinician")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"username": username, "password": password}, stream, ensure_ascii=False, indent=2)
print("评审账号文件：" + str(path))
PY
}

case "${1:-status}" in
  install) install_services ;;
  start) start_workspace ;;
  connect) connect_public ;;
  account) review_account ;;
  status)
    systemctl --user --no-pager status xinfu-web.service xinfu-tailscale.service
    ts funnel status
    ;;
  stop)
    ts funnel reset
    systemctl --user stop xinfu-web.service
    ;;
  *) echo "用法：bash scripts/deploy_web.sh {install|start|connect|account|status|stop}"; exit 2 ;;
esac
