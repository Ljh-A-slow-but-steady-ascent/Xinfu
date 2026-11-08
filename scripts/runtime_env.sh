#!/usr/bin/env bash

xinfu_env_bin() {
  local environment="$1" override="${2:-}" conda_command
  if [[ -n "$override" ]]; then
    printf '%s\n' "$override"
    return
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$environment" && -n "${CONDA_PREFIX:-}" ]]; then
    printf '%s/bin\n' "$CONDA_PREFIX"
    return
  fi
  conda_command="${CONDA_EXE:-conda}"
  if ! command -v "$conda_command" >/dev/null 2>&1; then
    echo "未找到 Conda。请先激活 $environment，或设置对应的环境路径。" >&2
    return 1
  fi
  "$conda_command" run -n "$environment" python -c 'import sys; from pathlib import Path; print(Path(sys.executable).parent)'
}
