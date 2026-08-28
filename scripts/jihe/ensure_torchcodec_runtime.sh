#!/usr/bin/env bash
set -euo pipefail

# JiHe training images may contain the Python TorchCodec wheel without the
# FFmpeg shared libraries that it loads at runtime. This helper is sourced by
# the Stage 3 launchers before any output directory or GPU process is created.

[[ -n "${FASTWAM_ENV:-}" ]] || {
  echo "[error] FASTWAM_ENV must be set before sourcing ensure_torchcodec_runtime.sh" >&2
  return 1
}

torchcodec_import_ok() {
  "${FASTWAM_ENV}/bin/python" - <<'PY' >/dev/null 2>&1
import torchcodec
from torchcodec.decoders import VideoDecoder
PY
}

if ! torchcodec_import_ok; then
  if [[ "${FASTWAM_AUTO_INSTALL_FFMPEG:-1}" != "1" ]]; then
    echo "[error] TorchCodec cannot load FFmpeg and automatic installation is disabled" >&2
    return 1
  fi
  if ((EUID != 0)); then
    echo "[error] TorchCodec cannot load FFmpeg; rerun in a root JiHe container or preinstall FFmpeg 4-7" >&2
    return 1
  fi
  command -v apt-get >/dev/null 2>&1 || {
    echo "[error] TorchCodec cannot load FFmpeg and apt-get is unavailable" >&2
    return 1
  }

  echo "[video] FFmpeg shared libraries are missing; installing the Ubuntu ffmpeg runtime"
  export DEBIAN_FRONTEND=noninteractive
  apt-get -o Acquire::Retries=3 update
  apt-get -o Acquire::Retries=3 install -y --no-install-recommends ffmpeg
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig
  fi
fi

if ! torchcodec_import_ok; then
  echo "[error] TorchCodec still cannot load after FFmpeg installation" >&2
  echo "[error] Expected a TorchCodec-compatible FFmpeg 4, 5, 6, or 7 runtime" >&2
  return 1
fi

"${FASTWAM_ENV}/bin/python" - <<'PY'
import torchcodec
from torchcodec.decoders import VideoDecoder
print(f"[video] torchcodec={torchcodec.__version__} ffmpeg_runtime=ok")
PY
