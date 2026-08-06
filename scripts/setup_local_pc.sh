#!/usr/bin/env bash
# =============================================================================
# setup_local_pc.sh
#
# Bu script'i model donusturme (.pt -> .onnx -> .rknn) islemini yapacagin
# PC'de (WSL2 veya native Linux, x86_64) calistir. Rock5B/RK3588 kartinda
# CALISTIRMA — o icin scripts/setup_rk3588_board.sh var.
#
# Onemli: rknn-toolkit2'nin native Windows icin resmi wheel paketi yok,
# sadece Linux (manylinux_x86_64) icin var. Bu yuzden bu script'i
# PowerShell'den DEGIL, WSL (ya da baska bir Linux) icinden calistir.
#
# Ne yapar:
#   - uv yoksa kurar (Astral'in resmi kurulum script'i ile)
#   - "rknn_env" adinda Python 3.10 sanal ortami olusturur (uv venv)
#   - Model donusturme icin gereken tum Python paketlerini kurar
#     (ultralytics, onnx, onnxslim, rknn-toolkit2, opencv-python)
#
# Kullanim:
#   chmod +x scripts/setup_local_pc.sh
#   ./scripts/setup_local_pc.sh
#   source rknn_env/bin/activate
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="rknn_env"
PY_VERSION="3.10"

echo "== [1/4] Ortam kontrolu =="
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "WSL tespit edildi, devam ediliyor."
elif [[ "$(uname -s)" == "Linux" ]]; then
  echo "Native Linux tespit edildi, devam ediliyor."
else
  echo "UYARI: Bu script Linux/WSL disinda calisamaz (rknn-toolkit2 native"
  echo "Windows wheel paketi sunmuyor). PowerShell'den degil, WSL terminalinden"
  echo "calistir."
  exit 1
fi

echo "== [2/4] Klasor / dizin adi kontrolu =="
CURRENT_PATH="$(pwd)"
if echo "$CURRENT_PATH" | grep -qP '[^\x00-\x7F]'; then
  echo "UYARI: Bulundugun klasor yolunda ASCII disi (orn. Turkce) karakter var:"
  echo "  $CURRENT_PATH"
  echo "Bu, scp / cmake / uv ile bilinen sorunlara yol aciyor."
  read -rp "Yine de devam etmek istiyor musun? [y/N] " ans
  [[ "${ans:-N}" =~ ^[Yy]$ ]] || exit 1
fi

echo "== [3/4] uv kontrolu =="
if ! command -v uv &>/dev/null; then
  echo "uv bulunamadi, kuruluyor..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv surumu: $(uv --version)"

echo "== [4/4] '$ENV_NAME' sanal ortami olusturuluyor (Python $PY_VERSION) =="
if [[ -d "$ENV_NAME" ]]; then
  echo "'$ENV_NAME' zaten var, atlaniyor (yeniden kurmak icin klasoru sil: rm -rf $ENV_NAME)."
else
  uv venv --python "$PY_VERSION" "$ENV_NAME"
fi

# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"

echo "-- Python paketleri kuruluyor --"
# rknn-toolkit2 numpy/protobuf/onnx surumlerine karsi hassas; once temel
# paketleri sabitli kuruyoruz, sonra rknn-toolkit2'yi kuruyoruz.
uv pip install "numpy<2.0" "protobuf==3.20.3"
uv pip install onnx onnxslim
uv pip install ultralytics
uv pip install opencv-python
uv pip install rknn-toolkit2

echo ""
echo "Kurulum tamamlandi."
echo "Kullanmak icin:"
echo "  source $ENV_NAME/bin/activate"
echo "  cd model_convert"
echo "  ./convert_all.sh --weights yolo26n.pt --video ../test_video.mp4"
