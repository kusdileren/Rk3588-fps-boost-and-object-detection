#!/usr/bin/env bash
# =============================================================================
# setup_local_pc.sh
#
# Bu script'i model donusturme (.pt -> .onnx -> .rknn) islemini yapacagin
# PC'de (WSL2 veya native Linux, x86_64) calistir. Rock5B/RK3588 kartinda
# CALISTIRMA — o icin scripts/setup_rk3588_board.sh var.
#
# Ne yapar:
#   - Miniconda yoksa kurar
#   - "rknn_env" adinda Python 3.10 conda ortami olusturur
#   - Model donusturme icin gereken tum Python paketlerini kurar
#     (ultralytics, onnx, onnxslim, rknn-toolkit2, opencv-python)
#
# Kullanim:
#   chmod +x scripts/setup_local_pc.sh
#   ./scripts/setup_local_pc.sh
#   conda activate rknn_env
# =============================================================================
set -euo pipefail

ENV_NAME="rknn_env"
PY_VERSION="3.10"

echo "== [1/4] Klasor / dizin adi kontrolu =="
CURRENT_PATH="$(pwd)"
if echo "$CURRENT_PATH" | grep -qP '[^\x00-\x7F]'; then
  echo "UYARI: Bulundugun klasor yolunda ASCII disi (orn. Turkce) karakter var:"
  echo "  $CURRENT_PATH"
  echo "Bu, scp / cmake / conda ile bilinen sorunlara yol aciyor."
  echo "Devam etmeden once repo'yu sadece ASCII karakter iceren bir yola tasimani oneririm."
  read -rp "Yine de devam etmek istiyor musun? [y/N] " ans
  [[ "${ans:-N}" =~ ^[Yy]$ ]] || exit 1
fi

echo "== [2/4] Miniconda kontrolu =="
if ! command -v conda &>/dev/null; then
  echo "Miniconda bulunamadi, kuruluyor..."
  MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"
  ARCH="$(uname -m)"
  if [[ "$ARCH" == "x86_64" ]]; then
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  elif [[ "$ARCH" == "aarch64" ]]; then
    URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
  else
    echo "Desteklenmeyen mimari: $ARCH"; exit 1
  fi
  wget -O "$MINICONDA_INSTALLER" "$URL"
  bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda init bash || true
  echo "Miniconda kuruldu. Yeni terminalde tekrar calistirman gerekebilir."
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

echo "== [3/4] '$ENV_NAME' conda ortami olusturuluyor (Python $PY_VERSION) =="
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  echo "'$ENV_NAME' zaten var, atlaniyor."
else
  conda create -y -n "$ENV_NAME" python="$PY_VERSION"
fi

conda activate "$ENV_NAME"

echo "== [4/4] Python paketleri kuruluyor =="
# rknn-toolkit2 numpy/protobuf/onnx surumlerine karsi hassas; once temel
# paketleri sabitli kuruyoruz, sonra rknn-toolkit2'yi kuruyoruz.
pip install --upgrade pip
pip install "numpy<2.0" "protobuf==3.20.3"
pip install onnx onnxslim
pip install ultralytics
pip install opencv-python

echo "-- rknn-toolkit2 kuruluyor --"
pip install rknn-toolkit2 -i https://pypi.org/simple

echo ""
echo "Kurulum tamamlandi."
echo "Kullanmak icin:"
echo "  conda activate $ENV_NAME"
echo "  cd model_convert"
echo "  python3 convert_to_onnx.py --weights yolo26n.pt"