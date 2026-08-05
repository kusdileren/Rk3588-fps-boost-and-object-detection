#!/usr/bin/env bash
# =============================================================================
# setup_rk3588_board.sh
#
# Bu script'i Rock5B / RK3588 kartinin UZERINDE calistir. Model donusturme
# yapilan PC'de degil — o icin scripts/setup_local_pc.sh var.
#
# Ne yapar:
#   1) apt ile build araclarini (cmake, g++, git, wget) kurar
#   2) OpenCV'yi (FFmpeg destegiyle) kontrol eder
#   3) RKNN Runtime SDK'yi resmi airockchip/rknn-toolkit2 reposundan cekip
#      sudo ile /usr/include/rknn ve /usr/lib altina kurar
#      (CMakeLists.txt bu yollari sabit bekliyor)
#   4) ONNX Runtime (aarch64) release tarball'ini indirip
#      $HOME/onnxruntime-linux-aarch64-<versiyon> altina acar
#      (CMakeLists.txt varsayilan olarak burayi ariyor)
#   5) Sonunda projeyi hemen build edip etmeyecegini sorar
#
# Kullanim:
#   chmod +x scripts/setup_rk3588_board.sh
#   ./scripts/setup_rk3588_board.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RKNN_SDK_TAG="v2.3.0"                # ihtiyaca gore guncelle (rknn_api.h RKNN_API_VERSION ile karsilastir)
ONNXRUNTIME_VERSION="1.19.2"
ONNXRUNTIME_ARCHIVE="onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_ARCHIVE}"
ONNXRUNTIME_INSTALL_DIR="$HOME/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}"

echo "Repo kok dizini: $REPO_ROOT"

echo "== [1/5] Sistem paketleri kuruluyor =="
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake g++ git wget pkg-config \
  libopencv-dev

echo "== [2/5] OpenCV FFmpeg destegi kontrolu =="
if command -v python3 &>/dev/null; then
  python3 - <<'PYEOF' || true
try:
    import cv2
    info = cv2.getBuildInformation()
    if "FFMPEG" in info and "YES" in info.split("FFMPEG")[1].splitlines()[0]:
        print("OK: OpenCV FFmpeg destegiyle derlenmis.")
    else:
        print("UYARI: OpenCV FFmpeg destegi bulunamadi. Video yazma calismayabilir.")
except ImportError:
    print("UYARI: python3-opencv bulunamadi, kontrol atlanmadi.")
PYEOF
fi

echo "== [3/5] RKNN Runtime SDK indirilip sisteme kuruluyor (${RKNN_SDK_TAG}) =="
if [[ -f /usr/include/rknn/rknn_api.h && -f /usr/lib/librknnrt.so ]]; then
  echo "RKNN SDK zaten kurulu (/usr/include/rknn, /usr/lib), atlaniyor."
else
  TMP_DIR="$(mktemp -d)"
  git clone --depth 1 --branch "$RKNN_SDK_TAG" \
    https://github.com/airockchip/rknn-toolkit2.git "$TMP_DIR/rknn-toolkit2" \
    || git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git "$TMP_DIR/rknn-toolkit2"

  sudo mkdir -p /usr/include/rknn
  sudo cp "$TMP_DIR"/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include/*.h \
     /usr/include/rknn/
  sudo cp "$TMP_DIR"/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so \
     /usr/lib/
  sudo ldconfig
  rm -rf "$TMP_DIR"
  echo "RKNN SDK /usr/include/rknn ve /usr/lib altina kuruldu."
fi

echo "== [4/5] ONNX Runtime (aarch64) indiriliyor (v${ONNXRUNTIME_VERSION}) =="
if [[ -d "$ONNXRUNTIME_INSTALL_DIR" ]]; then
  echo "$ONNXRUNTIME_INSTALL_DIR zaten var, atlaniyor."
else
  TMP_TGZ="$(mktemp)"
  wget -O "$TMP_TGZ" "$ONNXRUNTIME_URL"
  mkdir -p "$ONNXRUNTIME_INSTALL_DIR"
  tar -xzf "$TMP_TGZ" -C "$ONNXRUNTIME_INSTALL_DIR" --strip-components=1
  rm -f "$TMP_TGZ"
  echo "ONNX Runtime $ONNXRUNTIME_INSTALL_DIR altina cikartildi."
fi

echo "== [5/5] Kurulum tamamlandi =="
echo "RKNN SDK   : /usr/include/rknn, /usr/lib/librknnrt.so"
echo "ONNX Runtime: $ONNXRUNTIME_INSTALL_DIR"

echo ""
read -rp "Projeyi simdi build etmek ister misin? [y/N] " do_build
if [[ "${do_build:-N}" =~ ^[Yy]$ ]]; then
  echo "== Build baslatiliyor (cpp_bench/build) =="
  cd cpp_bench
  mkdir -p build && cd build
  cmake ..
  make -j"$(nproc)"
  echo ""
  echo "Build tamamlandi. Binary'ler: cpp_bench/build/benchmark_cpu, cpp_bench/build/benchmark_npu"
else
  echo ""
  echo "Build atlandi. Elle build etmek icin:"
  echo "  cd cpp_bench && mkdir -p build && cd build"
  echo "  cmake .."
  echo "  make -j\$(nproc)"
  echo ""
  echo "(ONNX Runtime farkli bir yerdeyse: cmake .. -DONNXRUNTIME_ROOT=/senin/yolun)"
fi