#!/usr/bin/env bash
# =============================================================================
# convert_all.sh
#
# model_convert/ altindaki iki adimi (pt -> onnx/split-onnx, split-onnx ->
# int8 rknn) tek komutla zincirler, sonunda istersen board'a scp de eder.
#
# Bu script'i model donusturme PC'sinde, "rknn_env" sanal ortami zaten
# kuruluyken calistir (once ../scripts/setup_local_pc.sh calistirilmis
# olmali). Ortam otomatik aktiflestirilir, elle "source" etmene gerek yok.
#
# Kullanim (hepsi opsiyonel, verilmezse varsayilan/soru ile ilerler):
#   ./convert_all.sh --weights yolo26n.pt --video test_video.mp4 \
#       --target rk3588 --dtype int8
#
# Sadece calistirip sorulari cevaplamak da yeterli:
#   ./convert_all.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WEIGHTS=""
VIDEO=""
TARGET="rk3588"
DTYPE="int8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --video)   VIDEO="$2"; shift 2 ;;
    --target)  TARGET="$2"; shift 2 ;;
    --dtype)   DTYPE="$2"; shift 2 ;;
    *) echo "Bilinmeyen parametre: $1"; exit 1 ;;
  esac
done

echo "== [0/3] Ortam kontrolu =="
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Sanal ortam aktif degil, aktiflestiriliyor..."
  ENV_PATH="$(cd "$SCRIPT_DIR/.." && pwd)/rknn_env"
  if [[ ! -f "$ENV_PATH/bin/activate" ]]; then
    echo "HATA: $ENV_PATH bulunamadi. Once ../scripts/setup_local_pc.sh calistir."
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$ENV_PATH/bin/activate"
fi

# --- Eksik parametreleri sor -------------------------------------------------
if [[ -z "$WEIGHTS" ]]; then
  DEFAULT_W="$(ls -1 *.pt 2>/dev/null | head -n1 || true)"
  read -rp "Weights dosyasi (.pt) [${DEFAULT_W:-yolo26n.pt}]: " ans
  WEIGHTS="${ans:-${DEFAULT_W:-yolo26n.pt}}"
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "HATA: $WEIGHTS bulunamadi (model_convert/ altinda olmali)."; exit 1
fi

if [[ -z "$VIDEO" ]]; then
  DEFAULT_V="$(ls -1 ../*.mp4 2>/dev/null | head -n1 || true)"
  read -rp "Kalibrasyon videosu (.mp4) [${DEFAULT_V:-../test_video.mp4}]: " ans
  VIDEO="${ans:-${DEFAULT_V:-../test_video.mp4}}"
fi
if [[ ! -f "$VIDEO" ]]; then
  echo "HATA: $VIDEO bulunamadi."; exit 1
fi

# convert_to_onnx.py ciktilari, --weights ile verilen dosyanin bulundugu
# klasore yaziyor (cwd'ye degil) -- o yuzden yollari weights'in dizinine
# gore hesapliyoruz.
WEIGHTS_DIR="$(dirname "$WEIGHTS")"
BASE_NAME="$(basename "$WEIGHTS" .pt)"
ONNX_STD="${WEIGHTS_DIR}/${BASE_NAME}.onnx"
ONNX_SPLIT="${WEIGHTS_DIR}/${BASE_NAME}_split.onnx"
RKNN_OUT="${WEIGHTS_DIR}/${BASE_NAME}_${TARGET}.rknn"

echo "== [1/3] .pt -> .onnx (+ split) =="
python3 convert_to_onnx.py --weights "$WEIGHTS"

if [[ ! -f "$ONNX_SPLIT" ]]; then
  echo "HATA: beklenen $ONNX_SPLIT uretilmedi. convert_to_onnx.py ciktisini kontrol et."
  exit 1
fi

echo "== [2/3] split .onnx -> int8 .rknn =="
python3 convert_from_video.py \
  --weights "$ONNX_SPLIT" \
  --target "$TARGET" \
  --dtype "$DTYPE" \
  --video "$VIDEO" \
  --output "$RKNN_OUT"

echo ""
echo "Tamamlandi:"
echo "  CPU icin   -> $ONNX_STD"
echo "  NPU icin   -> $RKNN_OUT"

echo ""
read -rp "Bu iki dosyayi (ve kalibrasyon videosunu) Rock5B'ye scp ile gondermek ister misin? [y/N] " do_scp
if [[ "${do_scp:-N}" =~ ^[Yy]$ ]]; then
  read -rp "Board kullanici@ip (orn. rock@192.168.1.50): " BOARD_HOST
  # Onemli: build/ ICINE degil, cpp_bench/ (build'in bir ustu) icine
  # gonderiyoruz -- cunku "rm -rf build" ile temiz build alirken
  # build/ icindeki her sey (model dosyalari, test videosu dahil) silinir.
  # cpp_bench/ klasoru asla silinmedigi icin veri dosyalari icin guvenli yer.
  read -rp "Board hedef klasor [~/npu_test/cpp_bench/]: " BOARD_PATH
  BOARD_PATH="${BOARD_PATH:-~/npu_test/cpp_bench/}"
  scp "$ONNX_STD" "$RKNN_OUT" "$VIDEO" "${BOARD_HOST}:${BOARD_PATH}"
  echo "Gonderildi: ${BOARD_HOST}:${BOARD_PATH}"
  echo ""
  echo "NOT: build/ klasorunu 'rm -rf build' ile silsen bile bu dosyalar"
  echo "     etkilenmez, cunku build/'in DISINA gonderildi."
else
  echo "scp atlandi. Elle gondermek icin (build/ ICINE DEGIL, cpp_bench/ ICINE):"
  echo "  scp $ONNX_STD $RKNN_OUT $VIDEO <kullanici>@<board_ip>:~/npu_test/cpp_bench/"
fi
