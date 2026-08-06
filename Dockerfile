# =============================================================================
# Dockerfile — model donusturme (.pt -> .onnx -> .rknn) ortami
#
# scripts/setup_local_pc.sh (uv + rknn_env) yerine gecti. Amac ayni:
# rknn-toolkit2'nin sadece Linux (manylinux_x86_64) wheel'i oldugu icin,
# Windows'ta WSL ugrasmadan dogrudan Docker Desktop ile calistirmak.
#
# Kullanim:
#   docker compose build
#   docker compose run --rm model-convert bash
#   # container icinde:
#   ./convert_all.sh --weights yolo26n.pt --video ../test_video.mp4
# =============================================================================
FROM python:3.10-slim

# opencv (video okuma/yazma) ve derleme icin gerekli sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# rknn-toolkit2 numpy/protobuf/onnx surumlerine karsi hassas; once temel
# paketleri sabitli kuruyoruz, sonra rknn-toolkit2'yi kuruyoruz.
RUN pip install --no-cache-dir "numpy<2.0" "protobuf==3.20.3" \
    && pip install --no-cache-dir onnx onnxslim \
    && pip install --no-cache-dir ultralytics \
    && pip install --no-cache-dir opencv-python-headless \
    && pip install --no-cache-dir rknn-toolkit2

# convert_all.sh, VIRTUAL_ENV bos ise rknn_env'i aktiflestirmeye calisiyor.
# Container icinde paketler zaten global kurulu oldugu icin bu kontrolu
# atlatiyoruz.
ENV VIRTUAL_ENV=/usr/local

CMD ["/bin/bash"]
