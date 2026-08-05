# RK3588 NPU vs CPU Object Detection Benchmark

YOLO26n ile Rockchip RK3588 (Rock5B) uzerinde NPU (RKNN) ve CPU (ONNX Runtime)
inference karsilastirmasi. Ayni letterbox/NMS/cizim mantigini kullanan iki
ayri benchmark programi icerir; boylece iki calisma zamani dogrudan
karsilastirilabilir.

## Klasor yapisi

```
.
├── cpp_bench/
│   ├── CMakeLists.txt
│   ├── benchmark_cpu.cpp      # ONNX Runtime (CPU) inference
│   └── benchmark_npu.cpp      # RKNN API (NPU) inference
├── model_convert/
│   ├── convert_to_onnx.py     # .pt -> .onnx export + box/cls split
│   └── convert_from_video.py  # split .onnx -> int8 .rknn (video kalibrasyonlu)
├── third_party/
│   └── rknpu2/                 # (repoya girmez, asagida indirme talimati var)
├── .gitignore
└── README.md
```

> **Not:** Bu repo daha once `build/` klasoru icinde kaynak + derleme
> ciktilarinin karisik durdugu bir yapidan duzenlendi. Kendi makinende de
> ayni karisikligi yasamamak icin kaynak dosyalari (`.cpp`, `CMakeLists.txt`)
> her zaman `cpp_bench/` icinde tut, derlemeyi ayri bir `build/` klasorunde
> yap (asagida anlatiliyor). `build/` klasoru `.gitignore` ile disarida
> tutuluyor.

## 1) Gereksinimler

- Ubuntu 22.04+ (Rock5B icin resmi RK3588 imaji onerilir)
- CMake >= 3.16, g++ (C++17)
- OpenCV 4.x (**FFmpeg destegiyle derlenmis olmali**, video yazma icin;
  `python3 -c "import cv2; print(cv2.getBuildInformation())"` ile kontrol et)
- ONNX Runtime (C++ headers + lib) — CPU benchmark icin
- RKNN Runtime SDK (`librknnrt.so`, `rknn_api.h`) — NPU benchmark icin
- Python 3.10 + conda/venv — model donusturme icin
  - `ultralytics`, `onnx`, `onnxslim`, `rknn-toolkit2`, `opencv-python`

## 2) RKNN SDK kurulumu (NPU tarafi icin zorunlu)

RKNN Runtime SDK bu repoya dahil degil (buyuk ve platforma ozel). Kur:

```bash
git clone https://github.com/airockchip/rknn-toolkit2.git third_party/rknpu2_src
# Runtime .so ve header'lari kendi projene kopyala:
mkdir -p third_party/rknpu2/{include,lib}
cp third_party/rknpu2_src/rknpu2/runtime/Linux/librknn_api/include/*.h third_party/rknpu2/include/
cp third_party/rknpu2_src/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so third_party/rknpu2/lib/
```

`CMakeLists.txt` bu yollari referans alacak sekilde ayarlanmali (`target_include_directories`
/ `target_link_libraries`, dosyanin icinde ornek var).

## 3) Model hazirlama

### a) .pt -> split .onnx (RKNN icin)

```bash
cd model_convert
python3 convert_to_onnx.py --weights yolo26n.pt
```

Bu iki dosya uretir:
- `yolo26n.onnx` — standart export, **CPU benchmark icin kullan**
- `yolo26n_split.onnx` — box/cls ayri iki cikisli, **sadece RKNN donusturme icin**
- scp yolo26n.onnx alpagut@<ip>:~/npu_test/cpp_bench/build/
- scp yolo26n_rk3588_v2.rknn alpagut@<ip>:~/npu_test/cpp_bench/build/


> **Neden split gerekli:** Ultralytics'in tek-cikisli export'unda box
> koordinatlari (0-640 piksel) ve class skorlari (0-1) ayni int8 scale'i
> paylasiyor. Int8 quantize ederken bu, class skorlarinin neredeyse
> tamamen 0'a yuvarlanmasina ve NPU tarafinda tespit sayisinin 0
> cikmasina neden oluyor. `convert_to_onnx.py` bunu otomatik olarak
> ayirir.

### b) split .onnx -> int8 .rknn

```bash
python3 convert_from_video.py \
  --weights yolo26n_split.onnx \
  --target rk3588 \
  --dtype int8 \
  --video test_video.mp4 \
  --output yolo26n_rk3588.rknn
```

`--video` kalibrasyon icin kullanilir; gercek kullanim sahnesine ne kadar
yakinsa quantization o kadar iyi sonuc verir.

## 4) Derleme (out-of-source build)

```bash
cd cpp_bench
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Bu, `benchmark_cpu` ve `benchmark_npu` binary'lerini `build/` icinde
uretir; `build/` git'e girmez, her makinede yeniden olusturulur.

## 5) Calistirma

```bash
# CPU (fp32, tek-cikisli standart onnx ile)
./benchmark_cpu --input test_video.mp4 --model yolo26n.onnx --output cpu_test.mp4

# NPU (int8, split'ten uretilen rknn ile)
./benchmark_npu --input test_video.mp4 --output npu_test.mp4 --model yolo26n_rk3588.rknn
```

Ortak parametreler: `--conf 0.4`, `--imgsz 640`, `--show`,
`--core-mask auto|0|1|2|all` (sadece NPU).

## 6) Bilinen notlar / sorun giderme

- **Video ciktisi acilamiyor / GStreamer "cannot link elements":**
  OpenCV'nin video yazma backend'i (FFmpeg/GStreamer) sistemde eksik
  olabilir. `python3 -c "import cv2; print(cv2.getBuildInformation())"`
  ile "Video I/O" bolumunu kontrol et.
- **Video kalitesi dusuk:** Varsayilan `mp4v` codec dusuk bitrate
  kullanir; kod `avc1` (H.264) deniyor, olmazsa `mp4v`'ye dusuyor.
  Sistemde gercek bir H.264 encoder oldugundan emin ol
  (`ffmpeg -encoders | grep 264`).
- **NPU tarafinda tespit sayisi 0:** `.onnx` split edilmeden RKNN'e
  verilmis olabilir — yukaridaki adim 3'u kontrol et.
- **NPU kutulari CPU'ya gore daha "titrek":** int8 quantization'in
  beklenen bir yan etkisi (box koordinatlari da quantize ediliyor,
  anchor rekabeti degisebiliyor). Daha stabil gorunum icin basit bir
  EMA/tracker eklenebilir.

## Lisans

<buraya kendi lisansini ekle>
