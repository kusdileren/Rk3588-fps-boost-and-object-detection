# RK3588 NPU vs CPU Object Detection Benchmark

YOLO26n ile Rockchip RK3588 (Rock5B) uzerinde NPU (RKNN) ve CPU (ONNX Runtime)
inference karsilastirmasi. Ayni letterbox/NMS/cizim mantigini kullanan iki
ayri benchmark programi icerir; boylece iki calisma zamani dogrudan
karsilastirilabilir.

## Klasor yapisi

```
.
├── scripts/
│   ├── setup_local_pc.sh       # Model donusturme PC'si icin kurulum (conda/rknn_env)
│   └── setup_rk3588_board.sh   # Rock5B karti icin kurulum (SDK'lar + build)
├── cpp_bench/
│   ├── CMakeLists.txt
│   ├── benchmark_cpu.cpp      # ONNX Runtime (CPU) inference
│   └── benchmark_npu.cpp      # RKNN API (NPU) inference
├── model_convert/
│   ├── convert_to_onnx.py     # .pt -> .onnx export + box/cls split
│   └── convert_from_video.py  # split .onnx -> int8 .rknn (video kalibrasyonlu)
├── third_party/                # setup_rk3588_board.sh tarafindan doldurulur, repoya girmez
├── .gitignore
└── README.md
```

> **Not:** Bu repo daha once `build/` klasoru icinde kaynak + derleme
> ciktilarinin karisik durdugu bir yapidan duzenlendi. Kendi makinende de
> ayni karisikligi yasamamak icin kaynak dosyalari (`.cpp`, `CMakeLists.txt`)
> her zaman `cpp_bench/` icinde tut, derlemeyi ayri bir `build/` klasorunde
> yap. `build/` klasoru `.gitignore` ile disarida tutuluyor.

## Hizli baslangic

Iki farkli makine kullaniyorsun; her biri icin ayri script var.

### 1) Model donusturme PC'si (WSL2 / x86_64 Linux)

```bash
chmod +x scripts/setup_local_pc.sh
./scripts/setup_local_pc.sh
conda activate rknn_env
```

Bu, Miniconda'yi (yoksa) kurar, `rknn_env` adinda Python 3.10 ortami acar ve
`ultralytics`, `onnx`, `onnxslim`, `rknn-toolkit2`, `opencv-python`
paketlerini kurar.

### 2) Rock5B / RK3588 karti

```bash
chmod +x scripts/setup_rk3588_board.sh
./scripts/setup_rk3588_board.sh
```

Bu script:
- Build araclarini ve OpenCV'yi apt ile kurar
- RKNN Runtime SDK'yi `airockchip/rknn-toolkit2` reposundan cekip
  `third_party/rknpu2/` altina yerlestirir
- ONNX Runtime (aarch64) release'ini indirip `third_party/onnxruntime/`
  altina cikartir
- Sonunda **projeyi hemen build etmek isteyip istemedigini sorar** —
  isterse otomatik build eder, istemezse elle build komutlarini gosterir

> Eskiden burada RKNN SDK'yi elle klonlayip kopyalama adimlari vardi;
> artik hepsi `setup_rk3588_board.sh` icinde otomatik.

## 3) Model hazirlama

### a) .pt -> split .onnx (RKNN icin)

```bash
cd model_convert
python3 convert_to_onnx.py --weights yolo26n.pt
```

Bu iki dosya uretir:
- `yolo26n.onnx` — standart export, **CPU benchmark icin kullan**
- `yolo26n_split.onnx` — box/cls ayri iki cikisli, **sadece RKNN donusturme icin**

Kart tarafina gecirmek icin:
```bash
scp yolo26n.onnx <kullanici>@<kart_ip>:~/npu_test/cpp_bench/build/
scp yolo26n_rk3588.rknn <kullanici>@<kart_ip>:~/npu_test/cpp_bench/build/
```

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

`setup_rk3588_board.sh` sana zaten build edip etmeyecegini soruyor. Elle
yapmak istersen (veya kaynak degistirdikten sonra tekrar derlemek icin):

```bash
cd cpp_bench
mkdir -p build && cd build
cmake .. \
  -DONNXRUNTIME_ROOT=$(pwd)/../../third_party/onnxruntime \
  -DRKNN_ROOT=$(pwd)/../../third_party/rknpu2
make -j$(nproc)
```

`ONNXRUNTIME_ROOT` / `RKNN_ROOT` verilmezse `CMakeLists.txt` varsayilan
olarak repo-koku/`third_party/` altina bakar (yani `setup_rk3588_board.sh`
calistirdiysan hicbir sey vermene gerek yok, dogrudan `cmake ..` yeterli).

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
  ile "Video I/O" bolumunu kontrol et. `setup_rk3588_board.sh` bu kontrolu
  otomatik yapip uyari veriyor.
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
- **CMake "ONNX Runtime bulunamadi" hatasi verir:**
  `scripts/setup_rk3588_board.sh` calistirilmamis olabilir, ya da
  `third_party/onnxruntime` silinmis olabilir. Script'i tekrar calistir
  ya da `cmake .. -DONNXRUNTIME_ROOT=/dogru/yol` ile elle gecir.

## Lisans

<buraya kendi lisansini ekle>
