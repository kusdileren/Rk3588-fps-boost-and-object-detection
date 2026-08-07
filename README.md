# RK3588 NPU vs CPU Object Detection Benchmark

YOLO26n ile Rockchip RK3588 (Rock5B) uzerinde NPU (RKNN) ve CPU (ONNX Runtime)
inference karsilastirmasi. Ayni letterbox/NMS/cizim mantigini kullanan iki
ayri benchmark programi icerir; boylece iki calisma zamani dogrudan
karsilastirilabilir.

## Klasor yapisi

```
.
├── Dockerfile                  # Model donusturme ortami (onerilen yol)
├── docker-compose.yml          # Dockerfile'i model_convert/ mount'uyla calistirir
├── scripts/
│   ├── setup_local_pc.sh       # (legacy) uv/rknn_env ile native/WSL kurulum
│   └── setup_rk3588_board.sh   # Rock5B karti icin kurulum (SDK'lar + build)
├── cpp_bench/
│   ├── CMakeLists.txt
│   ├── benchmark_cpu.cpp      # ONNX Runtime (CPU) inference
│   └── benchmark_npu.cpp      # RKNN API (NPU) inference
├── model_convert/
│   ├── convert_to_onnx.py     # .pt -> .onnx export + box/cls split
│   └── convert_from_video.py  # split .onnx -> int8 .rknn (video kalibrasyonlu)
├── .gitignore
└── README.md
```

> `setup_rk3588_board.sh`, RKNN Runtime SDK'yi `/usr/include/rknn` ve
> `/usr/lib` altina (sistem geneli), ONNX Runtime'i ise
> `$HOME/onnxruntime-linux-aarch64-1.19.2` altina kurar. Ikisi de repo
> disinda oldugu icin repoya girmez.

> **Not:** Bu repo daha once `build/` klasoru icinde kaynak + derleme
> ciktilarinin karisik durdugu bir yapidan duzenlendi. Kendi makinende de
> ayni karisikligi yasamamak icin kaynak dosyalari (`.cpp`, `CMakeLists.txt`)
> her zaman `cpp_bench/` icinde tut, derlemeyi ayri bir `build/` klasorunde
> yap. `build/` klasoru `.gitignore` ile disarida tutuluyor.

## Hizli baslangic

Iki farkli makine kullaniyorsun; her biri icin ayri script var.

### 1) Model donusturme PC'si (Docker — onerilen)

`rknn-toolkit2`'nin native Windows icin resmi wheel paketi yok, sadece
Linux icin var. Eskiden bunun icin WSL + `uv` gerekiyordu; artik Docker
Desktop (Windows/Mac/Linux fark etmeksizin) yeterli, WSL ile ugrasmaya
gerek yok.

Once Docker kurulu olmali:
- **Windows/Mac:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) indirip kurun.
- **Linux (Ubuntu/Debian vb.):** Hizli kurulum icin asagidaki komutlari terminalde calistirabilirsiniz (veya [resmi dokumantasyona](https://docs.docker.com/engine/install/) bakabilirsiniz):

  ```bash
  curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
  sudo sh get-docker.sh
  sudo usermod -aG docker $USER
  newgrp docker 
  '''
Kurulumdan sonra:

```bash
docker compose build
docker compose run --rm model-convert bash
```

Bu, container icinde `/workspace/model_convert` altina `model_convert/`
klasorunu mount eder (ciktilar dogrudan host'ta olusur) ve seni bir
bash prompt'una birakir. Icinde:

```bash
./convert_all.sh --weights yolo26n.pt --video ../test_video.mp4
```

`docker-compose.yml`, kok dizindeki `test_video.mp4`'u da salt-okunur
olarak mount eder; kendi videon farkli bir isimdeyse
`docker-compose.yml` icindeki volume satirini guncelle.

> **Eski / legacy yol:** WSL2 veya native Linux'ta `uv` ile calismak
> istersen `scripts/setup_local_pc.sh` hala duruyor:
> ```bash
> chmod +x scripts/setup_local_pc.sh
> ./scripts/setup_local_pc.sh
> source rknn_env/bin/activate
> ```
> Bu, `uv`'yi (yoksa) kurar, `rknn_env` adinda Python 3.10 sanal ortami
> acar ve ayni paketleri (`ultralytics`, `onnx`, `onnxslim`,
> `rknn-toolkit2`, `opencv-python`) kurar. Docker kullanamiyorsan bu
> yolu tercih et.

### 2) Rock5B / RK3588 karti

```bash
chmod +x scripts/setup_rk3588_board.sh
./scripts/setup_rk3588_board.sh
```

Bu script:
- Build araclarini ve OpenCV'yi apt ile kurar
- RKNN Runtime SDK'yi `airockchip/rknn-toolkit2` reposundan cekip
  `sudo` ile `/usr/include/rknn` ve `/usr/lib` altina kurar
  (CMakeLists.txt bu yollari sabit bekliyor, RKNN SDK zaten
  kullaniciya/makineye ozel degil)
- ONNX Runtime (aarch64) release'ini indirip
  `$HOME/onnxruntime-linux-aarch64-1.19.2` altina cikartir
  (CMakeLists.txt varsayilan olarak burayi arar; farkli bir yere
  kurmak istersen `cmake .. -DONNXRUNTIME_ROOT=/senin/yolun` ile
  override edebilirsin)
- Sonunda **projeyi hemen build etmek isteyip istemedigini sorar** —
  isterse otomatik build eder, istemezse elle build komutlarini gosterir

> Eskiden burada RKNN SDK'yi elle klonlayip kopyalama adimlari vardi;
> artik hepsi `setup_rk3588_board.sh` icinde otomatik.

## 3) Model hazirlama

### Hizli yol: tek komut

```bash
cd model_convert
chmod +x convert_all.sh
./convert_all.sh --weights yolo26n.pt --video ../test_video.mp4
```

Bu, asagidaki (a) ve (b) adimlarini arka arkaya calistirir ve sonunda
board'a `scp` ile gonderip gondermek istemedigini sorar. `--weights` /
`--video` vermezsen mevcut `.pt` / `.mp4` dosyalarini bulup soruyla
teyit ettirir.

> **Not:** Bu script, `convert_to_onnx.py`'nin `<isim>_split.onnx`
> adlandirma kuralini varsayiyor (README'deki ornekle ayni). Kendi
> script'in farkli bir isimlendirme kullaniyorsa `convert_all.sh`
> icindeki `ONNX_SPLIT` degiskenini ona gore guncelle.

### Elle, adim adim

### a) .pt -> split .onnx (RKNN icin)

```bash
cd model_convert
python3 convert_to_onnx.py --weights yolo26n.pt
```

Bu iki dosya uretir:
- `yolo26n.onnx` — standart export, **CPU benchmark icin kullan**
- `yolo26n_split.onnx` — box/cls ayri iki cikisli, **sadece RKNN donusturme icin**

Kart tarafina gecirmek icin (DIKKAT: `build/` klasorunun ICINE degil,
`cpp_bench/` klasorune -- yani `build/`'in bir ustune -- gonder; cunku
temiz build almak icin zaman zaman `rm -rf build` yapacaksin ve build/
icine koydugun her sey bununla birlikte silinir):
```bash
scp yolo26n.onnx <kullanici>@<kart_ip>:~/Rk3588-fps-boost-and-object-detection/cpp_bench/
scp yolo26n_rk3588.rknn <kullanici>@<kart_ip>:~/Rk3588-fps-boost-and-object-detection/cpp_bench/
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
rm -rf build          # daha once bozuk/eski bir build/ varsa temizle
mkdir build && cd build
cmake ..
make -j$(nproc)
```

> **Dikkat — ic ice `build/build` tuzagi:** Bu adimlari calistirmadan
> once `pwd` ile hangi klasorde oldugunu kontrol et. Eger yanlislikla
> zaten `cpp_bench/build/` icindeyken tekrar `mkdir build && cd build`
> calistirirsan `cpp_bench/build/build/` gibi ic ice bir klasor
> olusturursun; bu durumda `cmake ..` bir onceki (dogru) build'in eski
> `CMakeCache.txt`'ini bulup sessizce onu kullanabilir, `make` de "no
> targets" hatasi verir. Emin degilsen `cd cpp_bench` (repo icindeki
> sabit path) ile en bastan basla:
> ```bash
> cd cpp_bench && pwd    # .../cpp_bench ile bitmeli, build/build DEGIL
> ```

`setup_rk3588_board.sh` calistirdiysan `ONNXRUNTIME_ROOT` icin hicbir sey
vermene gerek yok — `CMakeLists.txt` varsayilan olarak
`$HOME/onnxruntime-linux-aarch64-1.19.2` altina bakar. Farkli bir yere
kurduysan: `cmake .. -DONNXRUNTIME_ROOT=/senin/yolun`.

Bu, `benchmark_cpu` ve `benchmark_npu` binary'lerini `build/` icinde
uretir; `build/` git'e girmez, her makinede yeniden olusturulur.

## 5) Calistirma

Binary'ler `cpp_bench/build/` icinde, ama model dosyalarini ve test
videosunu `cpp_bench/` (bir ust dizin) icine koyduysan, `build/`
icinden calistirirken `../` ile referans ver:

```bash
cd build

# CPU (fp32, tek-cikisli standart onnx ile)
./benchmark_cpu --input ../test_video.mp4 --model ../yolo26n.onnx --output cpu_test.mp4

# NPU (int8, split'ten uretilen rknn ile)
./benchmark_npu --input ../test_video.mp4 --output npu_test.mp4 --model ../yolo26n_rk3588.rknn
```

> **Neden `../`:** `build/` klasorunu temiz build icin sik sik
> `rm -rf build` ile silip yeniden olusturacaksin. Model dosyalarini ve
> videoyu `build/` icine koyarsan bu komut onlari da siler. `cpp_bench/`
> icine koyup `../` ile referans vermek bu veri kaybini onler.

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
  `$HOME/onnxruntime-linux-aarch64-1.19.2` silinmis olabilir. Script'i
  tekrar calistir ya da `cmake .. -DONNXRUNTIME_ROOT=/dogru/yol` ile elle
  gecir.
- **`/usr/include/rknn/rknn_api.h` bulunamiyor:** RKNN SDK sisteme
  kurulmamis. `scripts/setup_rk3588_board.sh` tekrar calistir; script
  zaten `/usr/include/rknn` ve `/usr/lib/librknnrt.so` varsa adimi
  atlar, yoksa kurar.

## Lisans

<buraya kendi lisansini ekle>
