#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import cv2
import tempfile
import os
from rknn.api import RKNN


def extract_frames_from_video(video_path, temp_dir, num_frames=100):
    """Videodan eşit aralıklarla frame çeker ve geçici klasöre kaydeder."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise RuntimeError("Videoda hiç kare bulunamadı!")

    # Kareleri eşit aralıklarla çekmek için adım boyutu hesapla
    step = max(1, total_frames // num_frames)
    
    txt_path = os.path.join(temp_dir, 'dataset.txt')
    extracted = 0
    frame_idx = 0
    
    print(f"--> Videodan {num_frames} adet kalibrasyon karesi geçici belleğe çıkarılıyor...")
    
    with open(txt_path, 'w') as f:
        while cap.isOpened() and extracted < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0:
                img_path = os.path.join(temp_dir, f"frame_{extracted:03d}.jpg")
                cv2.imwrite(img_path, frame)
                f.write(f"{img_path}\n")
                extracted += 1
                
            frame_idx += 1

    cap.release()
    print(f"--> {extracted} adet kare başarıyla çıkarıldı.")
    return txt_path


def convert_onnx_to_rknn(
    weights, target, dtype, video_path, out_path, mean, std
):
    rknn = RKNN(verbose=True)

    # Geçici bir klasör oluşturuyoruz (with bloğu bittiğinde içindekilerle beraber otomatik silinir)
    with tempfile.TemporaryDirectory() as temp_dir:
        
        try:
            print('--> Model yapılandırılıyor...')
            rknn.config(
            mean_values=[mean],
            std_values=[std],
            target_platform=target,
            optimization_level=3,
            quantized_algorithm='normal',
            quantized_method='channel',
        )

            print('--> ONNX modeli yükleniyor...')
            ret = rknn.load_onnx(model=str(weights))
            if ret != 0:
                raise RuntimeError('ONNX modeli yüklenemedi!')

            print('--> Model derleniyor (Build)...')
            do_quant = dtype.lower() in ['int8', 'uint8']

            dataset_file = None
            if do_quant:
                if video_path is None:
                    raise ValueError(
                        'int8/uint8 kuantizasyonu için --video parametresi zorunludur!'
                    )
                # Videodan resimleri çıkar ve txt dosyasının yolunu al
                dataset_file = extract_frames_from_video(video_path, temp_dir)

            ret = rknn.build(do_quantization=do_quant, dataset=dataset_file)
            if ret != 0:
                raise RuntimeError('RKNN model derleme başarısız!')

            print(f'--> RKNN model aktarılıyor: {out_path}')
            ret = rknn.export_rknn(str(out_path))
            if ret != 0:
                raise RuntimeError('RKNN dışa aktarma başarısız!')

            print('Dönüştürme işlemi başarıyla tamamlandı.')
        finally:
            rknn.release()


def main():
    parser = argparse.ArgumentParser(
        description='ONNX modelini videodan kalibre ederek RKNN formatına dönüştürücü'
    )
    parser.add_argument(
        '--weights', type=str, required=True, help='ONNX model yolu'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Hedef platform (ör. rk3588, rk3566)',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='int8',
        choices=['int8', 'fp16'],
        help='Kuantizasyon tipi (varsayılan: int8)',
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Kuantizasyon kalibrasyonu için kullanılacak test videosunun yolu',
    )
    parser.add_argument(
        '--output', type=str, required=True, help='Çıktı .rknn dosya yolu'
    )
    parser.add_argument(
        '--mean',
        type=float,
        nargs=3,
        default=[0, 0, 0],
        help='Mean değerleri (ör: --mean 0 0 0)',
    )
    parser.add_argument(
        '--std',
        type=float,
        nargs=3,
        default=[255, 255, 255],
        help='Std değerleri (ör: --std 255 255 255)',
    )

    args = parser.parse_args()

    try:
        convert_onnx_to_rknn(
            weights=Path(args.weights),
            target=args.target,
            dtype=args.dtype,
            video_path=args.video,
            out_path=Path(args.output),
            mean=args.mean,
            std=args.std,
        )
    except Exception as e:
        print(f'Hata: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()