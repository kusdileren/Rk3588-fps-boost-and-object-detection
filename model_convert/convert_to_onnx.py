"""
convert_to_onnx.py
===================
.pt modelini once standart ONNX'e export eder, sonra RKNN int8 quantize
icin son Concat node'unu otomatik bulup box/cls ciktilarini AYRI iki
tensor olarak ayirir.

Neden gerekli:
Ultralytics export'u tek bir cikis tensoru uretir:
  output0: [1, 4+num_classes, num_anchors]  (box coord + sigmoid class skoru)
Bu tek tensoru RKNN int8 quantize ederken box (0-640 piksel) ile class
skoru (0-1 olasilik) AYNI int8 scale'i paylasir. Scale buyuk box
degerlerine gore secildigi icin class skorlari neredeyse tamamen 0'a
yuvarlanir -> NPU tarafinda tespit sayisi 0 cikar.

Kullanim:
    python3 convert_to_onnx.py
    python3 convert_to_onnx.py --weights yolo26n.pt --opset 12
"""

import argparse
import sys
from pathlib import Path

import onnx
from ultralytics import YOLO


# ----------------------------------------------------------------
# Adim 1: .pt -> ONNX (Ultralytics export)
# ----------------------------------------------------------------
def export_onnx(pt_path: str, opset: int = 12, simplify: bool = True,
                 end2end: bool = False) -> str:
    model = YOLO(pt_path)
    model.export(
        format="onnx",
        opset=opset,
        simplify=simplify,
        half=False,
        end2end=end2end  # YOLO26'da NPU segfault'unu onlemek icin False sart
    )
    onnx_path = str(Path(pt_path).with_suffix(".onnx"))
    print(f"--> ONNX export tamamlandi: {onnx_path}")
    return onnx_path


# ----------------------------------------------------------------
# Adim 2: tek cikisli ONNX -> box/cls ayri iki cikisli ONNX
# ----------------------------------------------------------------
def find_final_concat(model: onnx.ModelProto):
    """Tek cikisli graph'in son node'unu bulur (beklenen: Concat)."""
    graph = model.graph
    if len(graph.output) != 1:
        raise RuntimeError(
            f"Split icin tek cikisli model bekleniyor, bulunan cikis sayisi: {len(graph.output)}"
        )
    out_name = graph.output[0].name
    for node in graph.node:
        if out_name in node.output:
            return node
    raise RuntimeError(f"'{out_name}' tensorunu ureten node bulunamadi.")


def split_box_cls(input_path: str, output_path: str) -> str:
    model = onnx.load(input_path)
    concat_node = find_final_concat(model)

    if concat_node.op_type != "Concat":
        raise RuntimeError(
            f"Beklenen son node tipi 'Concat', bulunan: '{concat_node.op_type}'. "
            "Model yapisi farkli olabilir (ornegin end2end=True export), "
            "graph'i elle incelemek gerekebilir."
        )

    if len(concat_node.input) != 2:
        raise RuntimeError(
            f"Son Concat node'unun 2 girdisi bekleniyordu, bulunan: {len(concat_node.input)}. "
            "Graph'i elle incelemek gerekir."
        )

    input_a, input_b = concat_node.input
    print(f"--> Son Concat node bulundu: {concat_node.name}")
    print(f"    Girdi A: {input_a}")
    print(f"    Girdi B: {input_b}")

    graph_input_names = [i.name for i in model.graph.input]

    onnx.utils.extract_model(
        input_path,
        output_path,
        input_names=graph_input_names,
        output_names=[input_a, input_b],
    )

    split_model = onnx.load(output_path)
    print("--> Split model cikislari:")
    for o in split_model.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param
                for d in o.type.tensor_type.shape.dim]
        print(f"    {o.name}: {dims}")

    print(f"--> Split ONNX kaydedildi: {output_path}")
    return output_path


# ----------------------------------------------------------------
# Ana akis: .pt -> ONNX -> split ONNX
# ----------------------------------------------------------------
def convert_to_split_onnx(pt_path: str, opset: int = 12, simplify: bool = True,
                           end2end: bool = False) -> str:
    onnx_path = export_onnx(pt_path, opset=opset, simplify=simplify, end2end=end2end)
    split_path = str(Path(onnx_path).with_name(Path(onnx_path).stem + "_split.onnx"))
    split_box_cls(onnx_path, split_path)
    print(f"\nTamamlandi. RKNN donusturme icin kullanilacak dosya: {split_path}")
    return split_path


def main():
    parser = argparse.ArgumentParser(
        description=".pt -> ONNX export + RKNN icin box/cls split (tek adimda)"
    )
    parser.add_argument("--weights", type=str, default="yolo26n.pt", help=".pt model yolu")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--no-simplify", action="store_true", help="ONNX simplify adimini atla")
    parser.add_argument("--end2end", action="store_true",
                         help="DIKKAT: YOLO26'da NPU segfault'una neden olabilir")
    args = parser.parse_args()

    try:
        convert_to_split_onnx(
            pt_path=args.weights,
            opset=args.opset,
            simplify=not args.no_simplify,
            end2end=args.end2end,
        )
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()