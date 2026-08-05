/*
 * Rock5B (RK3588) - NPU Benchmark
 * ================================================================
 * YOLOv8/YOLO26 (.rknn) ile video uzerinde nesne tespiti + NPU
 * inference benchmarki. RKNN Runtime (rknn_api.h / librknnrt.so)
 * kullanir. benchmark_cpu.cpp (ONNX Runtime / CPU) ile ayni CLI,
 * ayni letterbox/NMS/cizim mantigina sahiptir; boylece iki sonuc
 * dogrudan karsilastirilabilir.
 *
 * Bu .rknn modeli, benchmark_cpu.cpp'nin kullandigi yolov8n.onnx ile
 * AYNI export'tan (tek cikis, [1, 84, 8400], 4 box + 80 class skoru,
 * objectness yok) rknn-toolkit2 (v2.3.2, target rk3588) ile
 * int8 quantize edilerek uretilmistir. Girdi tensoru NHWC/UINT8'dir;
 * normalizasyon (mean=0, std=255) donusum sirasinda modele gomulmustur,
 * bu yuzden CPU tarafinda oldugu gibi elle /255 bolme veya HWC->CHW
 * transpose yapmaya GEREK YOKTUR - ham (letterbox'lanmis, RGB, uint8)
 * piksel verisi dogrudan NPU'ya verilir. Cikis, rknn_output.want_float=1
 * ile otomatik dequantize edilerek float32 olarak alinir; boylece
 * postprocess mantigi ONNX/CPU versiyonuyla birebir ayni kalir.
 *
 * Build (CMakeLists.txt icine eklenecek ornek):
 *   find_package(OpenCV REQUIRED)
 *   # librknnrt.so ve rknn_api.h RK3588 RKNPU2 SDK'sindan gelir:
 *   #   https://github.com/airockchip/rknn-toolkit2 -> rknpu2/runtime/Linux
 *   add_executable(benchmark_npu benchmark_npu.cpp)
 *   target_include_directories(benchmark_npu PRIVATE /path/to/rknpu2/include)
 *   target_link_libraries(benchmark_npu ${OpenCV_LIBS}
 *       /path/to/rknpu2/lib/librknnrt.so)
 *
 * Kullanim:
 *   ./benchmark_npu --input video.mp4 --output cikti.mp4 --model yolo26n_rk3588.rknn
 *   ./benchmark_npu --input video.mp4 --show
 *   ./benchmark_npu --input video.mp4 --core-mask all   # 3 NPU cekirdegini birden kullan
 */

#include <opencv2/opencv.hpp>
#include "rknn_api.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// ----------------------------------------------------------------
// COCO sinif ID'leri (CPU versiyonuyla ayni)
// 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
// ----------------------------------------------------------------
struct ClassInfo {
    std::string name;
    cv::Scalar color; // BGR
};

static const std::map<int, ClassInfo> CLASS_MAP = {
    {0, {"insan",      cv::Scalar(0, 255, 0)}},
    {1, {"bisiklet",   cv::Scalar(255, 255, 0)}},
    {2, {"araba",      cv::Scalar(255, 0, 0)}},
    {3, {"motosiklet", cv::Scalar(255, 0, 255)}},
    {5, {"otobus",     cv::Scalar(0, 255, 255)}},
    {7, {"kamyon",     cv::Scalar(0, 0, 255)}},
};

struct Detection {
    cv::Rect box;
    int class_id;
    float score;
};

struct Args {
    std::string input;
    std::string output = "output.mp4";
    std::string model  = "yolo26n_rk3588_v2.rknn";
    float conf         = 0.4f;
    int imgsz          = 640;
    bool show          = false;
    std::string core_mask = "auto"; // auto | 0 | 1 | 2 | all
};

// ----------------------------------------------------------------
// Basit argument parser (benchmark_cpu.cpp ile ayni)
// ----------------------------------------------------------------
static Args parse_args(int argc, char** argv) {
    Args args;
    bool has_input = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next_val = [&](void) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Eksik deger: " + a);
            }
            return std::string(argv[++i]);
        };

        if (a == "--input") {
            args.input = next_val();
            has_input = true;
        } else if (a == "--output") {
            args.output = next_val();
        } else if (a == "--model") {
            args.model = next_val();
        } else if (a == "--conf") {
            args.conf = std::stof(next_val());
        } else if (a == "--imgsz") {
            args.imgsz = std::stoi(next_val());
        } else if (a == "--show") {
            args.show = true;
        } else if (a == "--core-mask") {
            args.core_mask = next_val();
        } else {
            throw std::runtime_error("Bilinmeyen argument: " + a);
        }
    }

    if (!has_input) {
        throw std::runtime_error("--input zorunludur");
    }
    return args;
}

static rknn_core_mask parse_core_mask(const std::string& s) {
    if (s == "0") return RKNN_NPU_CORE_0;
    if (s == "1") return RKNN_NPU_CORE_1;
    if (s == "2") return RKNN_NPU_CORE_2;
    if (s == "01" || s == "0_1") return RKNN_NPU_CORE_0_1;
    if (s == "all" || s == "012" || s == "0_1_2") return RKNN_NPU_CORE_0_1_2;
    return RKNN_NPU_CORE_AUTO;
}

// ----------------------------------------------------------------
// Letterbox resize (benchmark_cpu.cpp ile birebir ayni)
// ----------------------------------------------------------------
struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
};

static cv::Mat letterbox(const cv::Mat& src, int target_size, LetterboxInfo& info) {
    int w = src.cols;
    int h = src.rows;
    float scale = std::min(static_cast<float>(target_size) / w,
                            static_cast<float>(target_size) / h);
    int new_w = static_cast<int>(std::round(w * scale));
    int new_h = static_cast<int>(std::round(h * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    int pad_x = (target_size - new_w) / 2;
    int pad_y = (target_size - new_h) / 2;

    cv::Mat out(target_size, target_size, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(pad_x, pad_y, new_w, new_h)));

    info.scale = scale;
    info.pad_x = pad_x;
    info.pad_y = pad_y;
    return out;
}

// ----------------------------------------------------------------
// NMS (benchmark_cpu.cpp ile birebir ayni)
// ----------------------------------------------------------------
static std::vector<Detection> run_nms(std::vector<Detection>& dets, float iou_thresh = 0.45f) {
    std::vector<Detection> result;
    if (dets.empty()) return result;

    std::map<int, std::vector<int>> class_indices;
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
        class_indices[dets[i].class_id].push_back(i);
    }

    for (auto& kv : class_indices) {
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        for (int idx : kv.second) {
            boxes.push_back(dets[idx].box);
            scores.push_back(dets[idx].score);
        }
        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, 0.0f, iou_thresh, keep);
        for (int k : keep) {
            result.push_back(dets[kv.second[k]]);
        }
    }
    return result;
}

// ----------------------------------------------------------------
// Kucuk yardimci: rknn_query hata kontrolu
// ----------------------------------------------------------------
static void check_rknn(int ret, const std::string& where) {
    if (ret != RKNN_SUCC) {
        throw std::runtime_error(where + " basarisiz, rknn hata kodu: " + std::to_string(ret));
    }
}

static void print_tensor_attr(const rknn_tensor_attr& attr) {
    std::cout << "  index=" << attr.index << " name=" << attr.name
              << " n_dims=" << attr.n_dims << " dims=[";
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        std::cout << attr.dims[i] << (i + 1 < attr.n_dims ? "," : "");
    }
    std::cout << "] fmt=" << get_format_string(attr.fmt)
              << " type=" << get_type_string(attr.type)
              << " qnt=" << get_qnt_type_string(attr.qnt_type)
              << " zp=" << attr.zp << " scale=" << attr.scale << std::endl;
}

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument hatasi: " << e.what() << std::endl;
        return 1;
    }

    // ---------------- RKNN kurulumu (NPU) ----------------
    rknn_context ctx = 0;
    try {
        int ret = rknn_init(&ctx, const_cast<char*>(args.model.c_str()), 0, 0, nullptr);
        check_rknn(ret, "rknn_init");

        rknn_core_mask core_mask = parse_core_mask(args.core_mask);
        if (core_mask != RKNN_NPU_CORE_AUTO) {
            ret = rknn_set_core_mask(ctx, core_mask);
            check_rknn(ret, "rknn_set_core_mask");
        }

        rknn_sdk_version sdk_ver;
        if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver)) == RKNN_SUCC) {
            std::cout << "RKNN API: " << sdk_ver.api_version
                      << " | Driver: " << sdk_ver.drv_version << std::endl;
        }

        rknn_input_output_num io_num;
        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        check_rknn(ret, "rknn_query(IN_OUT_NUM)");
        if (io_num.n_input != 1 || io_num.n_output != 2) {
            throw std::runtime_error("Bu program tek girisli / iki cikisli (box + cls ayri) "
                                      "YOLO .rknn modelleri icin yazildi (bkz. split export). Bulunan: "
                                      + std::to_string(io_num.n_input) + " girdi, "
                                      + std::to_string(io_num.n_output) + " cikti.");
        }

        rknn_tensor_attr input_attr;
        std::memset(&input_attr, 0, sizeof(input_attr));
        input_attr.index = 0;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
        check_rknn(ret, "rknn_query(INPUT_ATTR)");

        // Model iki ayri cikis verir: biri box (4 x N), diğeri sigmoid sonrasi
        // class skorlari (80 x N). Hangisinin index=0/1 oldugunu varsaymak yerine
        // dims[1]'e (kanal sayisi) bakarak tespit ediyoruz -> daha saglam.
        rknn_tensor_attr out_attr_raw[2];
        for (int i = 0; i < 2; ++i) {
            std::memset(&out_attr_raw[i], 0, sizeof(rknn_tensor_attr));
            out_attr_raw[i].index = i;
            ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr_raw[i], sizeof(rknn_tensor_attr));
            check_rknn(ret, "rknn_query(OUTPUT_ATTR)");
        }

        int box_out_idx = (out_attr_raw[0].dims[1] == 4) ? 0 : 1;
        int cls_out_idx = 1 - box_out_idx;
        const rknn_tensor_attr& box_attr = out_attr_raw[box_out_idx];
        const rknn_tensor_attr& cls_attr = out_attr_raw[cls_out_idx];

        if (box_attr.dims[1] != 4) {
            throw std::runtime_error("Box cikis tensoru beklenen [.,4,N] formatinda degil.");
        }

        std::cout << "Girdi tensoru:" << std::endl;
        print_tensor_attr(input_attr);
        std::cout << "Cikti tensoru (box, index=" << box_out_idx << "):" << std::endl;
        print_tensor_attr(box_attr);
        std::cout << "Cikti tensoru (cls, index=" << cls_out_idx << "):" << std::endl;
        print_tensor_attr(cls_attr);

        // Modelin gercek giris boyutu (H/W) tensordan okunuyor; --imgsz
        // sadece bilgi amacli tutuluyor, letterbox bu deger ile yapiliyor.
        int model_h = args.imgsz, model_w = args.imgsz, model_c = 3;
        bool nhwc = (input_attr.fmt == RKNN_TENSOR_NHWC);
        if (input_attr.n_dims == 4) {
            if (nhwc) {
                model_h = input_attr.dims[1];
                model_w = input_attr.dims[2];
                model_c = input_attr.dims[3];
            } else { // NCHW
                model_c = input_attr.dims[1];
                model_h = input_attr.dims[2];
                model_w = input_attr.dims[3];
            }
        }
        if (model_h != model_w) {
            std::cerr << "Uyari: kare olmayan giris (" << model_w << "x" << model_h
                      << "), letterbox mantigi kare varsayiyor." << std::endl;
        }
        const int input_size = model_h;
        if (model_c != 3) {
            throw std::runtime_error("Beklenmeyen kanal sayisi: " + std::to_string(model_c));
        }

        // ---------------- Video ac ----------------
        cv::VideoCapture cap(args.input);
        if (!cap.isOpened()) {
            std::cerr << "Video acilamadi: " << args.input << std::endl;
            rknn_destroy(ctx);
            return 1;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 25.0;
        int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        cv::VideoWriter writer;
        bool has_writer = !args.output.empty();
        if (has_writer) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(args.output, fourcc, fps, cv::Size(width, height));
            if (!writer.isOpened()) {
                std::cerr << "Cikti video dosyasi acilamadi: " << args.output << std::endl;
                has_writer = false;
            }
        }

        // ---------------- Benchmark degiskenleri ----------------
        double total_infer_time_ms = 0.0; // sadece rknn_run() suresi
        long bench_frame_count = 0;
        long frame_idx = 0;

        // Girdi buffer'i: letterbox sonrasi RGB, uint8, HWC (NHWC).
        // Normalizasyon (/255) modele gomulu oldugu icin burada YAPILMAZ.
        std::vector<uint8_t> input_tensor_values(
            static_cast<size_t>(input_size) * input_size * 3);

        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) break;
            ++frame_idx;

            // ---------- Preprocess: letterbox + BGR->RGB ----------
            LetterboxInfo lb_info;
            cv::Mat letterboxed = letterbox(frame, input_size, lb_info);

            cv::Mat rgb;
            cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
            // rgb zaten CV_8UC3, HWC duzeninde -> dogrudan kopyala
            std::memcpy(input_tensor_values.data(), rgb.data, input_tensor_values.size());

            rknn_input rk_input;
            std::memset(&rk_input, 0, sizeof(rk_input));
            rk_input.index = 0;
            rk_input.buf   = input_tensor_values.data();
            rk_input.size  = static_cast<uint32_t>(input_tensor_values.size());
            rk_input.type  = RKNN_TENSOR_UINT8;
            rk_input.fmt   = RKNN_TENSOR_NHWC;
            rk_input.pass_through = 0;

            int rin = rknn_inputs_set(ctx, 1, &rk_input);
            check_rknn(rin, "rknn_inputs_set");

            // ---------- Inference (sadece bu blok olculuyor) ----------
            auto t0 = std::chrono::high_resolution_clock::now();
            int rrun = rknn_run(ctx, nullptr);
            auto t1 = std::chrono::high_resolution_clock::now();
            check_rknn(rrun, "rknn_run");

            double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            total_infer_time_ms += infer_ms;
            ++bench_frame_count;

            // ---------- Ciktiyi al (dequantize edilmis float olarak) ----------
            // Box ve class skorlari artik AYRI tensorler; her biri kendi
            // int8 scale'iyle dequantize edildigi icin class skorlari (0-1)
            // artik box koordinatlarinin (0-640) scale'i tarafindan ezilmiyor.
            rknn_output rk_outputs[2];
            std::memset(rk_outputs, 0, sizeof(rk_outputs));
            rk_outputs[0].index      = box_out_idx;
            rk_outputs[0].want_float = 1;
            rk_outputs[1].index      = cls_out_idx;
            rk_outputs[1].want_float = 1;

            int rget = rknn_outputs_get(ctx, 2, rk_outputs, nullptr);
            check_rknn(rget, "rknn_outputs_get");

            rknn_output& box_out = (rk_outputs[0].index == box_out_idx) ? rk_outputs[0] : rk_outputs[1];
            rknn_output& cls_out = (rk_outputs[0].index == cls_out_idx) ? rk_outputs[0] : rk_outputs[1];
            float* box_data = reinterpret_cast<float*>(box_out.buf);
            float* cls_data = reinterpret_cast<float*>(cls_out.buf);

            // ---------- Postprocess (benchmark_cpu.cpp ile ayni format) ----------
            // box_attr: (1, 4, N)  cls_attr: (1, num_classes, N)
            int num_classes = static_cast<int>(cls_attr.dims[1]);
            int num_anchors = static_cast<int>(box_attr.dims[2]);

            std::vector<Detection> raw_dets;
            raw_dets.reserve(64);

            for (int a = 0; a < num_anchors; ++a) {
                float cx = box_data[0 * num_anchors + a];
                float cy = box_data[1 * num_anchors + a];
                float w  = box_data[2 * num_anchors + a];
                float h  = box_data[3 * num_anchors + a];

                int best_class = -1;
                float best_score = 0.0f;
                for (int c = 0; c < num_classes; ++c) {
                    if (CLASS_MAP.find(c) == CLASS_MAP.end()) continue;
                    float score = cls_data[c * num_anchors + a];
                    if (score > best_score) {
                        best_score = score;
                        best_class = c;
                    }
                }

                if (best_class == -1 || best_score < args.conf) continue;

                float x1 = (cx - w / 2.0f - lb_info.pad_x) / lb_info.scale;
                float y1 = (cy - h / 2.0f - lb_info.pad_y) / lb_info.scale;
                float x2 = (cx + w / 2.0f - lb_info.pad_x) / lb_info.scale;
                float y2 = (cy + h / 2.0f - lb_info.pad_y) / lb_info.scale;

                x1 = std::clamp(x1, 0.0f, static_cast<float>(width - 1));
                y1 = std::clamp(y1, 0.0f, static_cast<float>(height - 1));
                x2 = std::clamp(x2, 0.0f, static_cast<float>(width - 1));
                y2 = std::clamp(y2, 0.0f, static_cast<float>(height - 1));

                Detection det;
                det.box = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                                    cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
                det.class_id = best_class;
                det.score = best_score;
                raw_dets.push_back(det);
            }

            rknn_outputs_release(ctx, 2, rk_outputs);

            std::vector<Detection> final_dets = run_nms(raw_dets);

            // ---------- Cizim (benchmark_cpu.cpp ile ayni) ----------
            for (const auto& det : final_dets) {
                const auto& info = CLASS_MAP.at(det.class_id);
                cv::rectangle(frame, det.box, info.color, 2);

                char label_buf[64];
                std::snprintf(label_buf, sizeof(label_buf), "%s %.2f", info.name.c_str(), det.score);
                std::string label(label_buf);

                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                cv::Point tl = det.box.tl();
                cv::rectangle(frame,
                              cv::Point(tl.x, tl.y - text_size.height - 8),
                              cv::Point(tl.x + text_size.width + 4, tl.y),
                              info.color, -1);
                cv::putText(frame, label, cv::Point(tl.x + 2, tl.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
            }

            if (has_writer) {
                writer.write(frame);
            }

            if (args.show) {
                cv::imshow("Object Detection (NPU)", frame);
                if (cv::waitKey(1) == 'q') break;
            }

            if (frame_idx % 30 == 0) {
                std::cout << "[" << frame_idx << "] islenen kare - tespit sayisi: "
                          << final_dets.size() << std::endl;
            }
        }

        cap.release();
        if (has_writer) writer.release();
        if (args.show) cv::destroyAllWindows();

        std::cout << "Bitti." << std::endl;
        if (has_writer) {
            std::cout << "Cikti kaydedildi: " << args.output << std::endl;
        }

        // ---------------- Benchmark ozeti ----------------
        if (bench_frame_count > 0) {
            double avg_ms = total_infer_time_ms / bench_frame_count;
            double avg_fps = 1000.0 / avg_ms;

            std::cout << "\n--- Benchmark Sonuclari (NPU) ---" << std::endl;
            std::cout << "Frame Sayisi   : " << bench_frame_count << std::endl;
            std::cout.precision(2);
            std::cout << std::fixed;
            std::cout << "Ortalama Sure  : " << avg_ms << " ms" << std::endl;
            std::cout.precision(4);
            std::cout << "FPS            : " << avg_fps << std::endl;
        }

        rknn_destroy(ctx);
    } catch (const std::exception& e) {
        std::cerr << "Hata: " << e.what() << std::endl;
        if (ctx != 0) rknn_destroy(ctx);
        return 1;
    }

    return 0;
}
