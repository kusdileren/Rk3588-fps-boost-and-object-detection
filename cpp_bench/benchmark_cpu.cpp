/*
 * Rock5B (RK3588) - CPU Benchmark
 * ================================================================
 * YOLOv8n (.onnx) ile video uzerinde nesne tespiti + CPU inference
 * benchmarki. ONNX Runtime (CPUExecutionProvider) kullanir.
 *
 * Insan, araba, kamyon, otobus, motosiklet, bisiklet siniflarini
 * tespit edip video uzerine kutu (bounding box) cizer.
 *
 * Build (CMakeLists.txt icine eklenecek ornek):
 *   find_package(OpenCV REQUIRED)
 *   find_package(onnxruntime REQUIRED)   # ya da FetchContent / manuel include+lib
 *   add_executable(benchmark_cpu benchmark_cpu.cpp)
 *   target_link_libraries(benchmark_cpu ${OpenCV_LIBS} onnxruntime)
 *
 * Kullanim:
 *   ./benchmark_cpu --input video.mp4 --output cikti.mp4 --model yolov8n.onnx
 *   ./benchmark_cpu --input test_video.mp4 --show
 */

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// ----------------------------------------------------------------
// COCO sinif ID'leri (yolov8n.onnx varsayilan modelinde):
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
    std::string model  = "yolo26n.onnx";
    float conf         = 0.4f;
    int imgsz          = 640;
    bool show          = false;
};

// ----------------------------------------------------------------
// Basit argument parser (Python'daki argparse'a benzer davranis)
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
        } else {
            throw std::runtime_error("Bilinmeyen argument: " + a);
        }
    }

    if (!has_input) {
        throw std::runtime_error("--input zorunludur");
    }
    return args;
}

// ----------------------------------------------------------------
// Letterbox resize: en-boy oranini koruyarak imgsz x imgsz kare
// tuvale yerlestirir (padding gri renkte, 114/114/114 - ultralytics
// varsayilani ile ayni).
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
// NMS (class-agnostic degil, sinif bazli - OpenCV'nin NMSBoxes'i)
// ----------------------------------------------------------------
static std::vector<Detection> run_nms(std::vector<Detection>& dets, float iou_thresh = 0.45f) {
    std::vector<Detection> result;
    if (dets.empty()) return result;

    // Sinif bazinda grupla, her sinif icin ayri NMS uygula
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

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument hatasi: " << e.what() << std::endl;
        return 1;
    }

    // ---------------- ONNX Runtime kurulumu (CPU) ----------------
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "benchmark_cpu");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4); // RK3588: gerekirse core sayisina gore ayarla
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // CPU icin ek bir provider eklemeye gerek yok; varsayilan CPUExecutionProvider kullanilir.

    Ort::Session session(env, args.model.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Girdi/cikti isimlerini modelden oku
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[]  = { input_name_ptr.get() };
    const char* output_names[] = { output_name_ptr.get() };

    // ---------------- Video ac ----------------
    cv::VideoCapture cap(args.input);
    if (!cap.isOpened()) {
        std::cerr << "Video acilamadi: " << args.input << std::endl;
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 25.0;
    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter writer;
    bool has_writer = !args.output.empty();
    if (has_writer) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); //avc1 yapmayı dene.
        writer.open(args.output, fourcc, fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "Cikti video dosyasi acilamadi: " << args.output << std::endl;
            has_writer = false;
        }
    }

    // ---------------- Benchmark degiskenleri ----------------
    double total_infer_time_ms = 0.0; // sadece Run() suresi
    long bench_frame_count = 0;
    long frame_idx = 0;

    const int input_size = args.imgsz;
    std::vector<float> input_tensor_values(1 * 3 * input_size * input_size);
    std::array<int64_t, 4> input_shape{1, 3, input_size, input_size};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;
        ++frame_idx;

        // ---------- Preprocess: letterbox + BGR->RGB + HWC->CHW + /255 ----------
        LetterboxInfo lb_info;
        cv::Mat letterboxed = letterbox(frame, input_size, lb_info);

        cv::Mat rgb;
        cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

        // HWC -> CHW
        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        size_t plane_size = static_cast<size_t>(input_size) * input_size;
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_tensor_values.data() + c * plane_size,
                        channels[c].data,
                        plane_size * sizeof(float));
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // ---------- Inference (sadece bu blok olculuyor) ----------
        auto t0 = std::chrono::high_resolution_clock::now();
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                           input_names, &input_tensor, 1,
                                           output_names, 1);
        auto t1 = std::chrono::high_resolution_clock::now();

        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_infer_time_ms += infer_ms;
        ++bench_frame_count;

        // ---------- Postprocess ----------
        // Beklenen cikti shape: (1, 84, 8400) -> [4 box coord + 80 class score, num_anchors]
        // (Ultralytics YOLOv8 ONNX export varsayilan formati; objectness skoru yok.)
        float* out_data = output_tensors[0].GetTensorMutableData<float>();
        auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        if (out_shape.size() != 3) {
            std::cerr << "Beklenmeyen cikti boyutu (rank=" << out_shape.size()
                      << "), model formatini kontrol edin." << std::endl;
            break;
        }

        int64_t dim1 = out_shape[1]; // 84
        int64_t dim2 = out_shape[2]; // 8400
        int num_classes = static_cast<int>(dim1 - 4);
        int num_anchors = static_cast<int>(dim2);

        std::vector<Detection> raw_dets;
        raw_dets.reserve(64);

        for (int a = 0; a < num_anchors; ++a) {
            // channel-major layout: out_data[channel * num_anchors + a]
            float cx = out_data[0 * num_anchors + a];
            float cy = out_data[1 * num_anchors + a];
            float w  = out_data[2 * num_anchors + a];
            float h  = out_data[3 * num_anchors + a];

            int best_class = -1;
            float best_score = 0.0f;
            for (int c = 0; c < num_classes; ++c) {
                // Sadece ilgilendigimiz siniflara bak (gereksiz hesaplamayi azaltmak icin)
                if (CLASS_MAP.find(c) == CLASS_MAP.end()) continue;
                float score = out_data[(4 + c) * num_anchors + a];
                if (score > best_score) {
                    best_score = score;
                    best_class = c;
                }
            }

            if (best_class == -1 || best_score < args.conf) continue;

            // letterbox koordinatlarindan orijinal frame koordinatlarina donusum
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

        std::vector<Detection> final_dets = run_nms(raw_dets);

        // ---------- Cizim ----------
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
            cv::imshow("Object Detection", frame);
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

        std::cout << "\n--- Benchmark Sonuclari ---" << std::endl;
        std::cout << "Frame Sayisi   : " << bench_frame_count << std::endl;
        std::cout.precision(2);
        std::cout << std::fixed;
        std::cout << "Ortalama Sure  : " << avg_ms << " ms" << std::endl;
        std::cout.precision(4);
        std::cout << "FPS            : " << avg_fps << std::endl;
    }

    return 0;
}
