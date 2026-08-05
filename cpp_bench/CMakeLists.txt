cmake_minimum_required(VERSION 3.10)
project(yolo_benchmark)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(/usr/include/rknn)
link_directories(/usr/lib)

include_directories(/home/alpagut/onnxruntime-linux-aarch64-1.19.2/include)
link_directories(/home/alpagut/onnxruntime-linux-aarch64-1.19.2/lib)

add_executable(benchmark_npu benchmark_npu.cpp)
target_link_libraries(benchmark_npu ${OpenCV_LIBS} rknnrt)

add_executable(benchmark_cpu benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu ${OpenCV_LIBS} onnxruntime)