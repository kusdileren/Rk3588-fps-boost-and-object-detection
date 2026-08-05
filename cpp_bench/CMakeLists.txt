cmake_minimum_required(VERSION 3.10)
project(yolo_benchmark)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(/usr/include/rknn)
link_directories(/usr/lib)

set(ONNXRUNTIME_VERSION "1.19.2")
set(ONNXRUNTIME_ROOT
    "$ENV{HOME}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}"
    CACHE PATH "ONNX Runtime kurulum dizini (include/ ve lib/ icerir)")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include")
  message(FATAL_ERROR
    "ONNX Runtime bulunamadi: ${ONNXRUNTIME_ROOT}\n"
    "Once scripts/setup_rk3588_board.sh calistir, ya da\n"
    "cmake .. -DONNXRUNTIME_ROOT=/senin/yolun ile gecir.")
endif()

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(benchmark_npu benchmark_npu.cpp)
target_link_libraries(benchmark_npu ${OpenCV_LIBS} rknnrt)

add_executable(benchmark_cpu benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu ${OpenCV_LIBS} onnxruntime)

set_target_properties(benchmark_cpu PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

message(STATUS "ONNXRUNTIME_ROOT = ${ONNXRUNTIME_ROOT}")