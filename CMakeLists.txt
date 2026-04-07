# CMakeLists.txt — PITNN DAB Converter C++ Inference Build
# ===========================================================
# Builds pitnn_inference.cpp using LibTorch.
#
# Prerequisites:
#   1. Download LibTorch from https://pytorch.org/get-started/locally/
#      Select: C++ / LibTorch / your CUDA version / ZIP
#      Extract to a folder, e.g. /opt/libtorch  or  C:\libtorch
#
#   2. Place these files in the same build output directory:
#        pitnn_scripted.pt
#        pitnn_mu.npy
#        pitnn_sigma.npy
#
# Build (Linux / Jetson):
#   mkdir build && cd build
#   cmake -DCMAKE_PREFIX_PATH=/opt/libtorch ..
#   make -j4
#   ./pitnn_inference
#
# Build (Windows):
#   mkdir build && cd build
#   cmake -DCMAKE_PREFIX_PATH=C:\libtorch -DCMAKE_BUILD_TYPE=Release ..
#   cmake --build . --config Release
#   Release\pitnn_inference.exe
#
# Copyright (c) 2026 Chukwuemeka Nzeadibe
# Mississippi State University — All Rights Reserved

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pitnn_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find LibTorch ──────────────────────────────────────────────────────────
find_package(Torch REQUIRED)

if(NOT Torch_FOUND)
  message(FATAL_ERROR
    "LibTorch not found.\n"
    "Download from https://pytorch.org/get-started/locally/ and pass:\n"
    "  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..")
endif()

message(STATUS "LibTorch version: ${TORCH_VERSION}")
message(STATUS "LibTorch include: ${TORCH_INCLUDE_DIRS}")

# ── Executable ─────────────────────────────────────────────────────────────
add_executable(pitnn_inference pitnn_inference.cpp)

target_include_directories(pitnn_inference PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(pitnn_inference PRIVATE ${TORCH_LIBRARIES})

# Enable optimisation flags
if(MSVC)
  target_compile_options(pitnn_inference PRIVATE /O2 /W3)
else()
  target_compile_options(pitnn_inference PRIVATE -O3 -Wall)
endif()

# ── Copy model files to build directory ────────────────────────────────────
# Automatically copies the three required files next to the executable
set(MODEL_FILES
  "${CMAKE_SOURCE_DIR}/pitnn_scripted.pt"
  "${CMAKE_SOURCE_DIR}/pitnn_mu.npy"
  "${CMAKE_SOURCE_DIR}/pitnn_sigma.npy"
)

foreach(FILE ${MODEL_FILES})
  if(EXISTS ${FILE})
    get_filename_component(FNAME ${FILE} NAME)
    configure_file(${FILE} ${CMAKE_BINARY_DIR}/${FNAME} COPYONLY)
    message(STATUS "Copying model file: ${FNAME}")
  else()
    message(WARNING
      "Model file not found: ${FILE}\n"
      "Copy pitnn_scripted.pt, pitnn_mu.npy, pitnn_sigma.npy "
      "to the source directory before building.")
  endif()
endforeach()

# ── Windows: copy LibTorch DLLs next to executable ─────────────────────────
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET pitnn_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${TORCH_DLLS}
      $<TARGET_FILE_DIR:pitnn_inference>
    COMMENT "Copying LibTorch DLLs..."
  )
endif()

# ── Print summary ──────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "PITNN Inference Build Summary")
message(STATUS "  Target  : pitnn_inference")
message(STATUS "  C++ std : C++17")
message(STATUS "  Torch   : ${TORCH_VERSION}")
message(STATUS "")
message(STATUS "After building, run from the build directory:")
if(MSVC)
  message(STATUS "  Release\\pitnn_inference.exe")
else()
  message(STATUS "  ./pitnn_inference")
endif()
