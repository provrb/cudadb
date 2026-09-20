cmake_minimum_required(VERSION 4.0.0)

project(CudaDB LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
set_target_properties(PROPERTIES CUDA_ARCHITECTURES "sm_11")

file(GLOB SOURCES "src/*.cpp" "src/*.cu")
file(GLOB HEADERS "src/hdr/*.h")

add_executable(main.exe ${SOURCES} ${HEADERS})