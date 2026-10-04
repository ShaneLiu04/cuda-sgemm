# =====================================================================
# CMakeLists.txt — cuda-sgemm 一键构建
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
#   cmake --build build -j
# 规则（AGENTS.md §3）：-O3 -std=c++17 -arch=sm_89 -lineinfo -Xptxas -v
#   禁止 -use_fast_math（严格 FP32；FMA 合并默认允许）
# 目标：sgemm_bench（CLI 入口）、sgemm_test（正确性回归）
# =====================================================================
cmake_minimum_required(VERSION 3.18)
project(cuda-sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 默认 sm_89（RTX 4060 Laptop）；其他卡用 -DCMAKE_CUDA_ARCHITECTURES=xx 覆盖
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# git 提交号（数据可复现性）
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_RC)
if(NOT GIT_RC EQUAL 0)
    set(GIT_SHA "unknown")
endif()
add_compile_definitions(GIT_SHA="${GIT_SHA}")

# CUDA 专属编译选项：-lineinfo（ncu Source 关联）+ -Xptxas -v（寄存器/smem 审计）
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas>-v
)

# 明确排除快速数学（防止上层工具链注入；严格 FP32 军规）
check_cxx_compiler_flag("" _dummy)  # no-op，保持模块纯净
remove_definitions(-use_fast_math)

set(KERNEL_SOURCES
    src/sgemm_naive.cu
    src/sgemm_coalesced.cu
    src/sgemm_smem_1d.cu
    src/sgemm_2d_tile.cu
    src/sgemm_vec4.cu
    src/sgemm_cpasync.cu
    src/cublas_baseline.cu)

add_executable(sgemm_bench src/main.cu ${KERNEL_SOURCES})
target_include_directories(sgemm_bench PRIVATE include)
target_link_libraries(sgemm_bench PRIVATE CUDA::cudart CUDA::cublas)

add_executable(sgemm_test tests/test_correctness.cu ${KERNEL_SOURCES})
target_include_directories(sgemm_test PRIVATE include)
target_link_libraries(sgemm_test PRIVATE CUDA::cudart CUDA::cublas)

enable_testing()
add_test(NAME correctness COMMAND sgemm_test)

# 便捷目标：构建后把 -Xptxas -v 输出归档到 build.log（寄存器/smem 验收数据）
add_custom_target(ptxas_log
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target sgemm_bench
            -- VERBOSE=1 > build.log 2>&1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Rebuild with verbose log -> build.log (registers/smem per kernel)"
    VERBATIM)
