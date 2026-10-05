# =====================================================================
# Makefile — cuda-sgemm 薄封装（跨 Linux / Windows-mingw32-make）
#   make / make build     构建全部目标
#   make test             正确性回归（全 kernel x 全测试矩阵）
#   make bench            benchmark（KERNEL=all M=4096 N=4096 K=4096 可覆盖）
#   make ptxas_log        重编译并归档 -Xptxas -v 到 build.log
#   make clean
# 实际构建系统为 CMake（见 CMakeLists.txt）。
# =====================================================================

BUILD_DIR := build
# 多配置生成器（VS）产物在 build/Release，单配置（Ninja/Make）在 build/
BIN_DIR := $(if $(wildcard $(BUILD_DIR)/Release),$(BUILD_DIR)/Release,$(BUILD_DIR))
BENCH    := $(BIN_DIR)/sgemm_bench
TEST     := $(BIN_DIR)/sgemm_test

KERNEL ?= all
M ?= 4096
N ?= 4096
K ?= 4096
ITERS ?= 100
WARMUP ?= 20

.PHONY: all build test bench ptxas_log clean

all: build

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
	cmake --build $(BUILD_DIR) --config Release -j

test: build
	$(TEST) $(if $(filter --verbose,$(VERBOSE)),--verbose,)

bench: build
	$(BENCH) --kernel $(KERNEL) --m $(M) --n $(N) --k $(K) \
	         --warmup $(WARMUP) --iters $(ITERS) --csv

ptxas_log:
	cmake --build $(BUILD_DIR) --config Release --clean-first -- VERBOSE=1 > build.log 2>&1
	@grep -E "Function properties|registers|spill|Used" build.log || true

clean:
	rm -rf $(BUILD_DIR)
