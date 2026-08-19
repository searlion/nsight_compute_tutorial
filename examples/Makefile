# Standalone build of the hand-written CUDA GEMM.
#
# The toolkit location and the GPU arch are discovered at build time by
# cuda_env.py, so there is nothing machine-specific to edit here:
#
#   make            # build ./gemm_cuda
#   make run        # build, then verify + benchmark
#   make ARCH=sm_90 # override the detected arch
#
# Defaults to the repo's own .venv if it exists, so `make` works whether or
# not you have the venv activated.

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
VENV_PY      := $(MAKEFILE_DIR)../.venv/bin/python
PYTHON       ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)

CUDA_HOME ?= $(shell $(PYTHON) $(MAKEFILE_DIR)cuda_env.py home)
CUDA_LIB  ?= $(shell $(PYTHON) $(MAKEFILE_DIR)cuda_env.py lib)
ARCH      ?= $(shell $(PYTHON) $(MAKEFILE_DIR)cuda_env.py arch)

NVCC      := $(CUDA_HOME)/bin/nvcc
NVCCFLAGS := -O3 -arch=$(ARCH) -std=c++17

.PHONY: all run clean

all: gemm_cuda

gemm_cuda: gemm_cuda.cu
	$(NVCC) $(NVCCFLAGS) -L $(CUDA_LIB) $< -o $@

run: gemm_cuda
	./gemm_cuda

clean:
	rm -f gemm_cuda
	rm -rf build __pycache__
