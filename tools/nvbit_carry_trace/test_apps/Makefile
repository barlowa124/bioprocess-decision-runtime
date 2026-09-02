# SPDX-License-Identifier: MIT

NVCC ?= nvcc

all: carry_test lea_test

carry_test: carry_test.cu
	$(NVCC) -O3 -arch=sm_89 -lineinfo $< -o $@

lea_test: lea_test.cu
	$(NVCC) -O3 -arch=sm_89 -lineinfo $< -o $@

clean:
	rm -f carry_test lea_test
