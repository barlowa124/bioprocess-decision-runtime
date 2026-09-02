// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void lea_kernel(const uint64_t* first, const uint64_t* second,
                           uint64_t* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = first[index] + (second[index] << 1);
    }
}

static void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(1);
    }
}

int main() {
    constexpr int count = 64;
    uint64_t first[count];
    uint64_t second[count];
    uint64_t expected[count];
    for (int index = 0; index < count; index++) {
        first[index] = index % 2 ? 0xffffffffffffffffULL : uint64_t(index);
        second[index] = uint64_t(index + 1);
        expected[index] = first[index] + (second[index] << 1);
    }
    uint64_t *device_first, *device_second, *device_output;
    check(cudaMalloc(&device_first, sizeof(first)), "cudaMalloc first");
    check(cudaMalloc(&device_second, sizeof(second)), "cudaMalloc second");
    check(cudaMalloc(&device_output, sizeof(expected)), "cudaMalloc output");
    check(cudaMemcpy(device_first, first, sizeof(first), cudaMemcpyHostToDevice),
          "cudaMemcpy first");
    check(cudaMemcpy(device_second, second, sizeof(second), cudaMemcpyHostToDevice),
          "cudaMemcpy second");
    uint64_t outputs[2][count];
    for (int launch = 0; launch < 2; launch++) {
        lea_kernel<<<2, 32>>>(device_first, device_second, device_output, count);
        check(cudaGetLastError(), "lea_kernel launch");
        check(cudaDeviceSynchronize(), "lea_kernel synchronize");
        check(cudaMemcpy(outputs[launch], device_output, sizeof(expected),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy output");
    }
    cudaFree(device_first);
    cudaFree(device_second);
    cudaFree(device_output);
    check(cudaDeviceReset(), "cudaDeviceReset");
    for (int launch = 0; launch < 2; launch++) {
        for (int index = 0; index < count; index++) {
            if (outputs[launch][index] != expected[index]) {
                return 2;
            }
        }
    }
    std::printf("lea_outputs_restored=true\n");
    return 0;
}
