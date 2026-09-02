// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

__global__ void carry_kernel(const unsigned long long* first,
                             const unsigned long long* second,
                             const unsigned long long* third,
                             unsigned long long* output, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = first[index] + second[index] + third[index];
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
    unsigned long long first[count];
    unsigned long long second[count];
    unsigned long long third[count];
    unsigned long long expected[count];
    const unsigned long long vectors[4][3] = {
        {1ULL, 2ULL, 3ULL},
        {0xffffffffULL, 1ULL, 0ULL},
        {0xffffffffffffffffULL, 1ULL, 1ULL},
        {0xffffffffULL, 0xffffffffULL, 0xffffffffULL},
    };
    for (int index = 0; index < count; index++) {
        first[index] = vectors[index % 4][0];
        second[index] = vectors[index % 4][1];
        third[index] = vectors[index % 4][2];
        expected[index] = first[index] + second[index] + third[index];
    }

    unsigned long long *device_first, *device_second, *device_third, *device_output;
    check(cudaMalloc(&device_first, sizeof(first)), "cudaMalloc first");
    check(cudaMalloc(&device_second, sizeof(second)), "cudaMalloc second");
    check(cudaMalloc(&device_third, sizeof(third)), "cudaMalloc third");
    check(cudaMalloc(&device_output, sizeof(expected)), "cudaMalloc output");
    check(cudaMemcpy(device_first, first, sizeof(first), cudaMemcpyHostToDevice),
          "cudaMemcpy first");
    check(cudaMemcpy(device_second, second, sizeof(second), cudaMemcpyHostToDevice),
          "cudaMemcpy second");
    check(cudaMemcpy(device_third, third, sizeof(third), cudaMemcpyHostToDevice),
          "cudaMemcpy third");

    unsigned long long outputs[2][count];
    for (int launch = 0; launch < 2; launch++) {
        carry_kernel<<<2, 32>>>(device_first, device_second, device_third,
                                device_output, count);
        check(cudaGetLastError(), "carry_kernel launch");
        check(cudaDeviceSynchronize(), "carry_kernel synchronize");
        check(cudaMemcpy(outputs[launch], device_output, sizeof(expected),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy output");
    }

    cudaFree(device_first);
    cudaFree(device_second);
    cudaFree(device_third);
    cudaFree(device_output);
    check(cudaDeviceReset(), "cudaDeviceReset");

    for (int launch = 0; launch < 2; launch++) {
        for (int index = 0; index < count; index++) {
            std::printf(
                "launch %d case %d output=0x%016llx expected=0x%016llx\n",
                launch, index, outputs[launch][index], expected[index]);
            if (outputs[launch][index] != expected[index]) {
                return 2;
            }
        }
    }

    return 0;
}
