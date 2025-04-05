#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Error checking macro
#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char* errStr; \
        cuGetErrorString(err, &errStr); \
        printf("CUDA Error: %s at %s:%d\n", errStr, __FILE__, __LINE__); \
        return -1; \
    } \
} while(0)

// Declaration of the PTX kernel
extern "C" __global__ void wmma_kernel(
    half* a,    // Input matrix A (fp16)
    half* b,    // Input matrix B (fp16)
    float* c,   // Input accumulator matrix C (fp32)
    float* d    // Output matrix D (fp32)
);

// Helper function to initialize matrices
void init_matrices(half* a, half* b, float* c, int m, int n, int k) {
    // Initialize A (m x k)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a[i * k + j] = __float2half(1.0f);  // All ones for simplicity
        }
    }
    
    // Initialize B (k x n)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = __float2half(1.0f);  // All ones for simplicity
        }
    }
    
    // Initialize C (m x n)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            c[i * n + j] = 0.0f;  // Zero initialization
        }
    }
}

// Helper function to verify results
void verify_results(float* d, int m, int n, int k) {
    bool passed = true;
    // For A and B filled with 1.0, each element in D should be k (dot product of k ones)
    float expected = (float)k;  // k ones multiplied together
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(d[i * n + j] - expected) > 1e-3) {
                printf("Mismatch at [%d,%d]: expected %.2f, got %.2f\n", 
                       i, j, expected, d[i * n + j]);
                passed = false;
            }
        }
    }
    printf("Test %s\n", passed ? "PASSED" : "FAILED");
    if (passed) {
        printf("All elements correctly computed as %.2f (dot product of %d ones)\n", 
               expected, k);
    }
}

int main() {
    const int M = 16;
    const int N = 16;
    const int K = 16;
    const int size_a = M * K;
    const int size_b = K * N;
    const int size_c = M * N;
    const int size_d = M * N;

    // Initialize CUDA driver API
    CHECK_CUDA(cuInit(0));

    // Get a CUDA device
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));

    // Create context
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // Load PTX module from file
    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "wmma_kernel_demo.ptx"));

    // Get kernel function
    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "wmma_kernel"));

    // Host memory
    half *h_a = new half[size_a];
    half *h_b = new half[size_b];
    float *h_c = new float[size_c];
    float *h_d = new float[size_d];

    // Initialize matrices with simpler values
    init_matrices(h_a, h_b, h_c, M, N, K);

    // Device memory
    CUdeviceptr d_a, d_b, d_c, d_d;
    CHECK_CUDA(cuMemAlloc(&d_a, size_a * sizeof(half)));
    CHECK_CUDA(cuMemAlloc(&d_b, size_b * sizeof(half)));
    CHECK_CUDA(cuMemAlloc(&d_c, size_c * sizeof(float)));
    CHECK_CUDA(cuMemAlloc(&d_d, size_d * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cuMemcpyHtoD(d_a, h_a, size_a * sizeof(half)));
    CHECK_CUDA(cuMemcpyHtoD(d_b, h_b, size_b * sizeof(half)));
    CHECK_CUDA(cuMemcpyHtoD(d_c, h_c, size_c * sizeof(float)));

    // Set up kernel parameters
    void* params[] = {
        &d_a,
        &d_b,
        &d_c,
        &d_d
    };

    // Launch kernel
    CHECK_CUDA(cuLaunchKernel(kernel,
        1, 1, 1,    // Grid dimensions
        32, 1, 1,   // Block dimensions (one warp)
        0, NULL,    // Shared memory and stream
        params,     // Parameters
        NULL        // Extra (unused)
    ));

    // Copy result back to host
    CHECK_CUDA(cuMemcpyDtoH(h_d, d_d, size_d * sizeof(float)));

    // Verify results with corrected expectations
    verify_results(h_d, M, N, K);

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_d;
    CHECK_CUDA(cuMemFree(d_a));
    CHECK_CUDA(cuMemFree(d_b));
    CHECK_CUDA(cuMemFree(d_c));
    CHECK_CUDA(cuMemFree(d_d));
    CHECK_CUDA(cuModuleUnload(module));
    CHECK_CUDA(cuCtxDestroy(context));

    return 0;
} 