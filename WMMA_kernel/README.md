### Prerequisites
```bash
sudo apt install nvidia-cuda-toolkit
```

### Step to compile and run it on NVIDIA A10G
```bash
ptxas wmma_kernel_demo.ptx -o wmma_kernel_demo.cubin -arch=sm_90
nvcc -arch=sm_90 wwma_demo_kernel_test-cu -o wmma_test -lcuda
./wmma_test
```