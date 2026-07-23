// N0 toolchain spike (nccl-ffi-plan_v1.txt): trivial XLA FFI handler on the CUDA
// stream (y = 2x), built against jax.ffi.include_dir() headers. Proves the
// build/register/call mechanics on aarch64 + jaxlib 0.10.2 BEFORE any NCCL code.
#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

__global__ void Scale2Kernel(const float* x, float* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = 2.0f * x[i];
}

static ffi::Error Spike0Impl(cudaStream_t stream, ffi::Buffer<ffi::F32> x,
                             ffi::ResultBuffer<ffi::F32> y) {
  long n = (long)x.element_count();
  int block = 256;
  long grid = (n + block - 1) / block;
  Scale2Kernel<<<grid, block, 0, stream>>>(x.typed_data(), y->typed_data(), n);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return ffi::Error::Internal(cudaGetErrorString(err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(Spike0, Spike0Impl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()   // x
                                  .Ret<ffi::Buffer<ffi::F32>>()); // y
