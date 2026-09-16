// Educational launcher: the repository's RMSNorm kernel is used unchanged.
#include <flashinfer/norm.cuh>
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;

int Threads(int d, int cap) {
  return flashinfer::ceil_div(std::min(cap, d / 8), 32) * 32;
}

void Run(TensorView out, TensorView input, TensorView weight, int64_t cap) {
  TVM_FFI_ICHECK(cap == 256 || cap == 1024);
  TVM_FFI_ICHECK(input.dtype() == dl_float16 && out.dtype() == dl_float16 &&
                weight.dtype() == dl_float16);
  TVM_FFI_ICHECK(input.ndim() == 2 && out.ndim() == 2 && weight.ndim() == 1);
  int m = input.size(0), d = input.size(1);
  TVM_FFI_ICHECK(m > 0 && d > 0 && d % 8 == 0);
  TVM_FFI_ICHECK(out.size(0) == m && out.size(1) == d && weight.size(0) == d);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(out);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, out);
  CHECK_DEVICE(input, weight);
  ffi::CUDADeviceGuard guard(input.device().device_id);
  int warps = Threads(d, cap) / 32;
  // <VEC_SIZE, T>
  flashinfer::norm::RMSNormKernel<8, half>
      <<<m, dim3(32, warps), warps * sizeof(float), get_stream(input.device())>>>(
          static_cast<half*>(input.data_ptr()), static_cast<half*>(weight.data_ptr()),
          static_cast<half*>(out.data_ptr()), d, input.stride(0), out.stride(0), 0.f, 1e-6f);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}

Array<int64_t> Resources(int64_t d, int64_t cap) {
  auto kernel = flashinfer::norm::RMSNormKernel<8, half>;
  cudaFuncAttributes attr;
  TVM_FFI_ICHECK(cudaFuncGetAttributes(&attr, kernel) == cudaSuccess);
  int threads = Threads(d, cap), blocks = 0;
  int smem = threads / 32 * sizeof(float);
  TVM_FFI_ICHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks, kernel, threads, smem) == cudaSuccess);
  return {threads, flashinfer::ceil_div(static_cast<int>(d), threads * 8), smem,
          attr.numRegs, static_cast<int64_t>(attr.localSizeBytes), blocks};
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(resources, Resources);
