import os
os.environ["FLASHINFER_USE_CUDA_NORM"] = "1"

import torch
import flashinfer

torch.manual_seed(0)

M, D = 32, 4096
eps = 1e-6

x = torch.randn(M, D, device="cuda", dtype=torch.float16)
residual = torch.randn_like(x)
weight = torch.randn(D, device="cuda", dtype=torch.float16)

# 对应本次 CUDA kernel 的 FP32 中间结果语义
z = x.float() + residual.float()
residual_ref = z.to(x.dtype)

inv_rms = torch.rsqrt(z.square().mean(dim=-1, keepdim=True) + eps)
output_ref = (z * inv_rms * weight.float()).to(x.dtype)

# API 原地修改 input 和 residual
x_actual = x.clone()
residual_actual = residual.clone()

flashinfer.fused_add_rmsnorm(
    x_actual, residual_actual, weight,
    eps=eps,
    enable_pdl=False,
)

torch.testing.assert_close(
    residual_actual, residual_ref, rtol=1e-3, atol=1e-3
)
torch.testing.assert_close(
    x_actual, output_ref, rtol=1e-3, atol=1e-3
)
print('flashinfer.fused_add_rmsnorm test passed')