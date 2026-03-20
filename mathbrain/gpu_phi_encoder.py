"""GPU CosineChaos φ encoder — CUDA warp shuffle (全寄存器)

D=32 = warp size. 每个 warp 处理 1 个 sample:
  - lane d 持有 E[d] 和 M[d] 在寄存器中
  - roll 通过 __shfl_sync 直接在寄存器间传递
  - 32 folds 零 global memory 访问
  
旧 Triton: 1.5ms (global memory roundtrip per fold)
新 CUDA:   ~0.1ms (全寄存器 + warp shuffle)
"""

import torch
import numpy as np
from torch.utils.cpp_extension import load_inline

# ── CUDA kernel source ──
_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// 全融合: Q → scale → P@Q → chaos → norm → φ̂
// 每个 warp (32 threads) 处理 1 个 sample
// thread lane_id 对应 d-维度 (0..31)

__global__ void phi_chaos_cuda_kernel(
    const float* __restrict__ Q,          // (n, N_in)
    const float* __restrict__ P,          // (D, N_in)
    const float* __restrict__ alpha_scale, // (N_in,)
    float* __restrict__ out,              // (n, D) — normalized φ
    float alpha,
    int n,
    int N_in,
    int D,
    int n_folds
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;   // which sample
    int lane_id = global_tid % 32;   // which d-dimension

    if (warp_id >= n) return;
    if (lane_id >= D) return;  // safety (D should == 32)

    // ── Step 1: Compute E[d] = sum_k P[d,k] * (Q[sample,k] * alpha_scale[k]) ──
    float e_val = 0.0f;
    for (int k = 0; k < N_in; k++) {
        float q_k = Q[warp_id * N_in + k];
        float as_k = alpha_scale[k];
        float p_dk = P[lane_id * N_in + k];
        e_val += p_dk * (q_k * as_k);
    }

    // ── Step 2: 32 folds chaos — ALL IN REGISTERS ──
    float m_val = 0.0f;
    for (int fold = 0; fold < n_folds; fold++) {
        // Roll: m_rolled[d] = m_old[(d-1+D) % D]
        // __shfl_sync reads from lane (lane_id+31)%32
        float m_rolled = __shfl_sync(0xFFFFFFFF, m_val, (lane_id + 31) % 32);
        m_val = cosf((m_rolled + e_val) * alpha);
    }

    // ── Step 3: L2 norm across warp (D=32 lanes) ──
    float sq = m_val * m_val;
    // Warp-level reduction for sum of squares
    for (int offset = 16; offset > 0; offset /= 2) {
        sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
    }
    // Broadcast norm from lane 0
    float norm = __shfl_sync(0xFFFFFFFF, sq, 0);
    norm = sqrtf(norm);
    if (norm < 1e-8f) norm = 1e-8f;

    // ── Step 4: Write normalized φ ──
    out[warp_id * D + lane_id] = m_val / norm;
}

torch::Tensor phi_chaos_forward(
    torch::Tensor Q,            // (n, N_in) float32, CUDA
    torch::Tensor P,            // (D, N_in) float32, CUDA
    torch::Tensor alpha_scale,  // (N_in,) float32, CUDA
    float alpha,
    int n_folds
) {
    int n = Q.size(0);
    int N_in = Q.size(1);
    int D = P.size(0);

    auto out = torch::empty({n, D}, Q.options());

    int threads_per_block = 256;  // 8 warps per block
    int total_threads = n * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

    phi_chaos_cuda_kernel<<<blocks, threads_per_block>>>(
        Q.data_ptr<float>(),
        P.data_ptr<float>(),
        alpha_scale.data_ptr<float>(),
        out.data_ptr<float>(),
        alpha,
        n,
        N_in,
        D,
        n_folds
    );

    return out;
}
"""

_cpp_src = r"""
torch::Tensor phi_chaos_forward(
    torch::Tensor Q,
    torch::Tensor P,
    torch::Tensor alpha_scale,
    float alpha,
    int n_folds
);
"""

# ── Lazy compile ──
_module = None

def _get_module():
    global _module
    if _module is None:
        import os
        if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
            # 自动检测当前 GPU 架构, 避免编译警告
            cap = torch.cuda.get_device_capability()
            os.environ['TORCH_CUDA_ARCH_LIST'] = f'{cap[0]}.{cap[1]}'
        _module = load_inline(
            name='phi_chaos_cuda',
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=['phi_chaos_forward'],
            verbose=False,
        )
    return _module


class GPUPhiEncoder:
    """GPU CosineChaos encoder — CUDA warp shuffle, 全寄存器"""

    def __init__(self, config, device='cuda'):
        from .phi_encoder import _butterfly_matrix

        self.N = config.N
        self.D = config.D_PHI
        self.n_folds = config.D_PHI if config.CHAOS_N_FOLDS < 0 else config.CHAOS_N_FOLDS
        self.alpha = float(config.CHAOS_ALPHA)
        self.device = torch.device(device)

        P_np = _butterfly_matrix(self.D, self.N, config.PHI_SIGMA)
        self.P = torch.from_numpy(P_np).to(self.device)  # (D, N)

        alpha_base = (np.pi / (2 * config.PHI_SIGMA * np.sqrt(self.D))) * 63.5
        alpha_scale = (config.inverse_weight * alpha_base).astype(np.float32)
        self.alpha_scale = torch.from_numpy(alpha_scale).to(self.device)

        # Force compile on init
        _get_module()

    @torch.no_grad()
    def encode(self, Q_values: torch.Tensor) -> torch.Tensor:
        """Q_values (n, N) → φ (n, D) — unnormalized"""
        n = Q_values.shape[0]
        if n == 0:
            return torch.zeros(0, self.D, device=self.device)

        mod = _get_module()
        # 全融合: Q → scale → P@Q → chaos (1 kernel)
        # 注意: 返回的是未归一化的, 但实际 kernel 里已经做了 norm
        return mod.phi_chaos_forward(
            Q_values.contiguous(),
            self.P,
            self.alpha_scale,
            self.alpha,
            self.n_folds,
        )

    @torch.no_grad()
    def encode_normalized(self, Q_values: torch.Tensor) -> torch.Tensor:
        """Q_values (n, N) → φ̂ (n, D), L2 normalized"""
        # CUDA kernel 已经做了 normalize, 直接返回
        return self.encode(Q_values)
