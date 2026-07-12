# /// script
# requires-python = ">=3.10"
# dependencies = ["vllm>=0.22,<0.24", "cupy-cuda12x"]
# ///
"""mikasa kernel R&D v0: hand-written int4-g128 GEMV vs Marlin at M=1 on A10G (sm_86).

Protocol (mistake-proofed):
  1. Logical weights are IDENTICAL for both kernels: marlin_quantize() produces (w_ref, q_w, s);
     we recover integer q from w_ref/s and pack our own layout. Correctness is gated vs the
     fp32 reference x @ w_ref BEFORE any timing is reported.
  2. Timing is CUDA-graph-captured only (eager launch overhead lies at these sizes).
  3. Our kernel is deterministic: fixed-order strided accumulation + warp-shuffle tree
     reduction, no atomics (greedy-identity requirement).
"""
import json, torch
import cupy as cp
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize, MarlinWorkspace
import vllm.model_executor.layers.quantization.utils.marlin_utils as mu
from vllm.scalar_type import scalar_types

dev = "cuda"
GS = 128
torch.manual_seed(0)
print(f"GPU: {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}", flush=True)

# ---------- peak copy bandwidth ----------
N0 = 1 << 27
a = torch.randn(N0, device=dev, dtype=torch.float16); b = torch.empty_like(a)
for _ in range(10): b.copy_(a)
torch.cuda.synchronize()
s0 = torch.cuda.Event(enable_timing=True); e0 = torch.cuda.Event(enable_timing=True)
s0.record()
for _ in range(50): b.copy_(a)
e0.record(); torch.cuda.synchronize()
PEAK = 2 * a.numel() * 2 / (s0.elapsed_time(e0) / 50 / 1e3) / 1e9
print(f"copy peak: {PEAK:.0f} GB/s", flush=True)

# ---------- our kernel (CUDA C via NVRTC) ----------
KSRC = r"""
#include <cuda_fp16.h>
extern "C" __global__ void gemv_int4_g128(
    const unsigned char* __restrict__ Wq,   // [N, K/2] row-major packed nibbles (low = even k)
    const __half*        __restrict__ S,    // [N, K/128]
    const __half*        __restrict__ X,    // [K]
    __half*              __restrict__ Y,    // [N]
    const int N, const int K)
{
    const int n = blockIdx.x;
    if (n >= N) return;
    const unsigned char* wrow = Wq + (long long)n * (K >> 1);
    const __half* srow = S + (long long)n * (K >> 7);
    const int tid = threadIdx.x;
    const int chunks = K >> 5;               // 32 weights per 16-byte chunk
    float acc = 0.f;
    for (int c = tid; c < chunks; c += blockDim.x) {
        const uint4 pv = __ldg(((const uint4*)wrow) + c);
        const int k0 = c << 5;
        const float s = __half2float(srow[k0 >> 7]);  // 32 | 128 -> no group straddle
        const __half* xp = X + k0;
        const unsigned char* pb = (const unsigned char*)&pv;
        float local = 0.f;
        #pragma unroll
        for (int bt = 0; bt < 16; ++bt) {
            const int byte = pb[bt];
            local += (float)((byte & 0xF) - 8) * __half2float(xp[2 * bt]);
            local += (float)((byte >> 4)  - 8) * __half2float(xp[2 * bt + 1]);
        }
        acc += s * local;
    }
    __shared__ float smem[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if ((tid & 31) == 0) smem[tid >> 5] = acc;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x >> 5)) ? smem[tid] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (tid == 0) Y[n] = __float2half(v);
    }
}
"""
mod = cp.RawModule(code=KSRC, options=("-arch=sm_86", "-std=c++17"), name_expressions=None)
kern = mod.get_function("gemv_int4_g128")
print("NVRTC compile: OK", flush=True)

def cpa(t):  # zero-copy torch->cupy
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(t))

def cap_graph(fn):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(5): fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return g

def bench_graph(g, iters=400, warm=100):
    for _ in range(warm): g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us

MIN_N = getattr(mu, "GPTQ_MARLIN_MIN_THREAD_N", 64)
MAXP  = getattr(mu, "GPTQ_MARLIN_MAX_PARALLEL", 16)
results = []
for (K, N) in [(2560, 10240), (10240, 2560), (2048, 16384)]:
    w_kn = (torch.randn(K, N, device=dev) / (K ** 0.5)).to(torch.float16)
    res = marlin_quantize(w_kn, scalar_types.uint4b8, GS, False)
    w_ref, q_w, s = res[0], res[1], res[2]
    g_idx, sort_idx = res[3], res[4]
    # derive our integer grid from marlin's (w_ref, s); small mismatch is FINE for a speed A/B
    # (identical bytes/layout; exact identity only matters at integration, where the checkpoint
    # stores (q,s) explicitly). Each kernel is gated against ITS OWN logical weights.
    s_exp = s.repeat_interleave(GS, dim=0).float()                      # [K,N]
    q_int = torch.round(w_ref.float() / s_exp).clamp(-8, 7)
    w_mine = q_int * s_exp                                               # our logical weights (fp32)
    grid_mismatch = (w_mine - w_ref.float()).abs().max().item()
    print(f"[{K}x{N}] grid mismatch vs marlin w_ref (info only): {grid_mismatch:.4f}", flush=True)
    q_nk = (q_int + 8).to(torch.uint8).t().contiguous()                  # [N,K] 0..15
    packed = (q_nk[:, 0::2] | (q_nk[:, 1::2] << 4)).contiguous()         # [N,K/2]
    s_nk = s.t().contiguous().to(torch.float16)                          # [N,K/128]

    x = torch.randn(1, K, device=dev, dtype=torch.float16)
    y_ref = (x.float() @ w_ref.float())                                  # marlin's fp32 reference
    y_ref_mine = (x.float() @ w_mine)                                    # OUR fp32 reference
    # marlin
    ws = MarlinWorkspace(N, MIN_N, MAXP)
    zp = torch.empty(0, dtype=torch.int, device=dev)
    om = torch.empty(1, N, device=dev, dtype=torch.float16)
    def marlin():
        om.copy_(mu.apply_gptq_marlin_linear(x, q_w, s, zp, g_idx, sort_idx, ws.scratch,
                                             scalar_types.uint4b8, N, K, True))
    # ours
    outs = {}
    for T in (128, 256):
        yo = torch.empty(N, device=dev, dtype=torch.float16)
        Wc, Sc, Xc, Yc = cpa(packed), cpa(s_nk), cpa(x.view(-1)), cpa(yo)
        def ours(T=T, Yc=Yc, Wc=Wc, Sc=Sc, Xc=Xc):
            with cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
                kern((N,), (T,), (Wc, Sc, Xc, Yc, N, K))
        outs[T] = (ours, yo)

    # ---- correctness gate (BEFORE timing) ----
    marlin(); torch.cuda.synchronize()
    rel_m = ((om.float() - y_ref).abs().mean() / y_ref.abs().mean()).item()
    rels = {}
    for T, (fn, yo) in outs.items():
        fn(); torch.cuda.synchronize()
        rels[T] = ((yo.float() - y_ref_mine.view(-1)).abs().mean() / y_ref_mine.abs().mean()).item()
    print(f"[{K}x{N}] correctness rel: marlin={rel_m:.2e} ours128={rels[128]:.2e} ours256={rels[256]:.2e}", flush=True)
    ok = all(r < max(2e-3, 3 * rel_m) for r in rels.values())
    if not ok:
        print(f"[{K}x{N}] CORRECTNESS FAIL — skipping timing for this shape", flush=True)
        results.append(dict(shape=f"{K}x{N}", status="correctness-fail", rels=rels, rel_marlin=rel_m))
        continue

    # ---- graphed timing ----
    tm = bench_graph(cap_graph(marlin))
    byts = K * N / 2 + (K // GS) * N * 2
    row = dict(shape=f"{K}x{N}", status="ok", rel_marlin=rel_m,
               marlin_us=round(tm, 2), marlin_roof=round(100 * byts / (tm / 1e6) / 1e9 / PEAK, 1))
    for T, (fn, yo) in outs.items():
        g = cap_graph(fn)
        yo.zero_(); g.replay(); torch.cuda.synchronize()
        assert yo.abs().sum().item() > 0, f"EMPTY GRAPH for ours{T} — kernel not captured"
        tt = bench_graph(g)
        roofpct = 100 * byts / (tt / 1e6) / 1e9 / PEAK
        assert roofpct <= 105, f"PHYSICS VIOLATION ours{T}: {roofpct:.0f}% of peak — timing invalid"
        row[f"ours{T}_us"] = round(tt, 2)
        row[f"ours{T}_roof"] = round(roofpct, 1)
        row[f"ours{T}_rel_err"] = rels[T]
        row[f"ours{T}_vs_marlin"] = round(tt / tm, 3)
    print("RESULT " + json.dumps(row), flush=True)
    results.append(row)

print("FINAL " + json.dumps({"peak_gbps": round(PEAK, 1), "rows": results}), flush=True)
