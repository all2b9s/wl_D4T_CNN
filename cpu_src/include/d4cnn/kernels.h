#pragma once
#include <cstddef>
#include <string>
#include <vector>
namespace d4cnn {
// Fused C32 epilogue: LayerNorm -> GELU -> residual, for primal + D-1 tangents.
// Inputs/outputs use [dual][pixel][32] planes. The epilogue either stores the
// result, or (last trunk layer) sinks it into pooled/canonical accumulators so
// the final feature map is never written back.
enum class EpiMode : int { raw=0, store=1, pool=2, canonical=3 };
struct Epi {
 EpiMode mode=EpiMode::store;
 const float* gamma=nullptr; const float* beta=nullptr; float residual=0;
 // Residual source: identity x[d*xstride+p*32+c], or projected skip[c]*x[d*xstride+p].
 const float* x=nullptr; int xstride=0; const float* skip=nullptr;
 // pool: pooled[d*32+c] += out*pool_w[p]
 float* pooled=nullptr; const float* pool_w=nullptr;
 // canonical: v=out/8; scalar[(d*ntok+token[p])*32+c] += v*token_scale;
 // e1/e2[(d*pixels+canon[p])*32+c] += sign*v. Both tables are indexed by the
 // orbit-frame pixel p (token[p] is the token of canonical pixel canon[p]).
 float* scalar=nullptr; float* e1=nullptr; float* e2=nullptr;
 const int* canon=nullptr; const int* token=nullptr;
 int ntok=0, pixels=0; float token_scale=1, sign1=1, sign2=1;
};
// Scalar ABI at every dispatch boundary: callers need no global ISA flags.
struct Kernels {
 std::string isa;
 int gemm_mr=0, gemm_nr=0, head_width=0;
 // C[M,N] = A[M,K] * B[K,N] (row-major, overwrite).
 void (*gemm)(const float* a,int lda,const float* b,int ldb,float* c,int ldc,int m,int n,int k)=nullptr;
 // Batched Winograd F(tile x tile,3x3), C32 -> C32, followed by the epilogue.
 void (*winograd)(int tile,int dual,const float* x,const float* weights,const float* bias,float* y,int h,const Epi& epi,float* scratch)=nullptr;
 // First layer C1 -> C32 (im2col + GEMM) followed by the epilogue.
 void (*first_layer)(int dual,const float* x,const float* w9,const float* bias,float* y,int h,const Epi& epi,float* scratch)=nullptr;
 // In-place epilogue on a stored convolution output (direct-convolution path).
 void (*epilogue_plane)(int dual,float* y,int pixels,const Epi& epi)=nullptr;
 // LayerNorm over 32 channels for rows of [dual][rows][32] (dstride floats per dual).
 void (*norm_rows)(int dual,float* x,int rows,int dstride,const float* gamma,const float* beta)=nullptr;
 // erf-GELU on count values of each dual component.
 void (*gelu)(int dual,float* x,int count,int dstride)=nullptr;
 // Dual softmax over cols for each of rows; [dual][rows][cols].
 void (*softmax_rows)(int dual,float* x,int rows,int cols,int dstride)=nullptr;
 // Multi-head attention context from qkv [dual][n][3*dim]; context [dual][n][dim].
 void (*attend)(int dual,const float* qkv,int n,int dim,int heads,float* context,float* scratch)=nullptr;
};
Kernels select_kernels(const std::string& isa);
// Seconds for a fixed GEMM workload with an explicit ISA (used by "auto").
double probe_isa_seconds(const std::string& isa);
// Scratch floats required by winograd/first_layer/attend for these sizes.
size_t kernel_scratch(int dual,int h,int tile,int tokens);
constexpr int winograd_group=8;
}
