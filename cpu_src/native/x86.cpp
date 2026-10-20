#include "d4cnn/x86.h"
#include "d4cnn/convolution.h"
#include <cmath>
#include <stdexcept>
#if defined(__x86_64__)
#include <immintrin.h>
namespace d4cnn {
namespace avx2_detail {
#define TARGET __attribute__((target("avx2,fma")))
#define WIDTH 8
#define V __m256
#define OP(name) _mm256_##name
#define EXP_POWER(n) OP(castsi256_ps)(OP(slli_epi32)(OP(add_epi32)(OP(cvtps_epi32)(n),OP(set1_epi32)(127)),23))
#define ROUND(n) OP(round_ps)(n,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC)
#define NEGATIVE(a,b) OP(blendv_ps)(a,b,OP(cmp_ps)(v,OP(setzero_ps)(),_CMP_LT_OQ))
#define CLAMP_EXP(x,result) OP(and_ps)(result,OP(cmp_ps)(x,OP(set1_ps)(-87.f),_CMP_GE_OQ))
#include "x86_impl.inc"
#undef TARGET
#undef WIDTH
#undef V
#undef OP
#undef EXP_POWER
#undef ROUND
#undef NEGATIVE
#undef CLAMP_EXP
}
namespace avx512_detail {
#define TARGET __attribute__((target("avx512f,fma")))
#define WIDTH 16
#define V __m512
#define OP(name) _mm512_##name
#define EXP_POWER(n) OP(castsi512_ps)(OP(slli_epi32)(OP(add_epi32)(OP(cvtps_epi32)(n),OP(set1_epi32)(127)),23))
#define ROUND(n) OP(roundscale_ps)(n,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC)
#define NEGATIVE(a,b) OP(mask_blend_ps)(OP(cmp_ps_mask)(v,OP(setzero_ps)(),_CMP_LT_OQ),a,b)
#define CLAMP_EXP(x,result) OP(maskz_mov_ps)(OP(cmp_ps_mask)(x,OP(set1_ps)(-87.f),_CMP_GE_OQ),result)
#include "x86_impl.inc"
#undef TARGET
#undef WIDTH
#undef V
#undef OP
#undef EXP_POWER
#undef ROUND
#undef NEGATIVE
#undef CLAMP_EXP
}
X86Kernels select_x86_kernels(const std::string& isa) {
 const auto resolved=resolve_isa(isa);
 if(resolved=="avx2")return {avx2_detail::epilogue,avx2_detail::winograd<2>,avx2_detail::winograd<4>};
 if(resolved=="avx512")return {avx512_detail::epilogue,avx512_detail::winograd<2>,avx512_detail::winograd<4>};
 return {};
}
}
#else
namespace d4cnn { X86Kernels select_x86_kernels(const std::string&) {return {};} }
#endif
