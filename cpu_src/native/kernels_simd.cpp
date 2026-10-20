#include "d4cnn/kernels.h"
#include "d4cnn/convolution.h"
#include "d4cnn/profile.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif
namespace d4cnn {
#if defined(__aarch64__)
namespace neon_detail {
#define D4_TARGET
using V=float32x4_t;constexpr int W=4;using H=float32x4_t;constexpr int HW=4;
// GEMM: transposed-A lane kernel (12x8), see D4_LANE_GEMM in simd_impl.inc.
#define D4_LANE_GEMM 1
constexpr int MRMAX=12,NR=8;
static inline V vzero(){return vdupq_n_f32(0.f);}
static inline V vset(float x){return vdupq_n_f32(x);}
static inline V vload(const float* p){return vld1q_f32(p);}
static inline void vstore(float* p,V v){vst1q_f32(p,v);}
static inline V vadd(V a,V b){return vaddq_f32(a,b);}
static inline V vsub(V a,V b){return vsubq_f32(a,b);}
static inline V vmul(V a,V b){return vmulq_f32(a,b);}
static inline V vmuls(V a,float s){return vmulq_n_f32(a,s);}
static inline V vfma(V a,V b,V c){return vfmaq_f32(c,a,b);}
static inline V vfnma(V a,V b,V c){return vfmsq_f32(c,a,b);}
static inline V vfmas(V a,float s,V c){return vfmaq_n_f32(c,a,s);}
static inline V vmax(V a,V b){return vmaxq_f32(a,b);}
static inline V vabs(V a){return vabsq_f32(a);}
static inline V selectneg(V v,V pos,V neg){return vbslq_f32(vcltq_f32(v,vdupq_n_f32(0.f)),neg,pos);}
static inline float hsum(V a){return vaddvq_f32(a);}
static inline float hmax(V a){return vmaxvq_f32(a);}
static inline V recip(V x){V e=vrecpeq_f32(x);e=vmulq_f32(vrecpsq_f32(x,e),e);return vmulq_f32(vrecpsq_f32(x,e),e);}
// Restricted-domain exp for x <= 0; inputs below -87 saturate at exp(-87)~1.6e-38.
static inline V exp_negative(V x){
 const V z=vmaxq_f32(x,vdupq_n_f32(-87.f));
 const V n=vrndnq_f32(vmulq_n_f32(z,1.4426950408889634f));
 V r=vfmsq_n_f32(z,n,0.693145751953125f);r=vfmsq_n_f32(r,n,1.428606765330187e-6f);
 // Degree-5 minimax of exp(r) on [-ln2/2,ln2/2]: relative error 7.5e-8.
 V p=vdupq_n_f32(0.008297686563183427f);
 p=vfmaq_f32(vdupq_n_f32(0.04191538491131187f),p,r);p=vfmaq_f32(vdupq_n_f32(0.16667574268138968f),p,r);
 p=vfmaq_f32(vdupq_n_f32(0.4999889481644092f),p,r);p=vfmaq_f32(vdupq_n_f32(0.9999996921226256f),p,r);
 p=vfmaq_f32(vdupq_n_f32(1.0000000716597748f),p,r);
 const V power=vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n),vdupq_n_s32(127)),23));
 return vmulq_f32(p,power);
}
static inline H hzero(){return vzero();}
static inline H hload(const float* p){return vload(p);}
static inline void hstore(float* p,H v){vstore(p,v);}
static inline H hfma(H a,H b,H c){return vfma(a,b,c);}
static inline H hfmas(H a,float s,H c){return vfmas(a,s,c);}
static inline float hsum_h(H a){return hsum(a);}
#include "simd_impl.inc"
#undef D4_LANE_GEMM
#undef D4_TARGET
}
#elif defined(__x86_64__)
namespace avx2_detail {
#define D4_TARGET __attribute__((target("avx2,fma")))
using V=__m256;constexpr int W=8;using H=__m256;constexpr int HW=8;
// 6x16 register block: 12 accumulators + 2 weight vectors + 1 broadcast.
constexpr int MRMAX=6,NR=16;
D4_TARGET static inline V vzero(){return _mm256_setzero_ps();}
D4_TARGET static inline V vset(float x){return _mm256_set1_ps(x);}
D4_TARGET static inline V vload(const float* p){return _mm256_loadu_ps(p);}
D4_TARGET static inline void vstore(float* p,V v){_mm256_storeu_ps(p,v);}
D4_TARGET static inline V vadd(V a,V b){return _mm256_add_ps(a,b);}
D4_TARGET static inline V vsub(V a,V b){return _mm256_sub_ps(a,b);}
D4_TARGET static inline V vmul(V a,V b){return _mm256_mul_ps(a,b);}
D4_TARGET static inline V vmuls(V a,float s){return _mm256_mul_ps(a,_mm256_set1_ps(s));}
D4_TARGET static inline V vfma(V a,V b,V c){return _mm256_fmadd_ps(a,b,c);}
D4_TARGET static inline V vfnma(V a,V b,V c){return _mm256_fnmadd_ps(a,b,c);}
D4_TARGET static inline V vfmas(V a,float s,V c){return _mm256_fmadd_ps(a,_mm256_set1_ps(s),c);}
D4_TARGET static inline V vmax(V a,V b){return _mm256_max_ps(a,b);}
D4_TARGET static inline V vabs(V a){return _mm256_andnot_ps(_mm256_set1_ps(-0.f),a);}
D4_TARGET static inline V selectneg(V v,V pos,V neg){return _mm256_blendv_ps(pos,neg,_mm256_cmp_ps(v,_mm256_setzero_ps(),_CMP_LT_OQ));}
D4_TARGET static inline float hsum(V a){
 __m128 x=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
 x=_mm_add_ps(x,_mm_movehl_ps(x,x));x=_mm_add_ss(x,_mm_shuffle_ps(x,x,1));return _mm_cvtss_f32(x);
}
D4_TARGET static inline float hmax(V a){
 __m128 x=_mm_max_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
 x=_mm_max_ps(x,_mm_movehl_ps(x,x));x=_mm_max_ss(x,_mm_shuffle_ps(x,x,1));return _mm_cvtss_f32(x);
}
// rcp_ps (12 bits) + one Newton step.
D4_TARGET static inline V recip(V x){V e=_mm256_rcp_ps(x);return _mm256_mul_ps(e,_mm256_fnmadd_ps(x,e,_mm256_set1_ps(2.f)));}
D4_TARGET static inline V exp_negative(V x){
 const V z=_mm256_max_ps(x,_mm256_set1_ps(-87.f));
 const V n=_mm256_round_ps(_mm256_mul_ps(z,_mm256_set1_ps(1.4426950408889634f)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
 V r=_mm256_fnmadd_ps(n,_mm256_set1_ps(.693145751953125f),z);r=_mm256_fnmadd_ps(n,_mm256_set1_ps(1.428606765330187e-6f),r);
 // Degree-5 minimax of exp(r) on [-ln2/2,ln2/2]: relative error 7.5e-8.
 V p=_mm256_set1_ps(0.008297686563183427f);
 p=vfma(p,r,vset(0.04191538491131187f));p=vfma(p,r,vset(0.16667574268138968f));
 p=vfma(p,r,vset(0.4999889481644092f));p=vfma(p,r,vset(0.9999996921226256f));
 p=vfma(p,r,vset(1.0000000716597748f));
 const V power=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n),_mm256_set1_epi32(127)),23));
 (void)x;return _mm256_mul_ps(p,power);
}
D4_TARGET static inline H hzero(){return vzero();}
D4_TARGET static inline H hload(const float* p){return vload(p);}
D4_TARGET static inline void hstore(float* p,H v){vstore(p,v);}
D4_TARGET static inline H hfma(H a,H b,H c){return vfma(a,b,c);}
D4_TARGET static inline H hfmas(H a,float s,H c){return vfmas(a,s,c);}
D4_TARGET static inline float hsum_h(H a){return hsum(a);}
#include "simd_impl.inc"
#undef D4_TARGET
}
namespace avx512_detail {
#define D4_TARGET __attribute__((target("avx512f,avx2,fma")))
using V=__m512;constexpr int W=16;using H=__m256;constexpr int HW=8;
// 12x32 register block: 24 accumulators + 2 weight vectors; broadcasts fold
// into the FMA memory operand.
constexpr int MRMAX=12,NR=32;
D4_TARGET static inline V vzero(){return _mm512_setzero_ps();}
D4_TARGET static inline V vset(float x){return _mm512_set1_ps(x);}
D4_TARGET static inline V vload(const float* p){return _mm512_loadu_ps(p);}
D4_TARGET static inline void vstore(float* p,V v){_mm512_storeu_ps(p,v);}
D4_TARGET static inline V vadd(V a,V b){return _mm512_add_ps(a,b);}
D4_TARGET static inline V vsub(V a,V b){return _mm512_sub_ps(a,b);}
D4_TARGET static inline V vmul(V a,V b){return _mm512_mul_ps(a,b);}
D4_TARGET static inline V vmuls(V a,float s){return _mm512_mul_ps(a,_mm512_set1_ps(s));}
D4_TARGET static inline V vfma(V a,V b,V c){return _mm512_fmadd_ps(a,b,c);}
D4_TARGET static inline V vfnma(V a,V b,V c){return _mm512_fnmadd_ps(a,b,c);}
D4_TARGET static inline V vfmas(V a,float s,V c){return _mm512_fmadd_ps(a,_mm512_set1_ps(s),c);}
D4_TARGET static inline V vmax(V a,V b){return _mm512_max_ps(a,b);}
D4_TARGET static inline V vabs(V a){return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a),_mm512_set1_epi32(0x7fffffff)));}
D4_TARGET static inline V selectneg(V v,V pos,V neg){return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(v,_mm512_setzero_ps(),_CMP_LT_OQ),pos,neg);}
D4_TARGET static inline float hsum(V a){
 __m256 x=_mm256_add_ps(_mm512_castps512_ps256(a),_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a),1)));
 __m128 y=_mm_add_ps(_mm256_castps256_ps128(x),_mm256_extractf128_ps(x,1));
 y=_mm_add_ps(y,_mm_movehl_ps(y,y));y=_mm_add_ss(y,_mm_shuffle_ps(y,y,1));return _mm_cvtss_f32(y);
}
D4_TARGET static inline float hmax(V a){
 __m256 x=_mm256_max_ps(_mm512_castps512_ps256(a),_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a),1)));
 __m128 y=_mm_max_ps(_mm256_castps256_ps128(x),_mm256_extractf128_ps(x,1));
 y=_mm_max_ps(y,_mm_movehl_ps(y,y));y=_mm_max_ss(y,_mm_shuffle_ps(y,y,1));return _mm_cvtss_f32(y);
}
// rcp14 (14 bits) + one Newton step.
D4_TARGET static inline V recip(V x){V e=_mm512_rcp14_ps(x);return _mm512_mul_ps(e,_mm512_fnmadd_ps(x,e,_mm512_set1_ps(2.f)));}
D4_TARGET static inline V exp_negative(V x){
 const V z=_mm512_max_ps(x,_mm512_set1_ps(-87.f));
 const V n=_mm512_roundscale_ps(_mm512_mul_ps(z,_mm512_set1_ps(1.4426950408889634f)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
 V r=_mm512_fnmadd_ps(n,_mm512_set1_ps(.693145751953125f),z);r=_mm512_fnmadd_ps(n,_mm512_set1_ps(1.428606765330187e-6f),r);
 // Degree-5 minimax of exp(r) on [-ln2/2,ln2/2]: relative error 7.5e-8.
 V p=_mm512_set1_ps(0.008297686563183427f);
 p=vfma(p,r,vset(0.04191538491131187f));p=vfma(p,r,vset(0.16667574268138968f));
 p=vfma(p,r,vset(0.4999889481644092f));p=vfma(p,r,vset(0.9999996921226256f));
 p=vfma(p,r,vset(1.0000000716597748f));
 const V power=_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n),_mm512_set1_epi32(127)),23));
 (void)x;return _mm512_mul_ps(p,power);
}
D4_TARGET static inline H hzero(){return _mm256_setzero_ps();}
D4_TARGET static inline H hload(const float* p){return _mm256_loadu_ps(p);}
D4_TARGET static inline void hstore(float* p,H v){_mm256_storeu_ps(p,v);}
D4_TARGET static inline H hfma(H a,H b,H c){return _mm256_fmadd_ps(a,b,c);}
D4_TARGET static inline H hfmas(H a,float s,H c){return _mm256_fmadd_ps(a,_mm256_set1_ps(s),c);}
D4_TARGET static inline float hsum_h(H a){
 __m128 x=_mm_add_ps(_mm256_castps256_ps128(a),_mm256_extractf128_ps(a,1));
 x=_mm_add_ps(x,_mm_movehl_ps(x,x));x=_mm_add_ss(x,_mm_shuffle_ps(x,x,1));return _mm_cvtss_f32(x);
}
#include "simd_impl.inc"
#undef D4_TARGET
}
#endif

size_t kernel_scratch(int dual,int h,int tile,int tokens){
 const int n=tile+2,p=winograd_group*tile*tile;
 const size_t wino=2*size_t(n*n)*winograd_group*dual*32+size_t(dual)*p*32;
 const size_t first=size_t(dual)*h*(9+32);
 const size_t attention=size_t(dual)*16*std::max(tokens,1);
 return std::max({wino,first,attention});
}

Kernels select_kernels(const std::string& isa){
 const auto s=resolve_isa(isa);
#if defined(__aarch64__)
 if(s=="neon")return neon_detail::table("neon");
#elif defined(__x86_64__)
 if(s=="avx2")return avx2_detail::table("avx2");
 if(s=="avx512")return avx512_detail::table("avx512");
#endif
 Kernels none;none.isa=s;return none;
}

// Seconds for a fixed batched-GEMM workload shaped like one F4 hidden layer
// group; used to choose between AVX2 and AVX512 on CPUs where the wider unit
// may be slower (e.g. single-FMA-port Xeons). Best of three repetitions.
double probe_isa_seconds(const std::string& isa){
 const auto k=select_kernels(isa);
 if(!k.gemm)return std::numeric_limits<double>::infinity();
 std::vector<float> a(36*24*32),b(36*1024),c(36*24*32);
 for(size_t i=0;i<a.size();++i)a[i]=float(i%7)*.01f;
 for(size_t i=0;i<b.size();++i)b[i]=float(i%5)*.01f;
 double best=std::numeric_limits<double>::infinity();
 for(int rep=0;rep<3;++rep){
  const auto start=std::chrono::steady_clock::now();
  for(int it=0;it<60;++it)for(int t=0;t<36;++t)k.gemm(a.data()+t*768,32,b.data()+t*1024,32,c.data()+t*768,32,24,32,32);
  best=std::min(best,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
 }
 return best;
}
}
