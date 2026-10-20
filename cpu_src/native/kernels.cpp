#include "d4cnn/convolution.h"
#include "d4cnn/kernels.h"
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif
namespace d4cnn {
#if defined(__aarch64__)
// The dual count and channel tile are compile-time constants so the compiler
// can keep independent accumulators in registers (24 for T=2, 20 for T=4).
template<int D,int L> static void neon_tiles(const float*x,const float*w,const float*b,float*y,int h,int ci,int co) {
 constexpr int width=4*L;
 int p=h*h;
 for(int pos=0;pos<p;++pos){int o=0;
  for(;o+width<=co;o+=width){
   float32x4_t acc[D][L];
   for(int d=0;d<D;++d)for(int l=0;l<L;++l)acc[d][l]=d==0?vld1q_f32(b+o+4*l):vdupq_n_f32(0);
   for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int yy=pos/h+ky-1,xx=pos%h+kx-1;
    if(yy<0||xx<0||yy>=h||xx>=h)continue;
    for(int i=0;i<ci;++i){
     float values[D];for(int d=0;d<D;++d)values[d]=x[(d*p+yy*h+xx)*ci+i];
     for(int l=0;l<L;++l){auto ww=vld1q_f32(w+((ky*3+kx)*ci+i)*co+o+4*l);
      for(int d=0;d<D;++d)acc[d][l]=vfmaq_n_f32(acc[d][l],ww,values[d]);}
    }
   }
   for(int d=0;d<D;++d)for(int l=0;l<L;++l)vst1q_f32(y+(d*p+pos)*co+o+4*l,acc[d][l]);
  }
  // Small models and non-multiple channel counts use a four-lane tail.
  for(;o+4<=co;o+=4){float32x4_t acc[D];for(int d=0;d<D;++d)acc[d]=d==0?vld1q_f32(b+o):vdupq_n_f32(0);
   for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int yy=pos/h+ky-1,xx=pos%h+kx-1;
    if(yy<0||xx<0||yy>=h||xx>=h)continue;
    for(int i=0;i<ci;++i){auto ww=vld1q_f32(w+((ky*3+kx)*ci+i)*co+o);
     for(int d=0;d<D;++d)acc[d]=vfmaq_n_f32(acc[d],ww,x[(d*p+yy*h+xx)*ci+i]);}}
   for(int d=0;d<D;++d)vst1q_f32(y+(d*p+pos)*co+o,acc[d]);
  }
  for(;o<co;++o)for(int d=0;d<D;++d){float v=d==0?b[o]:0;
   for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int yy=pos/h+ky-1,xx=pos%h+kx-1;
    if(yy<0||xx<0||yy>=h||xx>=h)continue;
    for(int i=0;i<ci;++i)v+=x[(d*p+yy*h+xx)*ci+i]*w[((ky*3+kx)*ci+i)*co+o];}y[(d*p+pos)*co+o]=v;}
 }
}
static void conv_neon(const float*x,const float*w,const float*b,float*y,int h,int ci,int co,int nd){
 if(nd==3)neon_tiles<3,8>(x,w,b,y,h,ci,co);
 else if(nd==5)neon_tiles<5,4>(x,w,b,y,h,ci,co);
 else throw std::invalid_argument("Only 3 or 5 dual components are supported");
}
#elif defined(__x86_64__)
#define DEFINE_X86(NAME,ATTR,VEC,LANES,ZERO,LOAD,STORE,BCAST,FMA) \
__attribute__((target(ATTR))) static void NAME(const float*x,const float*w,const float*b,float*y,int h,int ci,int co,int nd) { \
 int p=h*h; for(int pos=0;pos<p;++pos) { int o=0; \
 for(;o+LANES<=co;o+=LANES) { VEC acc[5]; for(int d=0;d<nd;++d) acc[d]=d==0?LOAD(b+o):ZERO(); \
 for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) { int yy=pos/h+ky-1,xx=pos%h+kx-1; \
 if(yy<0||xx<0||yy>=h||xx>=h) continue; for(int i=0;i<ci;++i) { VEC ww=LOAD(w+((ky*3+kx)*ci+i)*co+o); \
 for(int d=0;d<nd;++d) acc[d]=FMA(BCAST(x[(d*p+yy*h+xx)*ci+i]),ww,acc[d]); }} \
 for(int d=0;d<nd;++d) STORE(y+(d*p+pos)*co+o,acc[d]); } \
 for(;o<co;++o) for(int d=0;d<nd;++d) {float a=d==0?b[o]:0.f; \
 for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) {int yy=pos/h+ky-1,xx=pos%h+kx-1; \
 if(yy<0||xx<0||yy>=h||xx>=h) continue; for(int i=0;i<ci;++i) a+=x[(d*p+yy*h+xx)*ci+i]*w[((ky*3+kx)*ci+i)*co+o];} y[(d*p+pos)*co+o]=a;} }}
DEFINE_X86(conv_avx2,"avx2,fma",__m256,8,_mm256_setzero_ps,_mm256_loadu_ps,_mm256_storeu_ps,_mm256_set1_ps,_mm256_fmadd_ps)
DEFINE_X86(conv_avx512,"avx512f,fma",__m512,16,_mm512_setzero_ps,_mm512_loadu_ps,_mm512_storeu_ps,_mm512_set1_ps,_mm512_fmadd_ps)
#endif
std::vector<std::string> available_isas() {
 std::vector<std::string> out{"scalar"};
#if defined(__aarch64__)
 out.push_back("neon");
#elif defined(__x86_64__)
 __builtin_cpu_init();
 if(__builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma")) out.push_back("avx2");
 if(__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("fma")) out.push_back("avx512");
#endif
 return out;
}
// "auto": D4CNN_ISA if set; otherwise the widest ISA, except that x86 CPUs with
// both AVX2 and AVX512 time a short GEMM probe once per process and keep
// AVX512 only if it is at least 3% faster (D4CNN_ISA_PROBE=0 disables this).
std::string resolve_isa(const std::string& requested) {
 std::string s=requested; if(s=="auto") {const char* v=std::getenv("D4CNN_ISA");if(v)s=v;}
 auto all=available_isas();
 if(s=="auto"){
#if defined(__x86_64__)
  const char* probe=std::getenv("D4CNN_ISA_PROBE");
  if(all.back()=="avx512"&&!(probe&&std::string(probe)=="0")){
   static std::once_flag once;static std::string chosen;
   std::call_once(once,[]{chosen=probe_isa_seconds("avx512")<=.97*probe_isa_seconds("avx2")?"avx512":"avx2";});
   return chosen;
  }
#endif
  return all.back();
 }
 for(auto& a:all)if(s==a)return s;
 throw std::invalid_argument("ISA unavailable: "+s);
}
Conv select_conv(const std::string& isa) {
 auto s=resolve_isa(isa);
#if defined(__aarch64__)
 if(s=="neon")return conv_neon;
#elif defined(__x86_64__)
 if(s=="avx2")return conv_avx2;
 if(s=="avx512")return conv_avx512;
#endif
 return conv_scalar;
}
}
