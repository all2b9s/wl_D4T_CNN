#pragma once
#include <cmath>
#if defined(__aarch64__)
#include <arm_neon.h>
namespace d4cnn::simd {
// Restricted-domain exp for erf-GELU: x <= 0. Values below -87 contribute
// less than float32 rounding to GELU. No global fast-math/FP mode changes.
inline float32x4_t exp_negative(float32x4_t x) {
 auto z=vmaxq_f32(x,vdupq_n_f32(-87.f));
 auto n=vrndnq_f32(vmulq_n_f32(z,1.4426950408889634f));
 auto r=vfmsq_n_f32(z,n,0.693145751953125f);
 r=vfmsq_n_f32(r,n,1.428606765330187e-6f);
 auto p=vdupq_n_f32(1.f/5040.f);
 p=vfmaq_f32(vdupq_n_f32(1.f/720.f),p,r);
 p=vfmaq_f32(vdupq_n_f32(1.f/120.f),p,r);
 p=vfmaq_f32(vdupq_n_f32(1.f/24.f),p,r);
 p=vfmaq_f32(vdupq_n_f32(1.f/6.f),p,r);
 p=vfmaq_f32(vdupq_n_f32(.5f),p,r);
 p=vfmaq_f32(vdupq_n_f32(1.f),p,r);
 p=vfmaq_f32(vdupq_n_f32(1.f),p,r);
 auto power=vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n),vdupq_n_s32(127)),23));
 return vbslq_f32(vcgeq_f32(x,vdupq_n_f32(-87.f)),vmulq_f32(p,power),vdupq_n_f32(0));
}
inline void gelu4(float32x4_t v,float32x4_t& value,float32x4_t& df) {
  auto z=vmulq_n_f32(v,.7071067811865475f);
  auto t=vdivq_f32(vdupq_n_f32(1.f),vaddq_f32(vdupq_n_f32(1.f),vmulq_n_f32(vabsq_f32(z),.3275911f)));
  auto poly=vdupq_n_f32(1.061405429f);
  poly=vfmaq_f32(vdupq_n_f32(-1.453152027f),poly,t);
  poly=vfmaq_f32(vdupq_n_f32(1.421413741f),poly,t);
  poly=vfmaq_f32(vdupq_n_f32(-.284496736f),poly,t);
  poly=vfmaq_f32(vdupq_n_f32(.254829592f),poly,t);
  auto ex=exp_negative(vmulq_n_f32(vmulq_f32(v,v),-.5f));
  auto ea=vsubq_f32(vdupq_n_f32(1.f),vmulq_f32(vmulq_f32(poly,t),ex));
  auto erfv=vbslq_f32(vcltq_f32(v,vdupq_n_f32(0)),vnegq_f32(ea),ea);
  auto phi=vmulq_n_f32(vaddq_f32(vdupq_n_f32(1.f),erfv),.5f);
  df=vfmaq_f32(phi,vmulq_n_f32(v,.3989422804014327f),ex);
 value=vmulq_f32(v,phi);
}
template<int D> inline void gelu(float* base,int stride,int count) {
 int i=0;
 for(;i+4<=count;i+=4){
  float32x4_t value,df;gelu4(vld1q_f32(base+i),value,df);
  for(int d=1;d<D;++d)vst1q_f32(base+d*stride+i,vmulq_f32(vld1q_f32(base+d*stride+i),df));
  vst1q_f32(base+i,value);
 }
 for(;i<count;++i){float v=base[i],phi=.5f*(1.f+std::erf(v*.7071067811865475f));
  float df=phi+v*std::exp(-.5f*v*v)*.3989422804014327f;
  for(int d=1;d<D;++d)base[d*stride+i]*=df;base[i]=v*phi;
 }
}
template<int D> inline void norm32(float* base,int stride,const float* gamma,const float* beta) {
 float32x4_t sum=vdupq_n_f32(0),center[8];
 for(int j=0;j<8;++j)sum=vaddq_f32(sum,vld1q_f32(base+4*j));
 float mu=vaddvq_f32(sum)/32.f;sum=vdupq_n_f32(0);
 for(int j=0;j<8;++j){center[j]=vsubq_f32(vld1q_f32(base+4*j),vdupq_n_f32(mu));sum=vfmaq_f32(sum,center[j],center[j]);}
 float inv=1.f/std::sqrt(vaddvq_f32(sum)/32.f+1e-5f);
 for(int d=1;d<D;++d){float* tangent=base+d*stride;sum=vdupq_n_f32(0);
  for(int j=0;j<8;++j)sum=vaddq_f32(sum,vld1q_f32(tangent+4*j));
  float dm=vaddvq_f32(sum)/32.f;sum=vdupq_n_f32(0);
  for(int j=0;j<8;++j)sum=vfmaq_f32(sum,center[j],vsubq_f32(vld1q_f32(tangent+4*j),vdupq_n_f32(dm)));
  float cov=vaddvq_f32(sum)/32.f;
  for(int j=0;j<8;++j){auto dt=vsubq_f32(vld1q_f32(tangent+4*j),vdupq_n_f32(dm));
   dt=vsubq_f32(dt,vmulq_n_f32(vmulq_n_f32(vmulq_n_f32(center[j],cov),inv),inv));
   vst1q_f32(tangent+4*j,vmulq_f32(vmulq_n_f32(vld1q_f32(gamma+4*j),inv),dt));}
 }
 for(int j=0;j<8;++j)vst1q_f32(base+4*j,vaddq_f32(vmulq_f32(vmulq_n_f32(center[j],inv),vld1q_f32(gamma+4*j)),vld1q_f32(beta+4*j)));
}
// All dual statistics are reduced before overwriting the convolution output.
template<int D> inline void epilogue32(float* base,int stride,const float* gamma,const float* beta,const float* x,int xstride,const float* skip,float residual) {
 float32x4_t sum=vdupq_n_f32(0),center[8];
 for(int j=0;j<8;++j)sum=vaddq_f32(sum,vld1q_f32(base+4*j));
 float mu=vaddvq_f32(sum)/32.f;sum=vdupq_n_f32(0);
 for(int j=0;j<8;++j){center[j]=vsubq_f32(vld1q_f32(base+4*j),vdupq_n_f32(mu));sum=vfmaq_f32(sum,center[j],center[j]);}
 float inv=1.f/std::sqrt(vaddvq_f32(sum)/32.f+1e-5f),dm[D]{},cov[D]{};
 for(int d=1;d<D;++d){sum=vdupq_n_f32(0);
  for(int j=0;j<8;++j)sum=vaddq_f32(sum,vld1q_f32(base+d*stride+4*j));
  dm[d]=vaddvq_f32(sum)/32.f;sum=vdupq_n_f32(0);
  for(int j=0;j<8;++j)sum=vfmaq_f32(sum,center[j],vsubq_f32(vld1q_f32(base+d*stride+4*j),vdupq_n_f32(dm[d])));
  cov[d]=vaddvq_f32(sum)/32.f;
 }
 for(int j=0;j<8;++j){auto g=vld1q_f32(gamma+4*j);
  auto v=vaddq_f32(vmulq_f32(vmulq_n_f32(center[j],inv),g),vld1q_f32(beta+4*j));
  float32x4_t value,df;gelu4(v,value,df);
  for(int d=0;d<D;++d){auto out=value;
   if(d){auto dt=vsubq_f32(vld1q_f32(base+d*stride+4*j),vdupq_n_f32(dm[d]));
    dt=vsubq_f32(dt,vmulq_n_f32(vmulq_n_f32(vmulq_n_f32(center[j],cov[d]),inv),inv));
    out=vmulq_f32(vmulq_f32(vmulq_n_f32(g,inv),dt),df);}
   auto xx=skip?vmulq_n_f32(vld1q_f32(skip+4*j),x[d*xstride]):vld1q_f32(x+d*xstride+4*j);
   vst1q_f32(base+d*stride+4*j,vfmaq_n_f32(out,xx,residual));
  }
 }
}

}
#endif
