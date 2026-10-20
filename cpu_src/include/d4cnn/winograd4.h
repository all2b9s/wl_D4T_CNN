#pragma once
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace d4cnn {
// F(4x4,3x3). The 1D matrices satisfy
// sum_t A[o,t]*G[t,k]*B[t,i] == (i == o+k); see test_winograd_identity.py.
inline std::vector<float> winograd4_weights(const std::vector<float>& weights){
 constexpr double g[6][3]={{.25,0,0},{-1./6,-1./6,-1./6},
  {-1./6,1./6,-1./6},{1./24,1./12,1./6},{1./24,-1./12,1./6},{0,0,1}};
 std::vector<float> out(36*32*32);
 for(int i=0;i<32;++i)for(int o=0;o<32;++o)for(int a=0;a<6;++a)for(int b=0;b<6;++b){
  double v=0;for(int y=0;y<3;++y)for(int x=0;x<3;++x)
   v+=g[a][y]*double(weights[((y*3+x)*32+i)*32+o])*g[b][x];
  out[((a*6+b)*32+i)*32+o]=float(v);
 }return out;
}
#if defined(__aarch64__)
inline void winograd4_input(const float32x4_t* a,float32x4_t* b){
 b[0]=vfmaq_n_f32(vfmsq_n_f32(a[4],a[2],5.f),a[0],4.f);
 b[1]=vfmsq_n_f32(vaddq_f32(a[3],a[4]),vaddq_f32(a[1],a[2]),4.f);
 b[2]=vfmaq_n_f32(vsubq_f32(a[4],a[3]),vsubq_f32(a[1],a[2]),4.f);
 b[3]=vfmaq_n_f32(vsubq_f32(a[4],a[2]),vsubq_f32(a[3],a[1]),2.f);
 b[4]=vfmaq_n_f32(vsubq_f32(a[4],a[2]),vsubq_f32(a[1],a[3]),2.f);
 b[5]=vfmaq_n_f32(vfmsq_n_f32(a[5],a[3],5.f),a[1],4.f);
}
inline void winograd4_output(const float32x4_t* a,float32x4_t* b){
 auto sum12=vaddq_f32(a[1],a[2]),sum34=vaddq_f32(a[3],a[4]);
 auto diff12=vsubq_f32(a[1],a[2]),diff34=vsubq_f32(a[3],a[4]);
 b[0]=vaddq_f32(vaddq_f32(a[0],sum12),sum34);
 b[1]=vfmaq_n_f32(diff12,diff34,2.f);
 b[2]=vfmaq_n_f32(sum12,sum34,4.f);
 b[3]=vaddq_f32(vfmaq_n_f32(diff12,diff34,8.f),a[5]);
}
template<int D,class Epilogue>
void winograd4_32(const float* x,const float* weights,const float* bias,float* y,int h,Epilogue finish){
 const int stride=h*h*32;
 alignas(64) float input[36][D][32],output[36][D][32];
 for(int oy=0;oy<h;oy+=4)for(int ox=0;ox<h;ox+=4){
  for(int d=0;d<D;++d)for(int c=0;c<32;c+=4){
   float32x4_t t[6][6];
   for(int s=0;s<6;++s){float32x4_t a[6],b[6];
    for(int r=0;r<6;++r){int yy=oy+r-1,xx=ox+s-1;
     a[r]=(yy>=0&&yy<h&&xx>=0&&xx<h)?vld1q_f32(x+d*stride+(yy*h+xx)*32+c):vdupq_n_f32(0);}
    winograd4_input(a,b);for(int r=0;r<6;++r)t[r][s]=b[r];
   }
   for(int r=0;r<6;++r){float32x4_t b[6];winograd4_input(t[r],b);
    for(int s=0;s<6;++s)vst1q_f32(input[r*6+s][d]+c,b[s]);}
  }
  constexpr int lanes=D==3?8:4;
  for(int t=0;t<36;++t)for(int o=0;o<32;o+=4*lanes){
   float32x4_t acc[D][lanes];
   for(int d=0;d<D;++d)for(int j=0;j<lanes;++j)acc[d][j]=vdupq_n_f32(0);
   for(int i=0;i<32;++i){float v[D];for(int d=0;d<D;++d)v[d]=input[t][d][i];
    for(int j=0;j<lanes;++j){auto w=vld1q_f32(weights+(t*32+i)*32+o+4*j);
     for(int d=0;d<D;++d)acc[d][j]=vfmaq_n_f32(acc[d][j],w,v[d]);}}
   for(int d=0;d<D;++d)for(int j=0;j<lanes;++j)vst1q_f32(output[t][d]+o+4*j,acc[d][j]);
  }
  for(int d=0;d<D;++d)for(int c=0;c<32;c+=4){
   float32x4_t t[4][6];
   for(int s=0;s<6;++s){float32x4_t a[6],b[4];
    for(int r=0;r<6;++r)a[r]=vld1q_f32(output[r*6+s][d]+c);
    winograd4_output(a,b);for(int r=0;r<4;++r)t[r][s]=b[r];
   }
   auto biasv=d==0?vld1q_f32(bias+c):vdupq_n_f32(0);
   for(int r=0;r<4&&oy+r<h;++r){float32x4_t b[4];winograd4_output(t[r],b);
    for(int s=0;s<4&&ox+s<h;++s)vst1q_f32(y+d*stride+((oy+r)*h+ox+s)*32+c,vaddq_f32(b[s],biasv));}
  }
  for(int r=0;r<4&&oy+r<h;++r)for(int s=0;s<4&&ox+s<h;++s)finish((oy+r)*h+ox+s);
 }
}
#endif
}
