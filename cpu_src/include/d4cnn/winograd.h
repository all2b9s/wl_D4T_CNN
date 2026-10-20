#pragma once
#include <array>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace d4cnn {
// F(2x2,3x3), cross-correlation with one-pixel zero padding. Only C32.
// Packed weight order is [transform][input channel][output channel].
inline std::vector<float> winograd_weights(const std::vector<float>& weights) {
 constexpr float g[4][3]={{1,0,0},{.5,.5,.5},{.5,-.5,.5},{0,0,1}};
 std::vector<float> out(16*32*32);
 for(int i=0;i<32;++i)for(int o=0;o<32;++o)for(int a=0;a<4;++a)for(int b=0;b<4;++b){
  float v=0;for(int y=0;y<3;++y)for(int x=0;x<3;++x)v+=g[a][y]*weights[((y*3+x)*32+i)*32+o]*g[b][x];
  out[((a*4+b)*32+i)*32+o]=v;
 }return out;
}
#if defined(__aarch64__)
template<int D,class Epilogue> void winograd32(const float* x,const float* weights,const float* bias,float* y,int h,Epilogue finish) {
 const int stride=h*h*32;
 alignas(64) float input[16][D][32],output[16][D][32];
 for(int oy=0;oy<h;oy+=2)for(int ox=0;ox<h;ox+=2){
  for(int d=0;d<D;++d)for(int c=0;c<32;c+=4){
   float32x4_t a[4][4],t[4][4];
   for(int r=0;r<4;++r)for(int s=0;s<4;++s){int yy=oy+r-1,xx=ox+s-1;
    a[r][s]=(yy>=0&&yy<h&&xx>=0&&xx<h)?vld1q_f32(x+d*stride+(yy*h+xx)*32+c):vdupq_n_f32(0);}
   for(int s=0;s<4;++s){t[0][s]=vsubq_f32(a[0][s],a[2][s]);t[1][s]=vaddq_f32(a[1][s],a[2][s]);
    t[2][s]=vsubq_f32(a[2][s],a[1][s]);t[3][s]=vsubq_f32(a[1][s],a[3][s]);}
   for(int r=0;r<4;++r){
    vst1q_f32(input[r*4][d]+c,vsubq_f32(t[r][0],t[r][2]));
    vst1q_f32(input[r*4+1][d]+c,vaddq_f32(t[r][1],t[r][2]));
    vst1q_f32(input[r*4+2][d]+c,vsubq_f32(t[r][2],t[r][1]));
    vst1q_f32(input[r*4+3][d]+c,vsubq_f32(t[r][1],t[r][3]));}
  }
  constexpr int lanes=D==3?8:4;
  for(int t=0;t<16;++t)for(int o=0;o<32;o+=4*lanes){
   float32x4_t acc[D][lanes];
   for(int d=0;d<D;++d)for(int j=0;j<lanes;++j)acc[d][j]=vdupq_n_f32(0);
   for(int i=0;i<32;++i){float v[D];for(int d=0;d<D;++d)v[d]=input[t][d][i];
    for(int j=0;j<lanes;++j){auto w=vld1q_f32(weights+(t*32+i)*32+o+4*j);
     for(int d=0;d<D;++d)acc[d][j]=vfmaq_n_f32(acc[d][j],w,v[d]);}}
   for(int d=0;d<D;++d)for(int j=0;j<lanes;++j)vst1q_f32(output[t][d]+o+4*j,acc[d][j]);
  }
  for(int d=0;d<D;++d)for(int c=0;c<32;c+=4){
   float32x4_t t[2][4];
   for(int j=0;j<4;++j){auto a=vld1q_f32(output[j][d]+c),b=vld1q_f32(output[4+j][d]+c);
    auto cc=vld1q_f32(output[8+j][d]+c),dd=vld1q_f32(output[12+j][d]+c);
    t[0][j]=vaddq_f32(vaddq_f32(a,b),cc);t[1][j]=vsubq_f32(vsubq_f32(b,cc),dd);}
   auto b=d==0?vld1q_f32(bias+c):vdupq_n_f32(0);
   for(int r=0;r<2&&oy+r<h;++r){
    auto a=vaddq_f32(vaddq_f32(t[r][0],t[r][1]),t[r][2]);
    auto z=vsubq_f32(vsubq_f32(t[r][1],t[r][2]),t[r][3]);
    vst1q_f32(y+d*stride+((oy+r)*h+ox)*32+c,vaddq_f32(a,b));
    if(ox+1<h)vst1q_f32(y+d*stride+((oy+r)*h+ox+1)*32+c,vaddq_f32(z,b));}
  }
  for(int r=0;r<2&&oy+r<h;++r)for(int s=0;s<2&&ox+s<h;++s)finish((oy+r)*h+ox+s);
 }
}
#endif
}
