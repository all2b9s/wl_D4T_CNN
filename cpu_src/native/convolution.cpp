#include "d4cnn/convolution.h"
namespace d4cnn {
// Weights [ky,kx,ic,oc]; activations [dual,pixel,channel].
void conv_scalar(const float* x,const float* w,const float* b,float* y,
                 int h,int ci,int co,int nd) {
 const int p=h*h;
 for(int pos=0;pos<p;++pos) {
  for(int d=0;d<nd;++d) for(int o=0;o<co;++o) y[(d*p+pos)*co+o]=(d==0?b[o]:0.f);
  for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) {
   int yy=pos/h+ky-1,xx=pos%h+kx-1;
   if(yy<0||xx<0||yy>=h||xx>=h) continue;
   for(int i=0;i<ci;++i) {
    const float* ww=w+((ky*3+kx)*ci+i)*co;
    for(int d=0;d<nd;++d) {
     float a=x[(d*p+yy*h+xx)*ci+i]; float* out=y+(d*p+pos)*co;
     for(int o=0;o<co;++o) out[o]+=a*ww[o];
    }
   }
  }
 }
}
}
