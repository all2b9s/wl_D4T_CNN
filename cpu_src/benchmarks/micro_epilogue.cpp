// Micro-benchmark of the fused epilogue alone: ns/pixel for an L1-resident
// 8x8 plane vs a 64x64 plane (1.5 MB for D=3), store vs pool sink.
#include "d4cnn/kernels.h"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
int main(int argc,char** argv){
 const auto k=d4cnn::select_kernels(argc>1?argv[1]:"auto");
 std::mt19937 rng(1);std::normal_distribution<float> g(0,1);
 for(int h:{8,64}){
  const int dual=3,p=h*h;
  std::vector<float> y(size_t(dual)*p*32),x(y.size()),gamma(32),beta(32),pooled(dual*32),w(p,1.f/p);
  for(auto& v:y)v=g(rng);for(auto& v:x)v=g(rng);for(auto& v:gamma)v=1+.1f*g(rng);for(auto& v:beta)v=.1f*g(rng);
  const auto y0=y;
  for(int mode=0;mode<2;++mode){
   d4cnn::Epi e;e.mode=mode?d4cnn::EpiMode::pool:d4cnn::EpiMode::store;e.gamma=gamma.data();e.beta=beta.data();
   e.residual=.1f;e.x=x.data();e.xstride=p*32;e.pooled=pooled.data();e.pool_w=w.data();
   const int reps=h==8?20000:400;double best=1e30;
   for(int t=0;t<3;++t){
    auto start=std::chrono::steady_clock::now();
    for(int r=0;r<reps;++r){if(!mode)y=y0;k.epilogue_plane(dual,y.data(),p,e);}
    best=std::min(best,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
   }
   std::printf("%s h=%d mode=%s: %.1f ns/pixel (D=3, includes plane copy for store)\n",k.isa.c_str(),h,mode?"pool":"store",best/reps/p*1e9);
  }
  // Components: GELU alone (32 values + 2 tangents per pixel), LayerNorm alone, plane copy.
  const int reps=h==8?20000:400;
  for(int part=0;part<3;++part){double best=1e30;
   for(int t=0;t<3;++t){auto start=std::chrono::steady_clock::now();
    for(int r=0;r<reps;++r){y=y0;if(part==0)k.gelu(dual,y.data(),p*32,p*32);else if(part==1)k.norm_rows(dual,y.data(),p,p*32,gamma.data(),beta.data());}
    best=std::min(best,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());}
   std::printf("%s h=%d %s: %.1f ns/pixel\n",k.isa.c_str(),h,part==0?"gelu only":part==1?"layernorm only":"copy only",best/reps/p*1e9);
  }
 }
}
