#include "d4cnn/convolution.h"
#include "d4cnn/x86.h"
#include "d4cnn/winograd.h"
#include "d4cnn/winograd4.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

int main(){
 std::mt19937 rng(67);std::normal_distribution<float> normal(0,.1f);
 bool exercised=false;
 for(const auto& isa:d4cnn::available_isas()){
  if(isa!="avx2"&&isa!="avx512")continue;
  exercised=true;const auto kernels=d4cnn::select_x86_kernels(isa);
  for(int dual:{3,5}){
   // Independent double LN/GELU/dual reference, including strided buffers and
   // both identity and first-layer projected residuals.
   for(bool projected:{false,true}){
    constexpr int stride=64;
    std::vector<float> input(dual*stride),out,gamma(32),beta(32),x(dual*stride),skip(32);
    for(auto& v:input)v=normal(rng);for(auto& v:gamma)v=normal(rng);
    for(auto& v:beta)v=normal(rng);for(auto& v:x)v=normal(rng);for(auto& v:skip)v=normal(rng);
    out=input;kernels.epilogue(out.data(),stride,gamma.data(),beta.data(),x.data(),
                             projected?1:stride,projected?skip.data():nullptr,.7f,dual);
    double mu=0,var=0;for(int j=0;j<32;++j)mu+=input[j];mu/=32;
    for(int j=0;j<32;++j)var+=(input[j]-mu)*(input[j]-mu);
    double inv=1/std::sqrt(var/32+double(1e-5f));
    for(int d=0;d<dual;++d){double dm=0,cov=0;
     if(d){for(int j=0;j<32;++j)dm+=input[d*stride+j];dm/=32;
      for(int j=0;j<32;++j)cov+=(input[j]-mu)*(input[d*stride+j]-dm);cov/=32;}
     for(int j=0;j<32;++j){double v=(input[j]-mu)*inv*gamma[j]+beta[j];
      double phi=.5*(1+std::erf(v/std::sqrt(2.))),df=phi+v*std::exp(-.5*v*v)/std::sqrt(2*std::acos(-1.));
      double expected=d?gamma[j]*inv*(input[d*stride+j]-dm-(input[j]-mu)*cov*inv*inv)*df:v*phi;
      expected+=double(.7f)*(projected?double(skip[j])*x[d]:x[d*stride+j]);
      if(std::abs(expected-out[d*stride+j])>3e-6)throw std::runtime_error("x86 epilogue accuracy");}
     for(int j=32;j<stride;++j)if(out[d*stride+j]!=input[d*stride+j])throw std::runtime_error("x86 epilogue overwrote padding");
    }
   }
   for(int tile:{2,4})for(int h:{1,2,3,4,5,8,9,16}){
    int stride=h*h*32;
    std::vector<float> x(dual*stride),w(9*32*32),b(32),out(x.size()),fused(x.size()),gamma(32),beta(32);
    for(auto& v:x)v=normal(rng);for(auto& v:w)v=normal(rng);for(auto& v:b)v=normal(rng);
    for(auto& v:gamma)v=normal(rng);for(auto& v:beta)v=normal(rng);
    auto packed=tile==4?d4cnn::winograd4_weights(w):d4cnn::winograd_weights(w);
    auto kernel=tile==4?kernels.f4:kernels.f2;
    kernel(x.data(),packed.data(),b.data(),out.data(),h,dual,nullptr,nullptr,0,false);
    double largest=0;
    for(int d=0;d<dual;++d)for(int y=0;y<h;++y)for(int xx=0;xx<h;++xx)for(int o=0;o<32;++o){
     double sum=d==0?b[o]:0;
     for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int yy=y+ky-1,z=xx+kx-1;
      if(yy<0||yy>=h||z<0||z>=h)continue;
      for(int c=0;c<32;++c)sum+=double(x[(d*h*h+yy*h+z)*32+c])*w[((ky*3+kx)*32+c)*32+o];}
     largest=std::max(largest,std::abs(sum-out[(d*h*h+y*h+xx)*32+o]));
    }
    std::cout<<isa<<" F="<<tile<<" D="<<dual<<" H="<<h<<" error="<<largest<<std::endl;
    if(largest>(tile==4?1e-5:2e-6))throw std::runtime_error("x86 Winograd accuracy");
    kernel(x.data(),packed.data(),b.data(),fused.data(),h,dual,gamma.data(),beta.data(),.7f,true);
    for(int p=0;p<h*h;++p)kernels.epilogue(out.data()+p*32,stride,gamma.data(),beta.data(),x.data()+p*32,stride,nullptr,.7f,dual);
    for(size_t i=0;i<out.size();++i)if(std::abs(out[i]-fused[i])>1e-6)throw std::runtime_error("x86 fused tile accuracy");
   }
  }
 }
 if(!exercised){std::cout<<"SKIP: no AVX2/AVX512 CPU available"<<std::endl;return 77;}
 std::cout<<"x86 SIMD operator checks passed"<<std::endl;
}
