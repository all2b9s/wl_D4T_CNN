#include "d4cnn/simd_math.h"
#include "d4cnn/winograd.h"
#include "d4cnn/winograd4.h"
#include "d4cnn/parallel.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
int main(){
 for(int threads:{2,4,2}){
  std::vector<int> values(73);
  for(int pass=0;pass<10;++pass){
   d4cnn::parallel_for(73,threads,[&](int i){values[i]+=1;});
  }
  for(auto v:values)if(v!=10)throw std::runtime_error("executor lost a task");
  bool caught=false;
  try{d4cnn::parallel_for(73,threads,[](int i){if(i==3)throw std::runtime_error("test");});}
  catch(const std::runtime_error&){caught=true;}
  if(!caught)throw std::runtime_error("executor lost exception");
  d4cnn::parallel_for(73,threads,[&](int i){values[i]=i;});
  for(int i=0;i<73;++i)if(values[i]!=i)throw std::runtime_error("executor recovery");
 }
 std::cout<<"Executor reuse, resize and exception recovery passed"<<std::endl;
#if defined(__aarch64__)
 float max_value=0,max_derivative=0;
 for(int i=0;i<=30000;i+=4){float a[12];
  for(int j=0;j<4;++j){a[j]=-15.f+(i+j)*.001f;a[4+j]=1;a[8+j]=-2;}
  float original[4];std::copy(a,a+4,original);d4cnn::simd::gelu<3>(a,4,4);
  for(int j=0;j<4;++j){double v=original[j],phi=.5*(1+std::erf(v/std::sqrt(2.)));
   double derivative=phi+v*std::exp(-.5*v*v)/std::sqrt(2*std::acos(-1.));
   max_value=std::max(max_value,float(std::abs(a[j]-v*phi)));
   max_derivative=std::max(max_derivative,float(std::abs(a[4+j]-derivative)));
   if(a[8+j]!=-2*a[4+j])throw std::runtime_error("dual direction scaling");}
 }
 std::cout<<"GELU max value error "<<max_value<<" derivative "<<max_derivative<<std::endl;
 if(max_value>3e-6||max_derivative>5e-7)throw std::runtime_error("GELU accuracy");
 std::mt19937 rng(67);std::normal_distribution<float> normal(0,.1);
 auto test=[&](auto tag){constexpr int D=decltype(tag)::value;
  // Independent existing composition checks both identity and projected skips.
  for(bool projected:{false,true}) {
   std::vector<float> input(D*32),expected,actual,gamma(32),beta(32),skip(32),x(D*32);
   for(auto& v:input)v=normal(rng);for(auto& v:gamma)v=normal(rng);
   for(auto& v:beta)v=normal(rng);for(auto& v:skip)v=normal(rng);for(auto& v:x)v=normal(rng);
   expected=input;actual=input;
   d4cnn::simd::norm32<D>(expected.data(),32,gamma.data(),beta.data());
   d4cnn::simd::gelu<D>(expected.data(),32,32);
   for(int d=0;d<D;++d)for(int j=0;j<32;++j) {
    float residual=projected?skip[j]*x[d]:x[d*32+j];
    expected[d*32+j]=std::fma(residual,.7f,expected[d*32+j]);
   }
   d4cnn::simd::epilogue32<D>(actual.data(),32,gamma.data(),beta.data(),x.data(),projected?1:32,projected?skip.data():nullptr,.7f);
   for(size_t i=0;i<actual.size();++i)if(std::abs(actual[i]-expected[i])>1e-6f)
    throw std::runtime_error("fused epilogue accuracy");
  }
  for(int tile:{2,4})for(int h:{1,2,3,4,5,8,9,16}){std::vector<float>x(D*h*h*32),w(9*32*32),b(32),out(x.size());
   for(auto& v:x)v=normal(rng);for(auto& v:w)v=normal(rng);for(auto& v:b)v=normal(rng);
   auto packed=tile==4?d4cnn::winograd4_weights(w):d4cnn::winograd_weights(w);
   if(tile==4)d4cnn::winograd4_32<D>(x.data(),packed.data(),b.data(),out.data(),h,[](int){});
   else d4cnn::winograd32<D>(x.data(),packed.data(),b.data(),out.data(),h,[](int){});
   double largest=0;
   for(int d=0;d<D;++d)for(int y=0;y<h;++y)for(int xx=0;xx<h;++xx)for(int o=0;o<32;++o){
    double sum=d==0?b[o]:0;
    for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int yy=y+ky-1,z=xx+kx-1;if(yy<0||yy>=h||z<0||z>=h)continue;
     for(int c=0;c<32;++c)sum+=double(x[(d*h*h+yy*h+z)*32+c])*w[((ky*3+kx)*32+c)*32+o];}
    largest=std::max(largest,std::abs(sum-out[(d*h*h+y*h+xx)*32+o]));}
   std::cout<<"Winograd F="<<tile<<" D="<<D<<" H="<<h<<" error="<<largest<<std::endl;
   // F4 has a less well-conditioned transform than F2. This is an operator
   // regression bound for N(0,.1) fixtures, not the unchanged e/R mean gate.
   if(largest>(tile==4?1e-5:2e-6))throw std::runtime_error("Winograd accuracy");
  }
 };
 test(std::integral_constant<int,3>{});test(std::integral_constant<int,5>{});
#endif
}
