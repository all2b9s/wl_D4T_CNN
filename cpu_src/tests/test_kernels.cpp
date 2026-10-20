// Standalone checks of every SIMD kernel on every available ISA against
// independent double-precision references. Exit 77 when no SIMD ISA exists.
#include "d4cnn/convolution.h"
#include "d4cnn/kernels.h"
#include "d4cnn/parallel.h"
#include "d4cnn/winograd.h"
#include "d4cnn/winograd4.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using d4cnn::Epi;using d4cnn::EpiMode;
static std::mt19937 rng(67);
static std::vector<float> normal(size_t n,float scale=.1f){std::normal_distribution<float> g(0,scale);std::vector<float> v(n);for(auto& x:v)x=g(rng);return v;}
static void check(bool ok,const std::string& what){if(!ok)throw std::runtime_error(what);}
static double gelu(double v,double* df){double phi=.5*(1+std::erf(v/std::sqrt(2.)));*df=phi+v*std::exp(-.5*v*v)/std::sqrt(2*std::acos(-1.));return v*phi;}
// Double reference of one epilogue pixel for dual component d and channel j.
static double epilogue_ref(const float* in,int istride,const Epi& e,int p,int d,int j){
 double mu=0,var=0;for(int c=0;c<32;++c)mu+=in[c];mu/=32;
 for(int c=0;c<32;++c)var+=(in[c]-mu)*(in[c]-mu);
 const double inv=1/std::sqrt(var/32+double(1e-5f));
 const double v=(in[j]-mu)*inv*e.gamma[j]+e.beta[j];double df,out=gelu(v,&df);
 if(d){double dm=0,cov=0;for(int c=0;c<32;++c)dm+=in[d*istride+c];dm/=32;
  for(int c=0;c<32;++c)cov+=(in[c]-mu)*(in[d*istride+c]-dm);cov/=32;
  out=e.gamma[j]*inv*(in[d*istride+j]-dm-(in[j]-mu)*cov*inv*inv)*df;}
 return out+double(e.residual)*(e.skip?double(e.skip[j])*e.x[size_t(d)*e.xstride+p]:e.x[size_t(d)*e.xstride+size_t(p)*32+j]);
}
static double conv_ref(const std::vector<float>& x,int ci,const std::vector<float>& w,const std::vector<float>& b,int h,int d,int y,int xx,int o){
 double sum=d==0?b[o]:0;
 for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int yy=y+ky-1,z=xx+kx-1;if(yy<0||yy>=h||z<0||z>=h)continue;
  for(int c=0;c<ci;++c)sum+=double(x[(size_t(d)*h*h+yy*h+z)*ci+c])*w[((ky*3+kx)*ci+c)*32+o];}
 return sum;
}
static void test_isa(const std::string& isa){
 const auto k=d4cnn::select_kernels(isa);check(k.gemm!=nullptr,"missing kernels "+isa);
 // GEMM: every row tail, column tails and K values used by the model.
 for(int m:{1,2,5,6,7,11,12,13,24,25,40,77})for(int n:{1,8,16,32,64,96})for(int kk:{9,32,34,64}){
  auto a=normal(size_t(m)*kk),b=normal(size_t(kk)*n);std::vector<float> c(size_t(m)*n,-1);
  k.gemm(a.data(),kk,b.data(),n,c.data(),n,m,n,kk);
  for(int r=0;r<m;++r)for(int col=0;col<n;++col){double s=0;for(int i=0;i<kk;++i)s+=double(a[r*kk+i])*b[i*n+col];
   check(std::abs(s-c[r*n+col])<1e-5,isa+" gemm m="+std::to_string(m)+" n="+std::to_string(n));}
 }
 for(int dual:{3,5}){
  auto gamma=normal(32,1),beta=normal(32),skip=normal(32);
  for(auto& g:gamma)g+=1.f;
  // Winograd raw output against direct double convolution.
  for(int tile:{2,4})for(int h:{1,2,3,4,5,8,9,16,17}){
   const int stride=h*h*32;
   auto x=normal(size_t(dual)*stride),w=normal(9*32*32),b=normal(32);
   auto tw=tile==4?d4cnn::winograd4_weights(w):d4cnn::winograd_weights(w);
   std::vector<float> scratch(d4cnn::kernel_scratch(dual,h,tile,0)),raw(x.size()),fused(x.size()),plane;
   Epi e;e.mode=EpiMode::raw;
   k.winograd(tile,dual,x.data(),tw.data(),b.data(),raw.data(),h,e,scratch.data());
   double largest=0;
   for(int d=0;d<dual;++d)for(int y=0;y<h;++y)for(int xx=0;xx<h;++xx)for(int o=0;o<32;++o)
    largest=std::max(largest,std::abs(conv_ref(x,32,w,b,h,d,y,xx,o)-raw[(size_t(d)*h*h+y*h+xx)*32+o]));
   check(largest<(tile==4?1e-5:2e-6),isa+" winograd F"+std::to_string(tile)+" h="+std::to_string(h));
   // Fused store == raw + in-place epilogue plane.
   e.mode=EpiMode::store;e.gamma=gamma.data();e.beta=beta.data();e.residual=.1f;e.x=x.data();e.xstride=stride;
   k.winograd(tile,dual,x.data(),tw.data(),b.data(),fused.data(),h,e,scratch.data());
   plane=raw;k.epilogue_plane(dual,plane.data(),h*h,e);
   for(size_t i=0;i<plane.size();++i)check(std::abs(plane[i]-fused[i])<1e-6,isa+" fused winograd");
   // Epilogue against the double reference (on the raw conv output).
   for(int p=0;p<h*h;++p)for(int d=0;d<dual;++d)for(int j=0;j<32;++j){
    const double ref=epilogue_ref(raw.data()+size_t(p)*32,stride,e,p,d,j);
    check(std::abs(ref-plane[(size_t(d)*h*h+p)*32+j])<3e-6*(1+std::abs(ref)),isa+" epilogue");}
   // Sinks: pool and canonical accumulate exactly what store mode writes.
   std::vector<float> pooled(size_t(dual)*32,0.f),pool_w=normal(h*h);
   Epi pe=e;pe.mode=EpiMode::pool;pe.pooled=pooled.data();pe.pool_w=pool_w.data();
   k.winograd(tile,dual,x.data(),tw.data(),b.data(),raw.data(),h,pe,scratch.data());
   for(int d=0;d<dual;++d)for(int j=0;j<32;++j){double s=0;for(int p=0;p<h*h;++p)s+=double(fused[(size_t(d)*h*h+p)*32+j])*pool_w[p];
    check(std::abs(s-pooled[d*32+j])<1e-5,isa+" pool sink");}
   const int ntok=std::max(1,h*h/4);std::vector<int> canon(h*h),token(h*h);
   for(int p=0;p<h*h;++p){canon[p]=h*h-1-p;token[p]=canon[p]%ntok;}
   std::vector<float> sc(size_t(dual)*ntok*32,0.f),e1(size_t(dual)*h*h*32,0.f),e2=e1;
   Epi ce=e;ce.mode=EpiMode::canonical;ce.scalar=sc.data();ce.e1=e1.data();ce.e2=e2.data();ce.canon=canon.data();
   ce.token=token.data();ce.ntok=ntok;ce.pixels=h*h;ce.token_scale=.25f;ce.sign1=1;ce.sign2=-1;
   k.winograd(tile,dual,x.data(),tw.data(),b.data(),raw.data(),h,ce,scratch.data());
   std::vector<double> sref(sc.size(),0.);
   for(int d=0;d<dual;++d)for(int p=0;p<h*h;++p)for(int j=0;j<32;++j){
    const double v=fused[(size_t(d)*h*h+p)*32+j]/8.;
    sref[(size_t(d)*ntok+token[p])*32+j]+=v*.25;
    check(std::abs(e1[(size_t(d)*h*h+canon[p])*32+j]-v)<1e-6&&std::abs(e2[(size_t(d)*h*h+canon[p])*32+j]+v)<1e-6,isa+" canonical sink");}
   for(size_t i=0;i<sc.size();++i)check(std::abs(sc[i]-sref[i])<1e-6,isa+" token sink");
  }
  // First layer (C1) with projected skip.
  for(int h:{1,3,8,13}){
   auto x=normal(size_t(dual)*h*h),w=normal(9*32),b=normal(32);
   std::vector<float> scratch(d4cnn::kernel_scratch(dual,h,4,0)),raw(size_t(dual)*h*h*32),out(raw.size());
   Epi e;e.mode=EpiMode::raw;
   k.first_layer(dual,x.data(),w.data(),b.data(),raw.data(),h,e,scratch.data());
   for(int d=0;d<dual;++d)for(int y=0;y<h;++y)for(int xx=0;xx<h;++xx)for(int o=0;o<32;++o)
    check(std::abs(conv_ref(x,1,w,b,h,d,y,xx,o)-raw[(size_t(d)*h*h+y*h+xx)*32+o])<1e-6,isa+" first layer");
   e.mode=EpiMode::store;e.gamma=gamma.data();e.beta=beta.data();e.residual=.1f;e.x=x.data();e.xstride=h*h;e.skip=skip.data();
   k.first_layer(dual,x.data(),w.data(),b.data(),out.data(),h,e,scratch.data());
   for(int p=0;p<h*h;++p)for(int d=0;d<dual;++d)for(int j=0;j<32;++j){
    const double ref=epilogue_ref(raw.data()+size_t(p)*32,h*h*32,e,p,d,j);
    check(std::abs(ref-out[(size_t(d)*h*h+p)*32+j])<3e-6*(1+std::abs(ref)),isa+" first layer epilogue");}
  }
  // Row LayerNorm, GELU and dual softmax.
  {const int rows=7;auto x=normal(size_t(dual)*rows*32,1),y=x;
   k.norm_rows(dual,y.data(),rows,rows*32,gamma.data(),beta.data());
   for(int p=0;p<rows;++p){const float* in=x.data()+p*32;
    double mu=0,var=0;for(int c=0;c<32;++c)mu+=in[c];mu/=32;for(int c=0;c<32;++c)var+=(in[c]-mu)*(in[c]-mu);
    const double inv=1/std::sqrt(var/32+double(1e-5f));
    for(int d=0;d<dual;++d){double dm=0,cov=0;
     if(d){for(int c=0;c<32;++c)dm+=in[d*rows*32+c];dm/=32;for(int c=0;c<32;++c)cov+=(in[c]-mu)*(in[d*rows*32+c]-dm);cov/=32;}
     for(int j=0;j<32;++j){double ref=d?gamma[j]*inv*(in[d*rows*32+j]-dm-(in[j]-mu)*cov*inv*inv):(in[j]-mu)*inv*gamma[j]+beta[j];
      check(std::abs(ref-y[(size_t(d)*rows+p)*32+j])<2e-6*(1+std::abs(ref)),isa+" norm rows");}}}}
  {const int count=30001;std::vector<float> x(size_t(dual)*count);
   for(int i=0;i<count;++i){x[i]=-15.f+i*.001f;for(int d=1;d<dual;++d)x[size_t(d)*count+i]=float(d);}
   auto y=x;k.gelu(dual,y.data(),count,count);double mv=0,md=0;
   for(int i=0;i<count;++i){double df,ref=gelu(x[i],&df);mv=std::max(mv,std::abs(ref-y[i]));
    for(int d=1;d<dual;++d)md=std::max(md,std::abs(df*d-y[size_t(d)*count+i])/d);}
   std::cout<<isa<<" D="<<dual<<" GELU max error value "<<mv<<" derivative "<<md<<std::endl;
   check(mv<3e-6&&md<5e-7,isa+" GELU accuracy");}
  for(int cols:{1,5,16,37,256,4096}){const int rows=3;auto x=normal(size_t(dual)*rows*cols,2),y=x;
   k.softmax_rows(dual,y.data(),rows,cols,rows*cols);
   for(int r=0;r<rows;++r){const float* in=x.data()+r*cols;double mx=*std::max_element(in,in+cols),sum=0;
    std::vector<double> s(cols);for(int c=0;c<cols;++c){s[c]=std::exp(in[c]-mx);sum+=s[c];}for(auto& v:s)v/=sum;
    for(int d=0;d<dual;++d){double dot=0;if(d)for(int c=0;c<cols;++c)dot+=s[c]*in[size_t(d)*rows*cols+c];
     for(int c=0;c<cols;++c){double ref=d?s[c]*(in[size_t(d)*rows*cols+c]-dot):s[c];
      check(std::abs(ref-y[(size_t(d)*rows+r)*cols+c])<2e-6,isa+" softmax");}}}}
  // Attention context against a double reference (all tangents).
  for(int n:{9,40}){const int dim=32,heads=4,hd=8;auto qkv=normal(size_t(dual)*n*3*dim,1);
   std::vector<float> ctx(size_t(dual)*n*dim),scratch(d4cnn::kernel_scratch(dual,1,2,n));
   k.attend(dual,qkv.data(),n,dim,heads,ctx.data(),scratch.data());
   auto at=[&](int d,int p,int c){return double(qkv[(size_t(d)*n+p)*3*dim+c]);};
   for(int head=0;head<heads;++head)for(int p=0;p<n;++p){
    std::vector<double> s(n),ds((dual-1)*n);double mx=-1e300,sum=0;
    for(int q=0;q<n;++q){double v=0;for(int j=0;j<hd;++j)v+=at(0,p,head*hd+j)*at(0,q,dim+head*hd+j);s[q]=v/std::sqrt(8.);mx=std::max(mx,s[q]);}
    for(auto& v:s){v=std::exp(v-mx);sum+=v;}for(auto& v:s)v/=sum;
    for(int d=1;d<dual;++d){double dot=0;
     for(int q=0;q<n;++q){double v=0;for(int j=0;j<hd;++j){int c=head*hd+j;v+=at(d,p,c)*at(0,q,dim+c)+at(0,p,c)*at(d,q,dim+c);}
      ds[(d-1)*n+q]=v/std::sqrt(8.);dot+=s[q]*ds[(d-1)*n+q];}
     for(int q=0;q<n;++q)ds[(d-1)*n+q]=s[q]*(ds[(d-1)*n+q]-dot);}
    for(int d=0;d<dual;++d)for(int j=0;j<hd;++j){int c=head*hd+j;double ref=0;
     for(int q=0;q<n;++q)ref+=d?ds[(d-1)*n+q]*at(0,q,2*dim+c)+s[q]*at(d,q,2*dim+c):s[q]*at(0,q,2*dim+c);
     check(std::abs(ref-ctx[(size_t(d)*n+p)*dim+c])<2e-6,isa+" attention");}
   }}
 }
 std::cout<<isa<<" kernels passed (gemm block "<<k.gemm_mr<<"x"<<k.gemm_nr<<")"<<std::endl;
}
int main(){
 // D4CNN_TEST_NO_THREADS=1 skips the executor checks (e.g. under qemu-user
 // on 64 KB-page hosts, where guest thread creation is unsupported).
 const char* no_threads=std::getenv("D4CNN_TEST_NO_THREADS");
 if(!(no_threads&&std::string(no_threads)=="1"))for(int threads:{2,4,2}){
  std::vector<int> values(73);
  for(int pass=0;pass<10;++pass)d4cnn::parallel_for(73,threads,[&](int i){values[i]+=1;});
  for(auto v:values)check(v==10,"executor lost a task");
  bool caught=false;
  try{d4cnn::parallel_for(73,threads,[](int i){if(i==3)throw std::runtime_error("test");});}
  catch(const std::runtime_error&){caught=true;}
  check(caught,"executor lost exception");
  d4cnn::parallel_for(73,threads,[&](int i){values[i]=i;});
  for(int i=0;i<73;++i)check(values[i]==i,"executor recovery");
 }
 std::cout<<"Executor reuse, resize and exception recovery passed"<<std::endl;
 bool exercised=false;
 for(const auto& isa:d4cnn::available_isas()){if(isa=="scalar")continue;exercised=true;test_isa(isa);}
 std::cout<<"auto resolves to "<<d4cnn::resolve_isa("auto")<<std::endl;
 if(!exercised){std::cout<<"SKIP: no SIMD ISA available"<<std::endl;return 77;}
 std::cout<<"All SIMD kernel checks passed"<<std::endl;
}
