#pragma once
#include "convolution.h"
#include "kernels.h"
#include "profile.h"
#include "winograd.h"
#include "winograd4.h"
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>
namespace d4cnn {
struct Config {
 int size=64, channels=32, layers=5, head_depth=3, grid=16, dim=32, heads=4;
 double residual=.1, nl_sigma=16, gw_sigma=16, r_cut=24;
 int order=8;
 bool attention=false, shared=false, suppress=true;
};
struct Weight {std::vector<double> values; std::vector<int> shape;};
using Weights=std::map<std::string,Weight>;
template<class S,int D> struct Tensor {
 int p=0,c=0; std::vector<S> a;
 Tensor()=default;
 Tensor(int p_,int c_):p(p_),c(c_),a(size_t(D)*p_*c_,S(0)){}
 void resize(int pixels,int channels,bool clear=false){
  p=pixels;c=channels;a.resize(size_t(D)*p*c);
  if(clear)std::fill(a.begin(),a.end(),S(0));
 }
 S& at(int d,int i,int j=0){return a[(size_t(d)*p+i)*c+j];}
 S at(int d,int i,int j=0)const{return a[(size_t(d)*p+i)*c+j];}
};
template<class S> std::vector<S> gaussian(int h,double sigma,bool suppress,double radius,int order) {
 if(h<1||!std::isfinite(sigma)||sigma<=0||!std::isfinite(radius)||radius<=0||order<2||order%2)
  throw std::invalid_argument("Invalid Gaussian map configuration");
 std::vector<S> out(h*h);
 for(int y=0;y<h;++y)for(int x=0;x<h;++x){
  // Use model precision, as in torch's runtime-generated maps.
  S yy=S(y)-S(h-1)/S(2),xx=S(x)-S(h-1)/S(2),r2=xx*xx+yy*yy;
  S g=std::exp(S(-.5)*r2/S(sigma*sigma))/S(2*std::acos(-1.)*sigma*sigma);
  if(suppress){S u=r2/S(radius*radius),v=1;for(int k=0;k<order/2;++k)v*=u;g*=std::exp(-v);}
  out[y*h+x]=g;
 }
 auto src=out;
 for(int y=0;y<h;++y)for(int x=0;x<h;++x)
  out[y*h+x]=(src[y*h+x]+src[(h-1-y)*h+h-1-x]+src[(h-1-y)*h+x]+src[y*h+h-1-x])/S(4);
 return out;
}
template<class S,int D> class Model {
 using X=Tensor<S,D>;
 struct Workspace {X normalized,input,buffers[2],scalar,e1,e2;std::vector<float> scratch,pooled;};
 Config cfg; std::map<std::string,std::vector<S>> w;
 std::vector<std::vector<S>> packed;
 std::vector<std::vector<float>> transformed;
 // Transposed [in][out] float copies of the attention linear layers (GEMM B operands).
 std::map<std::string,std::vector<float>> transposed;
 bool use_winograd=false;
 bool use_winograd4=false;
 bool use_simd=false;
 // Fused C32 trunk through the ISA kernel table (float32, SIMD ISA, 32 channels).
 bool fast=false;
 std::vector<S> norm_map,pool_map;
 // D4 tables: canon[g*p+i] is the canonical pixel of orbit g's pixel i;
 // orbit_token[g*p+i] the low-resolution token of that canonical pixel; pool_w
 // the fixW pooling weight per pixel (the Gaussian is D4 symmetric).
 std::vector<int> canon,orbit_token;
 std::vector<float> pool_w;
 Conv conv;
 Kernels k;
 const std::vector<S>& get(const std::string& key)const {
  auto i=w.find(key);if(i==w.end())throw std::invalid_argument("Missing parameter: "+key);return i->second;
 }
 void save(std::map<std::string,X>* trace,const std::string& name,const X& x)const{if(trace)(*trace)[name]=x;}
 X linear(const X& x,const std::string& key,int out,bool bias=true)const{
  if constexpr(std::is_same_v<S,float>){
   auto t=transposed.find(key);
   if(t!=transposed.end()&&out%k.gemm_nr==0){
    X y(x.p,out);k.gemm(x.a.data(),x.c,t->second.data(),out,y.a.data(),out,D*x.p,out,x.c);
    auto bi=w.find(key+".bias");
    if(bias&&bi!=w.end())for(int p=0;p<x.p;++p)for(int o=0;o<out;++o)y.at(0,p,o)+=bi->second[o];
    return y;
   }
  }
  X y(x.p,out);const auto& ww=get(key+".weight");
  auto bi=w.find(key+".bias");
  for(int d=0;d<D;++d)for(int p=0;p<x.p;++p)for(int o=0;o<out;++o){
   S v=(d==0&&bias&&bi!=w.end())?bi->second[o]:S(0);
   for(int i=0;i<x.c;++i)v+=x.at(d,p,i)*ww[o*x.c+i];
   y.at(d,p,o)=v;
  }return y;
 }
 void layernorm(X& x,const std::string& key)const{
  const auto& gamma=get(key+".weight");const auto& beta=get(key+".bias");
  if constexpr(std::is_same_v<S,float>){if(use_simd&&x.c==32){
   k.norm_rows(D,x.a.data(),x.p,x.p*32,gamma.data(),beta.data());return;
  }}
  for(int p=0;p<x.p;++p){
   S mu=0,var=0;for(int j=0;j<x.c;++j)mu+=x.at(0,p,j);mu/=S(x.c);
   for(int j=0;j<x.c;++j){S a=x.at(0,p,j)-mu;var+=a*a;}var/=S(x.c);
   S inv=S(1)/std::sqrt(var+S(1e-5));
   for(int d=1;d<D;++d){
    S dm=0,cov=0;for(int j=0;j<x.c;++j)dm+=x.at(d,p,j);dm/=S(x.c);
    for(int j=0;j<x.c;++j) { cov+=(x.at(0,p,j)-mu)*(x.at(d,p,j)-dm); }
    cov/=S(x.c);
    for(int j=0;j<x.c;++j)x.at(d,p,j)=gamma[j]*inv*(x.at(d,p,j)-dm-(x.at(0,p,j)-mu)*cov*inv*inv);
   }
   for(int j=0;j<x.c;++j)x.at(0,p,j)=(x.at(0,p,j)-mu)*inv*gamma[j]+beta[j];
  }
 }
 void activate(X& x,bool tanh=false)const{
  if constexpr(std::is_same_v<S,float>){if(use_simd&&!tanh){k.gelu(D,x.a.data(),x.p*x.c,x.p*x.c);return;}}
  const S invroot2=S(1/std::sqrt(2.)),invroot2pi=S(1/std::sqrt(2*std::acos(-1.)));
  for(int p=0;p<x.p;++p)for(int c=0;c<x.c;++c){
   S v=x.at(0,p,c),f,df;
   if(tanh){f=std::tanh(v);df=S(1)-f*f;}
   else {S phi=S(.5)*(S(1)+std::erf(v*invroot2));f=v*phi;df=phi+v*std::exp(S(-.5)*v*v)*invroot2pi;}
   for(int d=1;d<D;++d) { x.at(d,p,c)*=df; }
   x.at(0,p,c)=f;
  }
 }
 void add(X& x,const X& y,S scale=S(1))const{for(size_t i=0;i<x.a.size();++i)x.a[i]+=scale*y.a[i];}
 void softmax(X& x)const{
  if constexpr(std::is_same_v<S,float>){if(use_simd){k.softmax_rows(D,x.a.data(),x.p,x.c,x.p*x.c);return;}}
  for(int p=0;p<x.p;++p){
   S mx=x.at(0,p,0),sum=0;for(int c=1;c<x.c;++c)mx=std::max(mx,x.at(0,p,c));
   for(int c=0;c<x.c;++c){x.at(0,p,c)=std::exp(x.at(0,p,c)-mx);sum+=x.at(0,p,c);}
   for(int c=0;c<x.c;++c)x.at(0,p,c)/=sum;
   for(int d=1;d<D;++d){S dot=0;for(int c=0;c<x.c;++c)dot+=x.at(0,p,c)*x.at(d,p,c);
    for(int c=0;c<x.c;++c)x.at(d,p,c)=x.at(0,p,c)*(x.at(d,p,c)-dot);}
  }
 }
 // Base epilogue arguments for one trunk block reading x.
 Epi epilogue(const X& x,int layer)const{
  std::string key="blocks."+std::to_string(layer);Epi e;
  if constexpr(std::is_same_v<S,float>){
   e.gamma=get(key+".norm.weight").data();e.beta=get(key+".norm.bias").data();
   e.residual=float(cfg.residual);e.x=x.a.data();e.xstride=x.p*x.c;
   if(x.c!=cfg.channels)e.skip=get(key+".skip.weight").data();
  }
  return e;
 }
 // Convolution only (bias included), through the same kernels as the fused path.
 void convolve(const X& x,X& y,int layer,float* scratch)const{
  int h=cfg.size,co=cfg.channels;std::string key="blocks."+std::to_string(layer);
  const auto& b=get(key+".conv.bias");const auto& ww=packed[layer];
  if constexpr(std::is_same_v<S,float>){
   if(fast){Epi raw;raw.mode=EpiMode::raw;
    if(layer==0)k.first_layer(D,x.a.data(),ww.data(),b.data(),y.a.data(),h,raw,scratch);
    else if(use_winograd)k.winograd(use_winograd4?4:2,D,x.a.data(),transformed[layer].data(),b.data(),y.a.data(),h,raw,scratch);
    else conv(x.a.data(),ww.data(),b.data(),y.a.data(),h,x.c,co,D);
    return;
   }
   conv(x.a.data(),ww.data(),b.data(),y.a.data(),h,x.c,co,D);
  }
  else {
   for(int d=0;d<D;++d)for(int p=0;p<h*h;++p)for(int o=0;o<co;++o){
    S v=d==0?b[o]:S(0);
    for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){int yy=p/h+ky-1,xx=p%h+kx-1;
     if(yy<0||xx<0||yy>=h||xx>=h)continue;
     for(int i=0;i<x.c;++i)v+=x.at(d,yy*h+xx,i)*ww[((ky*3+kx)*x.c+i)*co+o];}
    y.at(d,p,o)=v;
   }
  }
 }
 // One ConvGELU_Res block. The fused path runs conv+LN+GELU+residual in one
 // kernel; `sink` (last layer only) receives the output instead of y.
 void block(const X& x,X& y,int layer,std::map<std::string,X>* trace,float* scratch,const Epi* sink=nullptr)const{
  int h=cfg.size,co=cfg.channels;std::string key="blocks."+std::to_string(layer);
  if constexpr(std::is_same_v<S,float>){
   if(fast&&!trace){
#ifdef D4CNN_PROFILE
    prof::Scope clock(layer==0?prof::first_layer:prof::hidden);
#endif
    Epi e=sink?*sink:epilogue(x,layer);
    if(!sink)e.mode=EpiMode::store;
    const auto& b=get(key+".conv.bias");
    if(layer==0)k.first_layer(D,x.a.data(),packed[0].data(),b.data(),y.a.data(),h,e,scratch);
    else if(use_winograd)k.winograd(use_winograd4?4:2,D,x.a.data(),transformed[layer].data(),b.data(),y.a.data(),h,e,scratch);
    else {conv(x.a.data(),packed[layer].data(),b.data(),y.a.data(),h,x.c,co,D);k.epilogue_plane(D,y.a.data(),h*h,e);}
    return;
   }
  }
  convolve(x,y,layer,scratch);
  save(trace,key+".conv",y);layernorm(y,key+".norm");save(trace,key+".norm",y);
  activate(y);save(trace,key+".gelu",y);
  if(x.c==co)add(y,x,S(cfg.residual));else {X skip=linear(x,key+".skip",co,false);add(y,skip,S(cfg.residual));}
  save(trace,key+".out",y);
 }
 int orbit_src(int orbit,int p)const{
  int h=cfg.size,y=p/h,x=p%h;if(orbit>=4)y=h-1-y;
  for(int r=0;r<orbit%4;++r){int t=y;y=x;x=h-1-t;}return y*h+x;
 }
 X attention_kernel(const X& field,std::map<std::string,X>* trace,float* scratch,bool prepooled=false)const{
  D4_PROF(attention);
  int h=cfg.size,s=std::min(cfg.grid,h),n=s*s,dim=cfg.dim,hd=dim/cfg.heads;
  X tokens(n,field.c+2);
  for(int oy=0;oy<s;++oy)for(int ox=0;ox<s;++ox){
   int y0=oy*h/s,y1=((oy+1)*h+s-1)/s,x0=ox*h/s,x1=((ox+1)*h+s-1)/s,p=oy*s+ox;
   for(int d=0;d<D;++d)for(int c=0;c<field.c;++c){
    if(prepooled)tokens.at(d,p,c)=field.at(d,p,c);
    else {S v=0;for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x)v+=field.at(d,y*h+x,c);
     tokens.at(d,p,c)=v/S((y1-y0)*(x1-x0));}}
   S scale=std::max(S(s-1)/S(2),S(1)),xx=(S(ox)-S(s-1)/S(2))/scale,yy=(S(oy)-S(s-1)/S(2))/scale;
   tokens.at(0,p,field.c)=xx*xx+yy*yy;tokens.at(0,p,field.c+1)=xx*xx*yy*yy;
  }
  tokens=linear(tokens,"token_projection",dim);save(trace,"tokens",tokens);
  X ln=tokens;layernorm(ln,"attn_norm");X qkv=linear(ln,"qkv",3*dim),context(n,dim);
  bool vector_heads=false;
  if constexpr(std::is_same_v<S,float>)vector_heads=use_simd&&k.head_width>0&&hd%k.head_width==0;
  if(vector_heads){D4_PROF(attention_core);if constexpr(std::is_same_v<S,float>)k.attend(D,qkv.a.data(),n,dim,cfg.heads,context.a.data(),scratch);}
  else for(int head=0;head<cfg.heads;++head)for(int first=0;first<n;first+=16){
   int rows=std::min(16,n-first);X scores(rows,n);S factor=S(1/std::sqrt(double(hd)));
   for(int row=0;row<rows;++row)for(int q=0;q<n;++q){int p=first+row;
    S v=0;for(int j=0;j<hd;++j){int c=head*hd+j;v+=qkv.at(0,p,c)*qkv.at(0,q,dim+c);}scores.at(0,row,q)=v*factor;
    for(int d=1;d<D;++d){S dv=0;for(int j=0;j<hd;++j){int c=head*hd+j;
     dv+=qkv.at(d,p,c)*qkv.at(0,q,dim+c)+qkv.at(0,p,c)*qkv.at(d,q,dim+c);}scores.at(d,row,q)=dv*factor;}
   }
   softmax(scores);
   for(int row=0;row<rows;++row){int p=first+row;
    for(int q=0;q<n;++q)for(int d=0;d<D;++d){
     S sv=scores.at(0,row,q),sd=scores.at(d,row,q);
     for(int j=0;j<hd;++j){int c=head*hd+j;
      if(d==0)context.at(0,p,c)+=sv*qkv.at(0,q,2*dim+c);
      else context.at(d,p,c)+=sd*qkv.at(0,q,2*dim+c)+sv*qkv.at(d,q,2*dim+c);}}
   }
  }
  add(tokens,linear(context,"attn_output",dim));save(trace,"attn_residual",tokens);
  ln=tokens;layernorm(ln,"ffn_norm");X ffn=linear(ln,"ffn.0",2*dim);activate(ffn);add(tokens,linear(ffn,"ffn.2",dim));
  X logits=linear(tokens,"kernel_head",1);save(trace,"kernel_low",logits);X up(h*h,1);
  for(int y=0;y<h;++y)for(int x=0;x<h;++x){
   S fy=std::max(S(0),(S(y)+S(.5))*S(s)/S(h)-S(.5)),fx=std::max(S(0),(S(x)+S(.5))*S(s)/S(h)-S(.5));
   int y0=std::min(int(fy),s-1),x0=std::min(int(fx),s-1),y1=std::min(y0+1,s-1),x1=std::min(x0+1,s-1);
   S ay=fy-S(y0),ax=fx-S(x0);
   for(int d=0;d<D;++d)up.at(d,y*h+x)=(S(1)-ay)*((S(1)-ax)*logits.at(d,y0*s+x0)+ax*logits.at(d,y0*s+x1))+
     ay*((S(1)-ax)*logits.at(d,y1*s+x0)+ax*logits.at(d,y1*s+x1));
  }
  X blurred(h*h,1);const auto& blur=get("kernel_blur");
  for(int d=0;d<D;++d)for(int y=0;y<h;++y)for(int x=0;x<h;++x){S v=0;
   for(int ky=0;ky<5;++ky)for(int kx=0;kx<5;++kx)v+=up.at(d,std::clamp(y+ky-2,0,h-1)*h+std::clamp(x+kx-2,0,h-1))*blur[ky*5+kx];
   blurred.at(d,y*h+x)=v;}
  if(cfg.suppress)for(int y=0;y<h;++y)for(int x=0;x<h;++x){S yy=S(y)-S(h-1)/2,xx=S(x)-S(h-1)/2;
   S u=(xx*xx+yy*yy)/S(cfg.r_cut*cfg.r_cut),v=1;for(int r=0;r<cfg.order/2;++r)v*=u;blurred.at(0,y*h+x)-=v;}
  save(trace,"kernel_logits",blurred);
  // Softmax across pixels, rather than channels.
  X flat(1,h*h);flat.a=blurred.a;softmax(flat);flat.p=h*h;flat.c=1;save(trace,"kernel",flat);return flat;
 }
 X head(X x,const std::string& key)const{
  D4_PROF(heads);
  for(int l=0;l<cfg.head_depth;++l){x=linear(x,key+".net."+std::to_string(1+2*l),head_width(key,l),false);activate(x,true);}
  return linear(x,key+".net."+std::to_string(1+2*cfg.head_depth),1,false);
 }
 int head_width(const std::string& key,int l)const{
  int in=l==0?cfg.channels:head_width(key,l-1);return int(get(key+".net."+std::to_string(1+2*l)+".weight").size())/in;
 }
public:
 Model(Config c,const Weights& weights,const std::string& isa):cfg(c),conv(select_conv(isa)){
  std::string algorithm=std::getenv("D4CNN_CONV")?std::getenv("D4CNN_CONV"):"auto";
  if(algorithm!="auto"&&algorithm!="direct"&&algorithm!="winograd"&&algorithm!="winograd4")throw std::invalid_argument("D4CNN_CONV must be auto, direct, winograd or winograd4");
  if constexpr(std::is_same_v<S,float>){if(resolve_isa(isa)!="scalar")k=select_kernels(isa);}
  use_simd=k.gemm!=nullptr;
  fast=use_simd&&c.channels==32;
  use_winograd=fast&&algorithm!="direct";
  use_winograd4=use_winograd&&(algorithm=="auto"||algorithm=="winograd4");
  if((algorithm=="winograd"||algorithm=="winograd4")&&!use_winograd)
   throw std::invalid_argument("Winograd requires float32, C32 and NEON/AVX2/AVX512");
  for(auto& [key,value]:weights)w[key]=std::vector<S>(value.values.begin(),value.values.end());
  norm_map=gaussian<S>(c.size,c.nl_sigma,c.suppress,c.r_cut,c.order);
  if(!c.attention)pool_map=gaussian<S>(c.size,c.gw_sigma,c.suppress,24,8); // fixW class uses Gaussian defaults
  for(int l=0;l<c.layers;++l){int ci=l==0?1:c.channels,co=c.channels;const auto& raw=get("blocks."+std::to_string(l)+".conv.weight");
   std::vector<S> pack(9*ci*co);for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx)for(int i=0;i<ci;++i)for(int o=0;o<co;++o) {
    pack[((ky*3+kx)*ci+i)*co+o]=raw[((o*ci+i)*3+ky)*3+kx];
   }
   packed.push_back(std::move(pack));
   transformed.emplace_back();
   if constexpr(std::is_same_v<S,float>){if(use_winograd&&l>0)transformed.back()=use_winograd4?winograd4_weights(packed.back()):winograd_weights(packed.back());}
  }
  const int h=c.size,p=h*h;canon.resize(8*size_t(p));
  for(int g=0;g<8;++g)for(int i=0;i<p;++i)canon[size_t(g)*p+i]=orbit_src(g,i);
  if(c.attention&&h%std::min(c.grid,h)==0){const int grid=std::min(c.grid,h),side=h/grid;orbit_token.resize(canon.size());
   for(size_t i=0;i<canon.size();++i)orbit_token[i]=(canon[i]/h/side)*grid+(canon[i]%h/side);}
  if(!c.attention)if constexpr(std::is_same_v<S,float>){
   S sum=std::accumulate(pool_map.begin(),pool_map.end(),S(0));pool_w.resize(p);
   for(int i=0;i<p;++i)pool_w[i]=pool_map[i]/sum/S(8);
  }
  if constexpr(std::is_same_v<S,float>){if(use_simd&&c.attention)
   for(auto key:{"token_projection","qkv","attn_output","ffn.0","ffn.2"}){
    const auto& m=get(std::string(key)+".weight");int out=weights.at(std::string(key)+".weight").shape[0],in=int(m.size())/out;
    std::vector<float> t(m.size());for(int o=0;o<out;++o)for(int i=0;i<in;++i)t[size_t(i)*out+o]=m[size_t(o)*in+i];
    transposed[key]=std::move(t);}}
 }
 std::string algorithm()const{return use_winograd?(use_winograd4?"winograd_f4":"winograd_f2"):"direct";}
 std::array<S,D*2> measure(const S* q,std::map<std::string,X>* trace=nullptr)const{
  // Scratch is private to each worker; all accumulation fields are reset below.
  D4_PROF(total);
  thread_local Workspace ws;
  int h=cfg.size,p=h*h,c=cfg.channels;
  if(use_simd)ws.scratch.resize(kernel_scratch(D,h,use_winograd4?4:2,cfg.attention?std::min(cfg.grid,h)*std::min(cfg.grid,h):0));
  float* scratch=ws.scratch.data();
  X& normalized=ws.normalized;normalized.resize(p,1);std::array<S,D> m{},norm{};
  for(int d=0;d<D;++d){for(int i=0;i<p;++i)m[d]+=q[d*p+i]*norm_map[i];m[d]*=S(100);}
  S z=m[0]-S(1);norm[0]=S(1)+(S(10)*z>S(20)?z:std::log1p(std::exp(S(10)*z))/S(10));
  S sig=S(10)*z>S(20)?S(1):(z>=0?S(1)/(S(1)+std::exp(-S(10)*z)):std::exp(S(10)*z)/(S(1)+std::exp(S(10)*z)));
  for(int d=1;d<D;++d)norm[d]=sig*m[d];
  for(int i=0;i<p;++i){normalized.at(0,i)=q[i]/norm[0];for(int d=1;d<D;++d)normalized.at(d,i)=(q[d*p+i]-normalized.at(0,i)*norm[d])/norm[0];}
  save(trace,"normalized",normalized);
  X& input=ws.input;auto& buffers=ws.buffers;
  input.resize(p,1);for(auto& buffer:buffers)buffer.resize(p,c);
  auto trunk=[&](int g,const Epi* sink)->const X&{
   {D4_PROF(orbit_gather);const int* src=canon.data()+size_t(g)*p;
   for(int d=0;d<D;++d)for(int i=0;i<p;++i)input.at(d,i)=normalized.at(d,src[i]);}
   for(int l=0;l<cfg.layers;++l){
    const X& from=l==0?input:buffers[(l-1)%2];
    Epi last;const Epi* use=nullptr;
    if(sink&&l==cfg.layers-1){last=epilogue(from,l);last.mode=sink->mode;
     last.pooled=sink->pooled;last.pool_w=sink->pool_w;last.scalar=sink->scalar;last.e1=sink->e1;last.e2=sink->e2;
     last.canon=sink->canon;last.token=sink->token;last.ntok=sink->ntok;last.pixels=sink->pixels;
     last.token_scale=sink->token_scale;last.sign1=sink->sign1;last.sign2=sink->sign2;use=&last;}
    block(from,buffers[l%2],l,g==0?trace:nullptr,scratch,use);
   }
   return buffers[(cfg.layers-1)%2];
  };
  const int sign1[8]={1,-1,1,-1,1,-1,1,-1},sign2[8]={1,-1,1,-1,-1,1,-1,1};
  if(!cfg.attention&&!trace&&std::is_same_v<S,float>){
   X v1(1,c),v2(1,c);
   if(fast){
    for(int g=0;g<8;++g){
     ws.pooled.assign(size_t(D)*c,0.f);
     Epi sink;sink.mode=EpiMode::pool;sink.pooled=ws.pooled.data();sink.pool_w=pool_w.data();
     trunk(g,&sink);
     for(int d=0;d<D;++d)for(int j=0;j<c;++j){S v=S(ws.pooled[size_t(d)*c+j]);v1.at(d,0,j)+=S(sign1[g])*v;v2.at(d,0,j)+=S(sign2[g])*v;}
    }
   }else{
    S sum=std::accumulate(pool_map.begin(),pool_map.end(),S(0));
    for(int g=0;g<8;++g){const X& x=trunk(g,nullptr);X pooled(1,c);
     for(int i=0;i<p;++i){S weight=pool_map[orbit_src(g,i)]/sum/S(8);
      for(int d=0;d<D;++d)for(int j=0;j<c;++j)pooled.at(d,0,j)+=x.at(d,i,j)*weight;}
     add(v1,pooled,S(sign1[g]));add(v2,pooled,S(sign2[g]));}
   }
   auto a=head(v1,cfg.shared?"head_shared":"head_e1"),b=head(v2,cfg.shared?"head_shared":"head_e2");std::array<S,D*2> out{};
   for(int d=0;d<D;++d){out[d*2]=a.at(d,0);out[d*2+1]=b.at(d,0);}return out;
  }
  int grid=std::min(cfg.grid,h),side=h/grid;
  bool prepooled=cfg.attention&&!trace&&std::is_same_v<S,float>&&h%grid==0;
  S token_scale=S(1)/S(side*side);
  X& scalar=ws.scalar;X& e1=ws.e1;X& e2=ws.e2;
  scalar.resize(prepooled?grid*grid:p,c,true);e1.resize(p,c,true);e2.resize(p,c,true);
  for(int g=0;g<8;++g){
   if constexpr(std::is_same_v<S,float>){if(fast&&!trace&&cfg.attention){
    Epi sink;sink.mode=EpiMode::canonical;sink.scalar=scalar.a.data();sink.e1=e1.a.data();sink.e2=e2.a.data();
    sink.canon=canon.data()+size_t(g)*p;sink.token=prepooled?orbit_token.data()+size_t(g)*p:sink.canon;sink.ntok=scalar.p;sink.pixels=p;
    sink.token_scale=prepooled?token_scale:1.f;sink.sign1=float(sign1[g]);sink.sign2=float(sign2[g]);
    trunk(g,&sink);continue;
   }}
   const X& x=trunk(g,nullptr);
   for(int i=0;i<p;++i){int canonical=orbit_src(g,i);
    int tk=prepooled?(canonical/h/side)*grid+(canonical%h/side):canonical;
    for(int d=0;d<D;++d)for(int j=0;j<c;++j){S v=x.at(d,i,j)/S(8);
     scalar.at(d,tk,j)+=prepooled?v*token_scale:v;
     e1.at(d,canonical,j)+=S(sign1[g])*v;e2.at(d,canonical,j)+=S(sign2[g])*v;}}
  }
  save(trace,"scalar_field",scalar);save(trace,"f_mean_e1",e1);save(trace,"f_mean_e2",e2);
  X kernel(p,1);
  if(cfg.attention)kernel=attention_kernel(scalar,trace,scratch,prepooled);
  else {S sum=std::accumulate(pool_map.begin(),pool_map.end(),S(0));for(int i=0;i<p;++i)kernel.at(0,i)=pool_map[i]/sum;save(trace,"kernel",kernel);}
  X v1(1,c),v2(1,c);
  for(int i=0;i<p;++i)for(int j=0;j<c;++j){
   v1.at(0,0,j)+=e1.at(0,i,j)*kernel.at(0,i);v2.at(0,0,j)+=e2.at(0,i,j)*kernel.at(0,i);
   for(int d=1;d<D;++d){v1.at(d,0,j)+=e1.at(d,i,j)*kernel.at(0,i)+e1.at(0,i,j)*kernel.at(d,i);
    v2.at(d,0,j)+=e2.at(d,i,j)*kernel.at(0,i)+e2.at(0,i,j)*kernel.at(d,i);}
  }
  save(trace,"pooled_e1",v1);save(trace,"pooled_e2",v2);
  auto a=head(v1,cfg.shared?"head_shared":"head_e1"),b=head(v2,cfg.shared?"head_shared":"head_e2");std::array<S,D*2> out{};
  for(int d=0;d<D;++d){out[d*2]=a.at(d,0);out[d*2+1]=b.at(d,0);}return out;
 }
};
}
