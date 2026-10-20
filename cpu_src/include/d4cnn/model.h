#pragma once
#include "convolution.h"
#include "simd_math.h"
#include "winograd.h"
#include "winograd4.h"
#include "x86.h"
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
 struct Workspace {X normalized,input,buffers[2],scalar,e1,e2;};
 Config cfg; std::map<std::string,std::vector<S>> w;
 std::vector<std::vector<S>> packed;
 std::vector<std::vector<float>> transformed;
 bool use_winograd=false;
 bool use_winograd4=false;
 bool use_simd=false;
 std::vector<S> norm_map,pool_map;
 Conv conv;
 X86Kernels x86;
 const std::vector<S>& get(const std::string& k)const {
  auto i=w.find(k);if(i==w.end())throw std::invalid_argument("Missing parameter: "+k);return i->second;
 }
 void save(std::map<std::string,X>* trace,const std::string& name,const X& x)const{if(trace)(*trace)[name]=x;}
 X linear(const X& x,const std::string& key,int out,bool bias=true)const{
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
#if defined(__aarch64__)
  if constexpr(std::is_same_v<S,float>){if(use_simd&&x.c==32){
   for(int p=0;p<x.p;++p)simd::norm32<D>(x.a.data()+p*32,x.p*32,gamma.data(),beta.data());return;
  }}
#endif
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
#if defined(__aarch64__)
  if constexpr(std::is_same_v<S,float>){if(use_simd&&!tanh){simd::gelu<D>(x.a.data(),x.p*x.c,x.p*x.c);return;}}
#endif
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
  for(int p=0;p<x.p;++p){
   S mx=x.at(0,p,0),sum=0;for(int c=1;c<x.c;++c)mx=std::max(mx,x.at(0,p,c));
   for(int c=0;c<x.c;++c){x.at(0,p,c)=std::exp(x.at(0,p,c)-mx);sum+=x.at(0,p,c);}
   for(int c=0;c<x.c;++c)x.at(0,p,c)/=sum;
   for(int d=1;d<D;++d){S dot=0;for(int c=0;c<x.c;++c)dot+=x.at(0,p,c)*x.at(d,p,c);
    for(int c=0;c<x.c;++c)x.at(d,p,c)=x.at(0,p,c)*(x.at(d,p,c)-dot);}
  }
 }
 void block(const X& x,X& y,int layer,std::map<std::string,X>* trace)const{
  int h=cfg.size,co=cfg.channels;std::string key="blocks."+std::to_string(layer);
  const auto& b=get(key+".conv.bias");const auto& ww=packed[layer];
  bool fused=false;
  if constexpr(std::is_same_v<S,float>){
#if defined(__aarch64__)
   const auto& gamma=get(key+".norm.weight");const auto& beta=get(key+".norm.bias");
   const S* skip=(x.c==1&&co==32)?get(key+".skip.weight").data():nullptr;
   auto finish=[&](int p){
    simd::epilogue32<D>(y.a.data()+p*32,h*h*32,gamma.data(),beta.data(),
      x.a.data()+p*x.c,h*h*x.c,skip,float(cfg.residual));
   };
   if(use_winograd&&layer>0){
    if(use_winograd4){
     if(!trace)winograd4_32<D>(x.a.data(),transformed[layer].data(),b.data(),y.a.data(),h,finish);
     else winograd4_32<D>(x.a.data(),transformed[layer].data(),b.data(),y.a.data(),h,[](int){});
    }else{
     if(!trace)winograd32<D>(x.a.data(),transformed[layer].data(),b.data(),y.a.data(),h,finish);
     else winograd32<D>(x.a.data(),transformed[layer].data(),b.data(),y.a.data(),h,[](int){});
    }
    fused=!trace;
   }else{
    conv(x.a.data(),ww.data(),b.data(),y.a.data(),h,x.c,co,D);
    if(use_simd&&!trace&&(x.c==32||x.c==1)&&co==32){for(int p=0;p<h*h;++p)finish(p);fused=true;}
   }
#elif defined(__x86_64__)
   if(use_simd&&co==32){
    const auto& gamma=get(key+".norm.weight");const auto& beta=get(key+".norm.bias");
    if(use_winograd&&layer>0){
     auto kernel=use_winograd4?x86.f4:x86.f2;
     kernel(x.a.data(),transformed[layer].data(),b.data(),y.a.data(),h,D,
            gamma.data(),beta.data(),float(cfg.residual),!trace);
     fused=!trace;
    }else{
     conv(x.a.data(),ww.data(),b.data(),y.a.data(),h,x.c,co,D);
     if(!trace&&(x.c==32||x.c==1)){
      const float* skip=x.c==1?get(key+".skip.weight").data():nullptr;
      for(int p=0;p<h*h;++p)x86.epilogue(y.a.data()+p*32,h*h*32,gamma.data(),beta.data(),
       x.a.data()+p*x.c,h*h*x.c,skip,float(cfg.residual),D);
      fused=true;
     }
    }
   }else conv(x.a.data(),ww.data(),b.data(),y.a.data(),h,x.c,co,D);
#else
   conv(x.a.data(),ww.data(),b.data(),y.a.data(),h,x.c,co,D);
#endif
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
  if(fused)return;
  save(trace,key+".conv",y);layernorm(y,key+".norm");save(trace,key+".norm",y);
  activate(y);save(trace,key+".gelu",y);
  if(x.c==co)add(y,x,S(cfg.residual));else {X skip=linear(x,key+".skip",co,false);add(y,skip,S(cfg.residual));}
  save(trace,key+".out",y);
 }
 int orbit_src(int orbit,int p)const{
  int h=cfg.size,y=p/h,x=p%h;if(orbit>=4)y=h-1-y;
  for(int k=0;k<orbit%4;++k){int t=y;y=x;x=h-1-t;}return y*h+x;
 }
 X attention_kernel(const X& field,std::map<std::string,X>* trace,bool prepooled=false)const{
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
  for(int head=0;head<cfg.heads;++head)for(int first=0;first<n;first+=16){
   int rows=std::min(16,n-first);X scores(rows,n);S factor=S(1/std::sqrt(double(hd)));
   for(int row=0;row<rows;++row)for(int q=0;q<n;++q){int p=first+row;
#if defined(__aarch64__)
    if constexpr(std::is_same_v<S,float>){if(use_simd&&hd==8){
     int c=head*hd,stride=n*3*dim;
     const float* qq=qkv.a.data()+p*3*dim+c;const float* kk=qkv.a.data()+q*3*dim+dim+c;
     auto q0=vld1q_f32(qq),q1=vld1q_f32(qq+4),k0=vld1q_f32(kk),k1=vld1q_f32(kk+4);
     scores.at(0,row,q)=vaddvq_f32(vfmaq_f32(vmulq_f32(q0,k0),q1,k1))*factor;
     for(int d=1;d<D;++d){
      auto t=vaddq_f32(vmulq_f32(vld1q_f32(qq+d*stride),k0),vmulq_f32(q0,vld1q_f32(kk+d*stride)));
      t=vaddq_f32(t,vaddq_f32(vmulq_f32(vld1q_f32(qq+d*stride+4),k1),vmulq_f32(q1,vld1q_f32(kk+d*stride+4))));
      scores.at(d,row,q)=vaddvq_f32(t)*factor;
     }continue;
    }}
#endif
    S v=0;for(int j=0;j<hd;++j){int c=head*hd+j;v+=qkv.at(0,p,c)*qkv.at(0,q,dim+c);}scores.at(0,row,q)=v*factor;
    for(int d=1;d<D;++d){S dv=0;for(int j=0;j<hd;++j){int c=head*hd+j;
     dv+=qkv.at(d,p,c)*qkv.at(0,q,dim+c)+qkv.at(0,p,c)*qkv.at(d,q,dim+c);}scores.at(d,row,q)=dv*factor;}
   }
   softmax(scores);
   for(int row=0;row<rows;++row){int p=first+row;
    for(int q=0;q<n;++q)for(int d=0;d<D;++d){
     S sv=scores.at(0,row,q),sd=scores.at(d,row,q);
#if defined(__aarch64__)
     if constexpr(std::is_same_v<S,float>){if(use_simd&&hd%4==0){
      for(int j=0;j<hd;j+=4){int c=head*hd+j;
       float* out=context.a.data()+(d*n+p)*dim+c;
       auto v=vld1q_f32(qkv.a.data()+q*3*dim+2*dim+c),a=vld1q_f32(out);
       if(d==0)a=vfmaq_n_f32(a,v,sv);
       else a=vaddq_f32(a,vaddq_f32(vmulq_n_f32(v,sd),
          vmulq_n_f32(vld1q_f32(qkv.a.data()+(d*n+q)*3*dim+2*dim+c),sv)));
       vst1q_f32(out,a);
      }continue;
     }}
#endif
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
   S u=(xx*xx+yy*yy)/S(cfg.r_cut*cfg.r_cut),v=1;for(int k=0;k<cfg.order/2;++k)v*=u;blurred.at(0,y*h+x)-=v;}
  save(trace,"kernel_logits",blurred);
  // Softmax across pixels, rather than channels.
  X flat(1,h*h);flat.a=blurred.a;softmax(flat);flat.p=h*h;flat.c=1;save(trace,"kernel",flat);return flat;
 }
 X head(X x,const std::string& key)const{
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
#if defined(__aarch64__) || defined(__x86_64__)
  const auto resolved=resolve_isa(isa);
  use_simd=std::is_same_v<S,float>&&resolved!="scalar";
#if defined(__x86_64__)
  if(use_simd)x86=select_x86_kernels(resolved);
#endif
  use_winograd=use_simd&&c.channels==32&&algorithm!="direct";
  use_winograd4=use_winograd&&(algorithm=="auto"||algorithm=="winograd4");
#endif
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
#if defined(__aarch64__) || defined(__x86_64__)
   if constexpr(std::is_same_v<S,float>){if(use_winograd&&l>0)transformed.back()=use_winograd4?winograd4_weights(packed.back()):winograd_weights(packed.back());}
#endif
  }
 }
 std::string algorithm()const{return use_winograd?(use_winograd4?"winograd_f4":"winograd_f2"):"direct";}
 std::array<S,D*2> measure(const S* q,std::map<std::string,X>* trace=nullptr)const{
  // Scratch is private to each worker; all accumulation fields are reset below.
  thread_local Workspace ws;
  int h=cfg.size,p=h*h,c=cfg.channels;
  X& normalized=ws.normalized;normalized.resize(p,1);std::array<S,D> m{},norm{};
  for(int d=0;d<D;++d){for(int i=0;i<p;++i)m[d]+=q[d*p+i]*norm_map[i];m[d]*=S(100);}
  S z=m[0]-S(1);norm[0]=S(1)+(S(10)*z>S(20)?z:std::log1p(std::exp(S(10)*z))/S(10));
  S sig=S(10)*z>S(20)?S(1):(z>=0?S(1)/(S(1)+std::exp(-S(10)*z)):std::exp(S(10)*z)/(S(1)+std::exp(S(10)*z)));
  for(int d=1;d<D;++d)norm[d]=sig*m[d];
  for(int i=0;i<p;++i){normalized.at(0,i)=q[i]/norm[0];for(int d=1;d<D;++d)normalized.at(d,i)=(q[d*p+i]-normalized.at(0,i)*norm[d])/norm[0];}
  save(trace,"normalized",normalized);
  X& input=ws.input;auto& buffers=ws.buffers;
  input.resize(p,1);for(auto& buffer:buffers)buffer.resize(p,c);
  auto trunk=[&](int g)->const X&{
   for(int d=0;d<D;++d)for(int i=0;i<p;++i)input.at(d,i)=normalized.at(d,orbit_src(g,i));
   for(int l=0;l<cfg.layers;++l)block(l==0?input:buffers[(l-1)%2],buffers[l%2],l,g==0?trace:nullptr);
   return buffers[(cfg.layers-1)%2];
  };
  const int sign1[8]={1,-1,1,-1,1,-1,1,-1},sign2[8]={1,-1,1,-1,-1,1,-1,1};
  if(!cfg.attention&&!trace&&std::is_same_v<S,float>){
   X v1(1,c),v2(1,c);S sum=std::accumulate(pool_map.begin(),pool_map.end(),S(0));
   for(int g=0;g<8;++g){const X& x=trunk(g);X pooled(1,c);
    for(int i=0;i<p;++i){S weight=pool_map[orbit_src(g,i)]/sum/S(8);
     for(int d=0;d<D;++d)for(int j=0;j<c;++j)pooled.at(d,0,j)+=x.at(d,i,j)*weight;}
    add(v1,pooled,S(sign1[g]));add(v2,pooled,S(sign2[g]));}
   auto a=head(v1,cfg.shared?"head_shared":"head_e1"),b=head(v2,cfg.shared?"head_shared":"head_e2");std::array<S,D*2> out{};
   for(int d=0;d<D;++d){out[d*2]=a.at(d,0);out[d*2+1]=b.at(d,0);}return out;
  }
  int grid=std::min(cfg.grid,h),side=h/grid;
  bool prepooled=cfg.attention&&!trace&&std::is_same_v<S,float>&&h%grid==0;
  S token_scale=S(1)/S(side*side);
  X& scalar=ws.scalar;X& e1=ws.e1;X& e2=ws.e2;
  scalar.resize(prepooled?grid*grid:p,c,true);e1.resize(p,c,true);e2.resize(p,c,true);
  for(int g=0;g<8;++g){const X& x=trunk(g);
   for(int i=0;i<p;++i){int canonical=orbit_src(g,i);
    int token=prepooled?(canonical/h/side)*grid+(canonical%h/side):canonical;
    for(int d=0;d<D;++d)for(int j=0;j<c;++j){S v=x.at(d,i,j)/S(8);
     scalar.at(d,token,j)+=prepooled?v*token_scale:v;
     e1.at(d,canonical,j)+=S(sign1[g])*v;e2.at(d,canonical,j)+=S(sign2[g])*v;}}
  }
  save(trace,"scalar_field",scalar);save(trace,"f_mean_e1",e1);save(trace,"f_mean_e2",e2);
  X kernel(p,1);
  if(cfg.attention)kernel=attention_kernel(scalar,trace,prepooled);
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
