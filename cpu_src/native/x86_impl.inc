// Included once per ISA from x86.cpp. Every vector-valued function has the
// matching target attribute; public entry points expose only scalar types.
TARGET static inline V constant(float x){return OP(set1_ps)(x);}
TARGET static inline V add(V a,V b){return OP(add_ps)(a,b);}
TARGET static inline V sub(V a,V b){return OP(sub_ps)(a,b);}
TARGET static inline V mul(V a,V b){return OP(mul_ps)(a,b);}
TARGET static inline V scale(V a,float b){return mul(a,constant(b));}
TARGET static inline V fma(V a,V b,V c){return OP(fmadd_ps)(a,b,c);}
TARGET static inline float sum(V a){
 alignas(64) float lanes[WIDTH];OP(storeu_ps)(lanes,a);
 float result=0;for(float v:lanes)result+=v;return result;
}
TARGET static inline V exp_negative(V x){
 auto z=OP(max_ps)(x,constant(-87.f));
 auto n=ROUND(scale(z,1.4426950408889634f));
 auto r=OP(fnmadd_ps)(n,constant(.693145751953125f),z);
 r=OP(fnmadd_ps)(n,constant(1.428606765330187e-6f),r);
 auto p=constant(1.f/5040.f);
 p=fma(p,r,constant(1.f/720.f));p=fma(p,r,constant(1.f/120.f));
 p=fma(p,r,constant(1.f/24.f));p=fma(p,r,constant(1.f/6.f));
 p=fma(p,r,constant(.5f));p=fma(p,r,constant(1.f));p=fma(p,r,constant(1.f));
 // Saturating exp below -87 is below float rounding for erf-GELU.
 return CLAMP_EXP(x,mul(p,EXP_POWER(n)));
}
TARGET static inline void gelu(V v,V& value,V& derivative){
 auto z=scale(v,.7071067811865475f);
 auto absz=OP(max_ps)(z,sub(constant(0),z));
 auto t=OP(div_ps)(constant(1),add(constant(1),scale(absz,.3275911f)));
 auto poly=constant(1.061405429f);
 poly=fma(poly,t,constant(-1.453152027f));poly=fma(poly,t,constant(1.421413741f));
 poly=fma(poly,t,constant(-.284496736f));poly=fma(poly,t,constant(.254829592f));
 auto ex=exp_negative(scale(mul(v,v),-.5f));
 auto ea=sub(constant(1),mul(mul(poly,t),ex));
 auto erfv=NEGATIVE(ea,sub(constant(0),ea));
 auto phi=scale(add(constant(1),erfv),.5f);
 value=mul(v,phi);derivative=fma(scale(v,.3989422804014327f),ex,phi);
}
template<int D> TARGET static void epilogue_dual(float* base,int stride,const float* gamma,const float* beta,
 const float* x,int xstride,const float* skip,float residual){
 constexpr int chunks=32/WIDTH;
 V total=constant(0),center[chunks];
 for(int j=0;j<chunks;++j)total=add(total,OP(loadu_ps)(base+j*WIDTH));
 float mu=sum(total)/32;total=constant(0);
 for(int j=0;j<chunks;++j){center[j]=sub(OP(loadu_ps)(base+j*WIDTH),constant(mu));total=fma(center[j],center[j],total);}
 float inv=1.f/std::sqrt(sum(total)/32+1e-5f),dm[D]{},cov[D]{};
 for(int d=1;d<D;++d){total=constant(0);
  for(int j=0;j<chunks;++j)total=add(total,OP(loadu_ps)(base+d*stride+j*WIDTH));
  dm[d]=sum(total)/32;total=constant(0);
  for(int j=0;j<chunks;++j)total=fma(center[j],sub(OP(loadu_ps)(base+d*stride+j*WIDTH),constant(dm[d])),total);
  cov[d]=sum(total)/32;
 }
 for(int j=0;j<chunks;++j){auto g=OP(loadu_ps)(gamma+j*WIDTH);
  auto v=add(mul(scale(center[j],inv),g),OP(loadu_ps)(beta+j*WIDTH));
  V value,df;gelu(v,value,df);
  for(int d=0;d<D;++d){auto out=value;
   if(d){auto dt=sub(OP(loadu_ps)(base+d*stride+j*WIDTH),constant(dm[d]));
    dt=sub(dt,scale(scale(scale(center[j],cov[d]),inv),inv));out=mul(mul(scale(g,inv),dt),df);}
   auto xx=skip?scale(OP(loadu_ps)(skip+j*WIDTH),x[d*xstride]):OP(loadu_ps)(x+d*xstride+j*WIDTH);
   OP(storeu_ps)(base+d*stride+j*WIDTH,fma(xx,constant(residual),out));
  }
 }
}
TARGET static void epilogue(float* base,int stride,const float* gamma,const float* beta,
 const float* x,int xstride,const float* skip,float residual,int dual){
 if(dual==3)epilogue_dual<3>(base,stride,gamma,beta,x,xstride,skip,residual);
 else if(dual==5)epilogue_dual<5>(base,stride,gamma,beta,x,xstride,skip,residual);
 else throw std::invalid_argument("Only 3 or 5 dual components are supported");
}
template<int Tile> TARGET static inline void input_transform(const V* a,V* b){
 if constexpr(Tile==2){b[0]=sub(a[0],a[2]);b[1]=add(a[1],a[2]);b[2]=sub(a[2],a[1]);b[3]=sub(a[1],a[3]);}
 else {
  b[0]=fma(a[0],constant(4),OP(fnmadd_ps)(a[2],constant(5),a[4]));
  b[1]=OP(fnmadd_ps)(add(a[1],a[2]),constant(4),add(a[3],a[4]));
  b[2]=fma(sub(a[1],a[2]),constant(4),sub(a[4],a[3]));
  b[3]=fma(sub(a[3],a[1]),constant(2),sub(a[4],a[2]));
  b[4]=fma(sub(a[1],a[3]),constant(2),sub(a[4],a[2]));
  b[5]=fma(a[1],constant(4),OP(fnmadd_ps)(a[3],constant(5),a[5]));
 }
}
template<int Tile> TARGET static inline void output_transform(const V* a,V* b){
 if constexpr(Tile==2){b[0]=add(add(a[0],a[1]),a[2]);b[1]=sub(sub(a[1],a[2]),a[3]);}
 else {auto s12=add(a[1],a[2]),s34=add(a[3],a[4]),d12=sub(a[1],a[2]),d34=sub(a[3],a[4]);
  b[0]=add(add(a[0],s12),s34);b[1]=fma(d34,constant(2),d12);
  b[2]=fma(s34,constant(4),s12);b[3]=add(fma(d34,constant(8),d12),a[5]);}
}
template<int Tile,int D> TARGET static void winograd_dual(const float* x,const float* weights,const float* bias,
 float* y,int h,const float* gamma,const float* beta,float residual,bool fused){
 constexpr int N=Tile+2, chunks=32/WIDTH;
 const int stride=h*h*32;
 alignas(64) float input[N*N][D][32],output[N*N][D][32];
 for(int oy=0;oy<h;oy+=Tile)for(int ox=0;ox<h;ox+=Tile){
  for(int d=0;d<D;++d)for(int c=0;c<32;c+=WIDTH){V t[N][N];
   for(int s=0;s<N;++s){V a[N],b[N];
    for(int r=0;r<N;++r){int yy=oy+r-1,xx=ox+s-1;
     a[r]=(yy>=0&&yy<h&&xx>=0&&xx<h)?OP(loadu_ps)(x+d*stride+(yy*h+xx)*32+c):constant(0);}
    input_transform<Tile>(a,b);for(int r=0;r<N;++r)t[r][s]=b[r];}
   for(int r=0;r<N;++r){V b[N];input_transform<Tile>(t[r],b);
    for(int s=0;s<N;++s)OP(storeu_ps)(input[r*N+s][d]+c,b[s]);}
  }
  // AVX2 uses at most 12 accumulators for D3 and 10 for D5; AVX512
  // uses 6/10. Each loaded weight is shared by every dual direction.
  constexpr int block=(WIDTH==8&&D==5)?2:chunks;
  for(int t=0;t<N*N;++t)for(int o=0;o<32;o+=WIDTH*block){V acc[D][block];
   for(int d=0;d<D;++d)for(int j=0;j<block;++j)acc[d][j]=constant(0);
   for(int i=0;i<32;++i)for(int j=0;j<block;++j){auto w=OP(loadu_ps)(weights+(t*32+i)*32+o+j*WIDTH);
    for(int d=0;d<D;++d)acc[d][j]=fma(w,constant(input[t][d][i]),acc[d][j]);}
   for(int d=0;d<D;++d)for(int j=0;j<block;++j)OP(storeu_ps)(output[t][d]+o+j*WIDTH,acc[d][j]);
  }
  for(int d=0;d<D;++d)for(int c=0;c<32;c+=WIDTH){V t[Tile][N];
   for(int s=0;s<N;++s){V a[N],b[Tile];
    for(int r=0;r<N;++r)a[r]=OP(loadu_ps)(output[r*N+s][d]+c);
    output_transform<Tile>(a,b);for(int r=0;r<Tile;++r)t[r][s]=b[r];}
   auto b=d==0?OP(loadu_ps)(bias+c):constant(0);
   for(int r=0;r<Tile&&oy+r<h;++r){V a[Tile];output_transform<Tile>(t[r],a);
    for(int s=0;s<Tile&&ox+s<h;++s)OP(storeu_ps)(y+d*stride+((oy+r)*h+ox+s)*32+c,add(a[s],b));}
  }
  if(fused)for(int r=0;r<Tile&&oy+r<h;++r)for(int s=0;s<Tile&&ox+s<h;++s){int p=(oy+r)*h+ox+s;
   epilogue_dual<D>(y+p*32,stride,gamma,beta,x+p*32,stride,nullptr,residual);}
 }
}
template<int Tile> TARGET static void winograd(const float* x,const float* weights,const float* bias,
 float* y,int h,int dual,const float* gamma,const float* beta,float residual,bool fused){
 if(dual==3)winograd_dual<Tile,3>(x,weights,bias,y,h,gamma,beta,residual,fused);
 else if(dual==5)winograd_dual<Tile,5>(x,weights,bias,y,h,gamma,beta,residual,fused);
 else throw std::invalid_argument("Only 3 or 5 dual components are supported");
}
