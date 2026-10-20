// ISA-generic float32 kernels. Included once per ISA from kernels_simd.cpp,
// inside a namespace that first defines:
//   D4_TARGET, V/W (main vector), H/HW (attention-head vector, HW<=8),
//   MRMAX/NR (GEMM register block) and the v*/h* wrappers used below.
// Every function carries D4_TARGET; entry points expose only scalar types.
constexpr int CH=32/W;

// ---------------------------------------------------------------- math
// erf-GELU value and derivative Phi(x)+x*phi(x). Abramowitz-Stegun 7.1.26;
// the reciprocal uses estimate+Newton instead of a vector division.
D4_TARGET static inline void gelu_v(V v,V& value,V& df){
 V z=vmuls(v,.7071067811865475f);
 V t=recip(vadd(vset(1.f),vmuls(vabs(z),.3275911f)));
 V poly=vset(1.061405429f);
 poly=vfma(poly,t,vset(-1.453152027f));poly=vfma(poly,t,vset(1.421413741f));
 poly=vfma(poly,t,vset(-.284496736f));poly=vfma(poly,t,vset(.254829592f));
 V ex=exp_negative(vmuls(vmul(v,v),-.5f));
 V ea=vsub(vset(1.f),vmul(vmul(poly,t),ex));
 V phi=vmuls(vadd(vset(1.f),selectneg(v,ea,vsub(vzero(),ea))),.5f);
 df=vfma(vmuls(v,.3989422804014327f),ex,phi);
 value=vmul(v,phi);
}
// Pairwise sum of N vectors: compile-time recursion gives straight-line
// register code (a runtime width loop compiled to a store/load chain).
template<int N> D4_TARGET static inline V tree(const V* a){
 if constexpr(N==1)return a[0];
 else return vadd(tree<N/2>(a),tree<N-N/2>(a+N/2));
}

// ---------------------------------------------------------------- GEMM
// C[MR,NR] = A[MR,K] * B[K,NR]; B rows are NR contiguous floats.
template<int MR> D4_TARGET static inline void micro(const float* a,int lda,const float* b,int ldb,float* c,int ldc,int k){
 constexpr int NV=NR/W;
 V acc[MR][NV];
 for(int r=0;r<MR;++r)for(int j=0;j<NV;++j)acc[r][j]=vzero();
 for(int i=0;i<k;++i){
  V w[NV];for(int j=0;j<NV;++j)w[j]=vload(b+i*ldb+j*W);
  for(int r=0;r<MR;++r){const float s=a[r*lda+i];for(int j=0;j<NV;++j)acc[r][j]=vfmas(w[j],s,acc[r][j]);}
 }
 for(int r=0;r<MR;++r)for(int j=0;j<NV;++j)vstore(c+r*ldc+j*W,acc[r][j]);
}
template<int R> D4_TARGET static inline void micro_tail(int rows,const float* a,int lda,const float* b,int ldb,float* c,int ldc,int k){
 if constexpr(R>0){if(rows==R)micro<R>(a,lda,b,ldb,c,ldc,k);else micro_tail<R-1>(rows,a,lda,b,ldb,c,ldc,k);}
}
#if defined(D4_LANE_GEMM)
// NEON: by-element FMLA from a broadcast scalar runs at ~half the rate of
// lane FMLA from a loaded vector on Neoverse V2 (micro-benchmark:
// 45 vs ~94 GFLOP/s). So A is first transposed to At[k][m] (4x4 register
// transposes, each element once), then a 12x8 kernel takes 4 rows per
// vector: 24 accumulators + 3 A vectors + 2 B vectors.
template<int MV> static inline void lane_micro(const float* at,int ldm,const float* b,int ldb,float* c,int ldc,int k,int rows){
 float32x4_t acc[4*MV][2];
 #pragma GCC unroll 12
 for(int r=0;r<4*MV;++r){acc[r][0]=vdupq_n_f32(0.f);acc[r][1]=vdupq_n_f32(0.f);}
 for(int i=0;i<k;++i){
  float32x4_t x[MV];
#pragma GCC unroll 3
  for(int v=0;v<MV;++v)x[v]=vld1q_f32(at+size_t(i)*ldm+4*v);
  const float32x4_t w0=vld1q_f32(b+size_t(i)*ldb),w1=vld1q_f32(b+size_t(i)*ldb+4);
#pragma GCC unroll 3
  for(int v=0;v<MV;++v){
   acc[4*v][0]=vfmaq_laneq_f32(acc[4*v][0],w0,x[v],0);acc[4*v][1]=vfmaq_laneq_f32(acc[4*v][1],w1,x[v],0);
   acc[4*v+1][0]=vfmaq_laneq_f32(acc[4*v+1][0],w0,x[v],1);acc[4*v+1][1]=vfmaq_laneq_f32(acc[4*v+1][1],w1,x[v],1);
   acc[4*v+2][0]=vfmaq_laneq_f32(acc[4*v+2][0],w0,x[v],2);acc[4*v+2][1]=vfmaq_laneq_f32(acc[4*v+2][1],w1,x[v],2);
   acc[4*v+3][0]=vfmaq_laneq_f32(acc[4*v+3][0],w0,x[v],3);acc[4*v+3][1]=vfmaq_laneq_f32(acc[4*v+3][1],w1,x[v],3);
  }
 }
 // Compile-time trip count keeps acc[] in registers (a runtime bound here
 // made GCC place the accumulators on the stack).
#pragma GCC unroll 12
 for(int r=0;r<4*MV;++r)if(r<rows){vst1q_f32(c+size_t(r)*ldc,acc[r][0]);vst1q_f32(c+size_t(r)*ldc+4,acc[r][1]);}
}
static inline void transpose_rows(const float* a,int lda,int m,int k,float* at,int mp){
 int r=0;
 for(;r+4<=m;r+=4){int i=0;
  for(;i+4<=k;i+=4){
   const float32x4_t r0=vld1q_f32(a+size_t(r)*lda+i),r1=vld1q_f32(a+size_t(r+1)*lda+i);
   const float32x4_t r2=vld1q_f32(a+size_t(r+2)*lda+i),r3=vld1q_f32(a+size_t(r+3)*lda+i);
   const float32x4_t t0=vtrn1q_f32(r0,r1),t1=vtrn2q_f32(r0,r1),t2=vtrn1q_f32(r2,r3),t3=vtrn2q_f32(r2,r3);
   vst1q_f32(at+size_t(i)*mp+r,vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0),vreinterpretq_f64_f32(t2))));
   vst1q_f32(at+size_t(i+1)*mp+r,vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1),vreinterpretq_f64_f32(t3))));
   vst1q_f32(at+size_t(i+2)*mp+r,vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0),vreinterpretq_f64_f32(t2))));
   vst1q_f32(at+size_t(i+3)*mp+r,vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1),vreinterpretq_f64_f32(t3))));
  }
  for(;i<k;++i)for(int q=0;q<4;++q)at[size_t(i)*mp+r+q]=a[size_t(r+q)*lda+i];
 }
 for(int i=0;i<k;++i){for(int q=r;q<m;++q)at[size_t(i)*mp+q]=a[size_t(q)*lda+i];for(int q=m;q<mp;++q)at[size_t(i)*mp+q]=0.f;}
}
D4_TARGET static void gemm_impl(const float* a,int lda,const float* b,int ldb,float* c,int ldc,int m,int n,int k){
 thread_local std::vector<float> buffer;
 const int mp=(m+3)/4*4;
 if(buffer.size()<size_t(k)*mp)buffer.resize(size_t(k)*mp);
 float* at=buffer.data();
 transpose_rows(a,lda,m,k,at,mp);
 int col=0;
 for(;col+8<=n;col+=8){
  int row=0;
  for(;row+12<=m;row+=12)lane_micro<3>(at+row,mp,b+col,ldb,c+size_t(row)*ldc+col,ldc,k,12);
  const int rest=m-row;
  if(rest>8)lane_micro<3>(at+row,mp,b+col,ldb,c+size_t(row)*ldc+col,ldc,k,rest);
  else if(rest>4)lane_micro<2>(at+row,mp,b+col,ldb,c+size_t(row)*ldc+col,ldc,k,rest);
  else if(rest>0)lane_micro<1>(at+row,mp,b+col,ldb,c+size_t(row)*ldc+col,ldc,k,rest);
 }
 for(;col<n;++col)for(int row=0;row<m;++row){
  float s=0;for(int i=0;i<k;++i)s+=a[size_t(row)*lda+i]*b[size_t(i)*ldb+col];c[size_t(row)*ldc+col]=s;
 }
}
#else
D4_TARGET static void gemm_impl(const float* a,int lda,const float* b,int ldb,float* c,int ldc,int m,int n,int k){
 int col=0;
 for(;col+NR<=n;col+=NR){
  int row=0;
  for(;row+MRMAX<=m;row+=MRMAX)micro<MRMAX>(a+size_t(row)*lda,lda,b+col,ldb,c+size_t(row)*ldc+col,ldc,k);
  if(row<m)micro_tail<MRMAX-1>(m-row,a+size_t(row)*lda,lda,b+col,ldb,c+size_t(row)*ldc+col,ldc,k);
 }
 for(;col<n;++col)for(int row=0;row<m;++row){
  float s=0;for(int i=0;i<k;++i)s+=a[size_t(row)*lda+i]*b[size_t(i)*ldb+col];c[size_t(row)*ldc+col]=s;
 }
}
#endif

// ---------------------------------------------------------------- epilogue
// One pixel: conv output `in` ([dual] at istride) -> LN -> GELU -> residual,
// then store to `out` ([dual] at ostride) or sink into accumulators. The sink
// mode is a template parameter and every loop has a compile-time trip count,
// so the whole pixel stays in registers (a runtime switch here made GCC keep
// center[]/tmp[] on the stack and cost ~3x). All statistics are reduced
// before any output is written, so in==out is safe.
// Per-pixel LayerNorm statistics. Computed one pixel ahead of the apply step
// (see run_pixels): the reductions form a long dependent chain, and the
// next pixel's chain then overlaps the independent GELU work of this one.
template<int D> struct Stats{float mu,inv,dm[D],cov[D];};
template<int D> D4_TARGET static inline void stats_px(const float* in,int istride,Stats<D>& st){
 V center[CH],tmp[CH],prod[CH];
#pragma GCC unroll 16
 for(int j=0;j<CH;++j)tmp[j]=vload(in+j*W);
 const float mu=hsum(tree<CH>(tmp))/32.f;
#pragma GCC unroll 16
 for(int j=0;j<CH;++j){center[j]=vsub(vload(in+j*W),vset(mu));tmp[j]=vmul(center[j],center[j]);}
 st.mu=mu;
 st.inv=1.f/std::sqrt(hsum(tree<CH>(tmp))/32.f+1e-5f);
 const float csum=hsum(tree<CH>(center));
 st.dm[0]=0.f;st.cov[0]=0.f;
 // One pass per tangent: cov = (sum c*t - dm*sum c)/32 == sum c*(t-dm)/32.
#pragma GCC unroll 8
 for(int d=1;d<D;++d){
  const float* t=in+d*istride;
#pragma GCC unroll 16
  for(int j=0;j<CH;++j){tmp[j]=vload(t+j*W);prod[j]=vmul(center[j],tmp[j]);}
  st.dm[d]=hsum(tree<CH>(tmp))/32.f;
  st.cov[d]=(hsum(tree<CH>(prod))-st.dm[d]*csum)/32.f;
 }
}
// conv output `in` ([dual] at istride) -> LN -> GELU -> residual, then store to
// `out` ([dual] at ostride) or sink into accumulators. The sink mode is a
// template parameter and every loop has a compile-time trip count, so the
// whole pixel stays in registers (a runtime switch here made GCC keep
// center[]/tmp[] on the stack and cost ~3x). in==out is safe: the statistics
// are complete before this pixel is written.
template<int D,EpiMode M> D4_TARGET static inline void apply_px(const float* in,int istride,float* out,int ostride,const Epi& e,int p,const Stats<D>& st){
 const float inv=st.inv;
 const V res=vset(e.residual),muv=vset(st.mu);
 [[maybe_unused]] V pw,ts,s1,s2;
 if constexpr(M==EpiMode::pool)pw=vset(e.pool_w[p]);
 if constexpr(M==EpiMode::canonical){ts=vset(e.token_scale);s1=vset(e.sign1);s2=vset(e.sign2);}
#pragma GCC unroll 16
 for(int j=0;j<CH;++j){
  const V center=vsub(vload(in+j*W),muv);
  const V g=vload(e.gamma+j*W),ginv=vmuls(g,inv);
  V value,df;gelu_v(vadd(vmul(vmuls(center,inv),g),vload(e.beta+j*W)),value,df);
#pragma GCC unroll 8
  for(int d=0;d<D;++d){
   V r=value;
   if(d){V dt=vsub(vload(in+d*istride+j*W),vset(st.dm[d]));
    dt=vsub(dt,vmuls(vmuls(vmuls(center,st.cov[d]),inv),inv));
    r=vmul(vmul(ginv,dt),df);}
   const V xx=e.skip?vmuls(vload(e.skip+j*W),e.x[size_t(d)*e.xstride+p])
                    :vload(e.x+size_t(d)*e.xstride+size_t(p)*32+j*W);
   r=vfma(xx,res,r);
   if constexpr(M==EpiMode::pool){float* acc=e.pooled+d*32+j*W;vstore(acc,vfma(r,pw,vload(acc)));}
   else if constexpr(M==EpiMode::canonical){
    const V v=vmuls(r,.125f);
    float* sp=e.scalar+(size_t(d)*e.ntok+e.token[p])*32+j*W;vstore(sp,vfma(v,ts,vload(sp)));
    const size_t at=(size_t(d)*e.pixels+e.canon[p])*32+j*W;
    vstore(e.e1+at,vfma(v,s1,vload(e.e1+at)));
    vstore(e.e2+at,vfma(v,s2,vload(e.e2+at)));}
   else vstore(out+size_t(d)*ostride+j*W,r);
  }
 }
}
struct PixelRef{const float* in;int p;};
// Applies the epilogue to a list of pixels, statistics one pixel ahead.
template<int D,EpiMode M> D4_TARGET static inline void run_pixels(const PixelRef* px,int n,int istride,float* y,int stride,const Epi& e){
 if(n<=0)return;
 Stats<D> cur,next;
 stats_px<D>(px[0].in,istride,cur);
 for(int i=0;i<n;++i){
  if(i+1<n)stats_px<D>(px[i+1].in,istride,next);
  apply_px<D,M>(px[i].in,istride,y+size_t(px[i].p)*32,stride,e,px[i].p,cur);
  cur=next;
 }
}
// Expands CALL(MODE) for the runtime mode, with the mode as a template argument.
#define D4_MODE_SWITCH(CALL) do{switch(e.mode){ \
 case EpiMode::raw:CALL(EpiMode::raw);break;case EpiMode::store:CALL(EpiMode::store);break; \
 case EpiMode::pool:CALL(EpiMode::pool);break;case EpiMode::canonical:CALL(EpiMode::canonical);break;}}while(0)
template<int D,EpiMode M> D4_TARGET static void epilogue_plane_mode(float* y,int pixels,const Epi& e){
 // Same in/out plane: stats of pixel p+1 only read pixel p+1, so in-place is safe.
 Stats<D> cur,next;
 stats_px<D>(y,pixels*32,cur);
 for(int p=0;p<pixels;++p){
  if(p+1<pixels)stats_px<D>(y+size_t(p+1)*32,pixels*32,next);
  apply_px<D,M>(y+size_t(p)*32,pixels*32,y+size_t(p)*32,pixels*32,e,p,cur);
  cur=next;
 }
}
template<int D> D4_TARGET static void epilogue_plane_impl(float* y,int pixels,const Epi& e){
#define D4_CALL(MODE) epilogue_plane_mode<D,MODE>(y,pixels,e)
 if(e.mode==EpiMode::raw)return;
 D4_MODE_SWITCH(D4_CALL);
#undef D4_CALL
}
// Finish the pixels of one Winograd tile group from O[D][group*Tile^2][32].
template<int Tile,int D,EpiMode M> D4_TARGET static void finish_group(const float* O,float* y,int h,const Epi& e,int g0,int nb){
 constexpr int P=winograd_group*Tile*Tile;
 const int stride=h*h*32,tiles=(h+Tile-1)/Tile;
 PixelRef px[P];int n=0;
 for(int b=0;b<nb;++b){
  const int oy=(g0+b)/tiles*Tile,ox=(g0+b)%tiles*Tile;
  for(int r=0;r<Tile&&oy+r<h;++r)for(int s=0;s<Tile&&ox+s<h;++s)
   px[n++]={O+size_t(b*Tile*Tile+r*Tile+s)*32,(oy+r)*h+ox+s};
 }
 if constexpr(M==EpiMode::raw){
  for(int i=0;i<n;++i)
   for(int d=0;d<D;++d)for(int j=0;j<CH;++j)vstore(y+size_t(d)*stride+size_t(px[i].p)*32+j*W,vload(px[i].in+d*P*32+j*W));
 }
 else run_pixels<D,M>(px,n,P*32,y,stride,e);
}
// Finish one first-layer image row from C[D][h][32].
template<int D,EpiMode M> D4_TARGET static void finish_row(const float* C,float* y,int h,const Epi& e,int row){
 PixelRef px[128];
 if(h>128)throw std::invalid_argument("first-layer row too wide");
 for(int i=0;i<h;++i)px[i]={C+size_t(i)*32,row*h+i};
 if constexpr(M==EpiMode::raw){
  for(int i=0;i<h;++i)
   for(int d=0;d<D;++d)for(int j=0;j<CH;++j)vstore(y+size_t(d)*h*h*32+size_t(px[i].p)*32+j*W,vload(px[i].in+d*h*32+j*W));
 }
 else run_pixels<D,M>(px,h,h*32,y,h*h*32,e);
}

// ---------------------------------------------------------------- Winograd
// Cross-correlation with one-pixel zero padding. 1D matrices: see
// winograd.h / winograd4.h and tests/test_winograd_identity.py.
template<int Tile> D4_TARGET static inline void input_tf(const V* a,V* b){
 if constexpr(Tile==2){b[0]=vsub(a[0],a[2]);b[1]=vadd(a[1],a[2]);b[2]=vsub(a[2],a[1]);b[3]=vsub(a[1],a[3]);}
 else{
  b[0]=vfma(a[0],vset(4.f),vfnma(a[2],vset(5.f),a[4]));
  b[1]=vfnma(vadd(a[1],a[2]),vset(4.f),vadd(a[3],a[4]));
  b[2]=vfma(vsub(a[1],a[2]),vset(4.f),vsub(a[4],a[3]));
  b[3]=vfma(vsub(a[3],a[1]),vset(2.f),vsub(a[4],a[2]));
  b[4]=vfma(vsub(a[1],a[3]),vset(2.f),vsub(a[4],a[2]));
  b[5]=vfma(a[1],vset(4.f),vfnma(a[3],vset(5.f),a[5]));
 }
}
template<int Tile> D4_TARGET static inline void output_tf(const V* a,V* b){
 if constexpr(Tile==2){b[0]=vadd(vadd(a[0],a[1]),a[2]);b[1]=vsub(vsub(a[1],a[2]),a[3]);}
 else{
  const V s12=vadd(a[1],a[2]),s34=vadd(a[3],a[4]),d12=vsub(a[1],a[2]),d34=vsub(a[3],a[4]);
  b[0]=vadd(vadd(a[0],s12),s34);b[1]=vfma(d34,vset(2.f),d12);
  b[2]=vfma(s34,vset(4.f),s12);b[3]=vadd(vfma(d34,vset(8.f),d12),a[5]);
 }
}
// Groups of winograd_group tiles share one GEMM per transform point:
// rows (tile,dual) x K=32 input channels x N=32 output channels. Each 4 KB
// weight block stays in L1 for the whole group instead of being re-read per
// tile. Scratch: U[T][R][32], C[T][R][32], O[D][group*Tile^2][32].
template<int Tile,int D> D4_TARGET static void winograd_impl(const float* x,const float* weights,const float* bias,float* y,int h,const Epi& e,float* scratch){
 constexpr int N=Tile+2,T=N*N,TB=winograd_group,R=TB*D,P=TB*Tile*Tile;
 const int stride=h*h*32,tiles=(h+Tile-1)/Tile,ntile=tiles*tiles;
 float* U=scratch;float* C=U+size_t(T)*R*32;float* O=C+size_t(T)*R*32;
 for(int g0=0;g0<ntile;g0+=TB){
  const int nb=std::min(TB,ntile-g0);
  D4_PROF_MARK(tic);
  for(int b=0;b<nb;++b){
   const int oy=(g0+b)/tiles*Tile,ox=(g0+b)%tiles*Tile;
   const bool interior=oy>=1&&ox>=1&&oy+N-1<=h&&ox+N-1<=h;
   for(int d=0;d<D;++d)for(int c=0;c<32;c+=W){
    const float* src=x+size_t(d)*stride+c;V t[N][N];
    for(int s=0;s<N;++s){V a[N],o[N];
     const int xx=ox+s-1;
     for(int r=0;r<N;++r){const int yy=oy+r-1;
      a[r]=(interior||(yy>=0&&yy<h&&xx>=0&&xx<h))?vload(src+size_t(yy*h+xx)*32):vzero();}
     input_tf<Tile>(a,o);for(int r=0;r<N;++r)t[r][s]=o[r];
    }
    for(int r=0;r<N;++r){V o[N];input_tf<Tile>(t[r],o);
     for(int s=0;s<N;++s)vstore(U+(size_t(r*N+s)*R+b*D+d)*32+c,o[s]);}
   }
  }
  D4_PROF_ADD(wino_input,tic);
  for(int t=0;t<T;++t)gemm_impl(U+size_t(t)*R*32,32,weights+size_t(t)*1024,32,C+size_t(t)*R*32,32,nb*D,32,32);
  D4_PROF_ADD(wino_gemm,tic);
  for(int b=0;b<nb;++b)for(int d=0;d<D;++d)for(int c=0;c<32;c+=W){
   V t[Tile][N];
   for(int s=0;s<N;++s){V a[N],o[Tile];
    for(int r=0;r<N;++r)a[r]=vload(C+(size_t(r*N+s)*R+b*D+d)*32+c);
    output_tf<Tile>(a,o);for(int r=0;r<Tile;++r)t[r][s]=o[r];
   }
   const V bv=d==0?vload(bias+c):vzero();
   for(int r=0;r<Tile;++r){V o[Tile];output_tf<Tile>(t[r],o);
    for(int s=0;s<Tile;++s)vstore(O+(size_t(d)*P+b*Tile*Tile+r*Tile+s)*32+c,vadd(o[s],bv));}
  }
  D4_PROF_ADD(wino_output,tic);
#define D4_CALL(MODE) finish_group<Tile,D,MODE>(O,y,h,e,g0,nb)
  D4_MODE_SWITCH(D4_CALL);
#undef D4_CALL
  D4_PROF_ADD(wino_epilogue,tic);
 }
}

// ---------------------------------------------------------------- first layer
// C1 -> C32: per image row, im2col [D*h][9] then one GEMM against w9[9][32].
template<int D> D4_TARGET static void first_impl(const float* x,const float* w9,const float* bias,float* y,int h,const Epi& e,float* scratch){
 const int hh=h*h;float* A=scratch;float* C=A+size_t(D)*h*9;
 for(int row=0;row<h;++row){
  for(int d=0;d<D;++d)for(int px=0;px<h;++px){float* a=A+(size_t(d)*h+px)*9;
   for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){const int yy=row+ky-1,xx=px+kx-1;
    a[ky*3+kx]=(yy>=0&&yy<h&&xx>=0&&xx<h)?x[size_t(d)*hh+yy*h+xx]:0.f;}}
  gemm_impl(A,9,w9,32,C,32,D*h,32,9);
  for(int px=0;px<h;++px)for(int c=0;c<32;c+=W)vstore(C+px*32+c,vadd(vload(C+px*32+c),vload(bias+c)));
#define D4_CALL(MODE) finish_row<D,MODE>(C,y,h,e,row)
  D4_MODE_SWITCH(D4_CALL);
#undef D4_CALL
 }
}

// ---------------------------------------------------------------- rows
template<int D> D4_TARGET static void norm_rows_impl(float* x,int rows,int dstride,const float* gamma,const float* beta){
 for(int p=0;p<rows;++p){
  float* base=x+size_t(p)*32;V center[CH],tmp[CH];
  for(int j=0;j<CH;++j)tmp[j]=vload(base+j*W);
  const float mu=hsum(tree<CH>(tmp))/32.f;
  for(int j=0;j<CH;++j){center[j]=vsub(vload(base+j*W),vset(mu));tmp[j]=vmul(center[j],center[j]);}
  const float inv=1.f/std::sqrt(hsum(tree<CH>(tmp))/32.f+1e-5f);
  for(int d=1;d<D;++d){float* t=base+size_t(d)*dstride;
   for(int j=0;j<CH;++j)tmp[j]=vload(t+j*W);
   const float dm=hsum(tree<CH>(tmp))/32.f;
   for(int j=0;j<CH;++j)tmp[j]=vmul(center[j],vsub(vload(t+j*W),vset(dm)));
   const float cov=hsum(tree<CH>(tmp))/32.f;
   for(int j=0;j<CH;++j){V dt=vsub(vsub(vload(t+j*W),vset(dm)),vmuls(vmuls(vmuls(center[j],cov),inv),inv));
    vstore(t+j*W,vmul(vmuls(vload(gamma+j*W),inv),dt));}
  }
  for(int j=0;j<CH;++j)vstore(base+j*W,vadd(vmul(vmuls(center[j],inv),vload(gamma+j*W)),vload(beta+j*W)));
 }
}
template<int D> D4_TARGET static void gelu_impl(float* x,int count,int dstride){
 int i=0;
#pragma GCC unroll 2
 for(;i+W<=count;i+=W){V value,df;gelu_v(vload(x+i),value,df);
  for(int d=1;d<D;++d)vstore(x+size_t(d)*dstride+i,vmul(vload(x+size_t(d)*dstride+i),df));
  vstore(x+i,value);}
 for(;i<count;++i){const float v=x[i],phi=.5f*(1.f+std::erf(v*.7071067811865475f));
  const float df=phi+v*std::exp(-.5f*v*v)*.3989422804014327f;
  for(int d=1;d<D;++d)x[size_t(d)*dstride+i]*=df;
  x[i]=v*phi;}
}
template<int D> D4_TARGET static void softmax_impl(float* x,int rows,int cols,int dstride){
 for(int row=0;row<rows;++row){
  float* s=x+size_t(row)*cols;int i=0;
  V mv=vset(-std::numeric_limits<float>::infinity());
  for(;i+W<=cols;i+=W)mv=vmax(mv,vload(s+i));
  float mx=hmax(mv);for(;i<cols;++i)mx=std::max(mx,s[i]);
  V sv=vzero();float sum=0;i=0;
  for(;i+W<=cols;i+=W){const V v=exp_negative(vsub(vload(s+i),vset(mx)));vstore(s+i,v);sv=vadd(sv,v);}
  for(;i<cols;++i){s[i]=std::exp(s[i]-mx);sum+=s[i];}
  const float scale=1.f/(hsum(sv)+sum);i=0;
  for(;i+W<=cols;i+=W)vstore(s+i,vmuls(vload(s+i),scale));
  for(;i<cols;++i)s[i]*=scale;
  for(int d=1;d<D;++d){float* t=s+size_t(d)*dstride;
   V dv=vzero();float dot=0;i=0;
   for(;i+W<=cols;i+=W)dv=vfma(vload(s+i),vload(t+i),dv);
   for(;i<cols;++i)dot+=s[i]*t[i];
   dot+=hsum(dv);i=0;
   for(;i+W<=cols;i+=W)vstore(t+i,vmul(vload(s+i),vsub(vload(t+i),vset(dot))));
   for(;i<cols;++i)t[i]=s[i]*(t[i]-dot);
  }
 }
}

// ---------------------------------------------------------------- attention
// Query tiles of 16 rows; scores [D][rows][n] in scratch. Head dim must be a
// multiple of HW (callers fall back to scalar otherwise).
template<int D> D4_TARGET static void attend_impl(const float* qkv,int n,int dim,int heads,float* ctx,float* scratch){
 const int hd=dim/heads,qs=3*dim;const size_t dq=size_t(n)*qs,dc=size_t(n)*dim;
 const float factor=float(1/std::sqrt(double(hd)));
 std::fill(ctx,ctx+D*dc,0.f);
 for(int head=0;head<heads;++head)for(int first=0;first<n;first+=16){
  const int rows=std::min(16,n-first);const size_t ds=size_t(rows)*n;
  for(int row=0;row<rows;++row){
   const float* qp=qkv+size_t(first+row)*qs+head*hd;
   for(int q=0;q<n;++q){
    const float* kq=qkv+size_t(q)*qs+dim+head*hd;
    H acc=hzero();for(int j=0;j<hd;j+=HW)acc=hfma(hload(qp+j),hload(kq+j),acc);
    scratch[size_t(row)*n+q]=hsum_h(acc)*factor;
    for(int d=1;d<D;++d){
     H t=hzero();
     for(int j=0;j<hd;j+=HW){t=hfma(hload(qp+d*dq+j),hload(kq+j),t);t=hfma(hload(qp+j),hload(kq+d*dq+j),t);}
     scratch[d*ds+size_t(row)*n+q]=hsum_h(t)*factor;
    }
   }
  }
  softmax_impl<D>(scratch,rows,n,int(ds));
  for(int row=0;row<rows;++row){const int p=first+row;
   for(int q=0;q<n;++q){
    const float sv=scratch[size_t(row)*n+q];
    const float* v0=qkv+size_t(q)*qs+2*dim+head*hd;
    for(int d=0;d<D;++d){float* out=ctx+d*dc+size_t(p)*dim+head*hd;
     if(d==0){for(int j=0;j<hd;j+=HW)hstore(out+j,hfmas(hload(v0+j),sv,hload(out+j)));}
     else{const float sd=scratch[d*ds+size_t(row)*n+q];
      for(int j=0;j<hd;j+=HW)hstore(out+j,hfmas(hload(v0+d*dq+j),sv,hfmas(hload(v0+j),sd,hload(out+j))));}
    }
   }
  }
 }
}

// ---------------------------------------------------------------- dispatch
static void gemm_entry(const float* a,int lda,const float* b,int ldb,float* c,int ldc,int m,int n,int k){gemm_impl(a,lda,b,ldb,c,ldc,m,n,k);}
#define D4_DUAL(CALL) do{if(dual==3){constexpr int D3=3;CALL(D3);}else if(dual==5){constexpr int D5=5;CALL(D5);}else throw std::invalid_argument("Only 3 or 5 dual components are supported");}while(0)
static void winograd_entry(int tile,int dual,const float* x,const float* w,const float* bias,float* y,int h,const Epi& e,float* scratch){
#define D4_CALL(DD) (tile==4?winograd_impl<4,DD>(x,w,bias,y,h,e,scratch):winograd_impl<2,DD>(x,w,bias,y,h,e,scratch))
 if(tile!=2&&tile!=4)throw std::invalid_argument("Winograd tile must be 2 or 4");
 D4_DUAL(D4_CALL);
#undef D4_CALL
}
static void first_entry(int dual,const float* x,const float* w9,const float* bias,float* y,int h,const Epi& e,float* scratch){
#define D4_CALL(DD) first_impl<DD>(x,w9,bias,y,h,e,scratch)
 D4_DUAL(D4_CALL);
#undef D4_CALL
}
static void epilogue_plane_entry(int dual,float* y,int pixels,const Epi& e){
#define D4_CALL(DD) epilogue_plane_impl<DD>(y,pixels,e)
 D4_DUAL(D4_CALL);
#undef D4_CALL
}
static void norm_rows_entry(int dual,float* x,int rows,int dstride,const float* gamma,const float* beta){
#define D4_CALL(DD) norm_rows_impl<DD>(x,rows,dstride,gamma,beta)
 D4_DUAL(D4_CALL);
#undef D4_CALL
}
static void gelu_entry(int dual,float* x,int count,int dstride){
#define D4_CALL(DD) gelu_impl<DD>(x,count,dstride)
 D4_DUAL(D4_CALL);
#undef D4_CALL
}
static void softmax_entry(int dual,float* x,int rows,int cols,int dstride){
#define D4_CALL(DD) softmax_impl<DD>(x,rows,cols,dstride)
 D4_DUAL(D4_CALL);
#undef D4_CALL
}
static void attend_entry(int dual,const float* qkv,int n,int dim,int heads,float* ctx,float* scratch){
 if(dim%heads||(dim/heads)%HW)throw std::invalid_argument("Attention head dim must be a multiple of the vector width");
#define D4_CALL(DD) attend_impl<DD>(qkv,n,dim,heads,ctx,scratch)
 D4_DUAL(D4_CALL);
#undef D4_CALL
}
#undef D4_DUAL
static Kernels table(const char* name){
 Kernels k;k.isa=name;k.gemm_mr=MRMAX;k.gemm_nr=NR;k.head_width=HW;
 k.gemm=gemm_entry;k.winograd=winograd_entry;k.first_layer=first_entry;k.epilogue_plane=epilogue_plane_entry;
 k.norm_rows=norm_rows_entry;k.gelu=gelu_entry;k.softmax_rows=softmax_entry;k.attend=attend_entry;
 return k;
}
