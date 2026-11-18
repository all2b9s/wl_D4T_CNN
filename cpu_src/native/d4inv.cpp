// CPU forward-mode inference for D4InvCNN (src/architecture/d4inv_CNN.py).
//
// One galaxy = one 64x64 q stamp [v, t_1..t_T]; output e[2] and R[2][T] = de/dt.
// All tangents share every weight load (dual numbers, T <= 4).
//
// Exact algebraic reductions of the PyTorch model (no approximation):
//  * ObservableCenter: Gaussian centroid weights are separable; the odd-grid
//    Fourier shift is a real Dirichlet-kernel Toeplitz operator, so the shift
//    is Ky * I * Kx^T (two 63^3 matmuls) instead of 127x127 FFTs.
//  * D4IsotropicConv2d kernels have edge = 4 * corner, i.e. two independent
//    taps: a 32->32 3x3 conv is a 64->32 GEMM on [x, 4*cross-sum + diag-sum].
//  * The final linear F8 scalar conv is only used after 63->16 adaptive average
//    pooling, so pooled shifted box sums are formed first (256-token GEMM).
//  * The spin-2 depthwise convolutions are only used through sum_p f(p) K(p);
//    the adjoint convolves the pool kernel K once instead of 32 channels.
//  * Bilinear upsample + replicate pad + separable blur = M L M^T.
// Only floating-point evaluation order differs; GELU-with-derivative uses
// Zelen-Severo Phi (|abs err| < 7.5e-8) sharing the exp of phi.
//
// Performance layout: planar features [comp][channel][65 x 68] vectorized over
// pixels (LayerNorm reductions are vertical adds); NEON lane-FMA micro-kernels
// for the front GEMM and attention on aarch64, generic vector code elsewhere
// (compile with -DD4_GENERIC to test that path on aarch64).
#if defined(__aarch64__) && !defined(D4_GENERIC)
#include <arm_neon.h>
#endif
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace py = pybind11;
#ifdef D4_PROFILE
#include <chrono>
static double g_prof[8];
#endif

namespace {

#if defined(__AVX2__)  // 256-bit max: attention keeps 8-float head vectors
constexpr int VB = 32;
#else
constexpr int VB = 16;
#endif

template <class T> struct VecT { typedef T type __attribute__((vector_size(VB))); };
template <class T> using vt = typename VecT<T>::type;
template <class T> constexpr int LANES = VB / sizeof(T);

template <class T> inline vt<T> ld(const T* p) { vt<T> v; std::memcpy(&v, p, sizeof v); return v; }
template <class T> inline void st(T* p, vt<T> v) { std::memcpy(p, &v, sizeof v); }
template <class T> inline vt<T> bc(T s) { return vt<T>{} + s; }
template <class V> inline auto hsum(V v) {
  using T = std::remove_cv_t<std::remove_reference_t<decltype(v[0])>>;
  T s = 0; for (int i = 0; i < (int)(sizeof(V) / sizeof(T)); ++i) s += v[i]; return s;
}

constexpr int S = 63, PW = 65, NPIX = S * S, CH = 32, NB = 16, NTOK = 256, AD = 32,
              NH = 4, HD = 8, FF = 64, GH = 64, MAXT = 4, KW = 67;

// ---------------------------------------------------------------- vector math
__attribute__((always_inline)) inline vt<float> vexp(vt<float> x) {
  typedef int vi __attribute__((vector_size(VB)));
  x = x < bc(88.0f) ? x : bc(88.0f);
  x = x > bc(-87.0f) ? x : bc(-87.0f);
  vt<float> t = x * 1.44269504088896341f + 12582912.0f;
  vt<float> n = t - 12582912.0f;
  vi ni = (vi)t - (vi)bc(12582912.0f);
  vt<float> r = x - n * 0.693359375f;
  r = r - n * (-2.12194440e-4f);
  vt<float> p = bc(1.9875691500E-4f);
  p = p * r + 1.3981999507E-3f;
  p = p * r + 8.3334519073E-3f;
  p = p * r + 4.1665795894E-2f;
  p = p * r + 1.6666665459E-1f;
  p = p * r + 5.0000001201E-1f;
  p = p * (r * r) + r + 1.0f;
  vi e = (ni + 127) << 23;
  return p * (vt<float>)e;
}
inline vt<double> vexp(vt<double> x) { for (int i = 0; i < LANES<double>; ++i) x[i] = std::exp(x[i]); return x; }

// exp for x <= 0 (softmax shifts, -z^2/2): one-sided clamp, and 2^n added
// directly to the exponent bits (no 2^n multiply). Same polynomial as vexp.
__attribute__((always_inline)) inline vt<float> vexp_neg(vt<float> x) {
  typedef int vi __attribute__((vector_size(VB)));
#if defined(__aarch64__) && !defined(D4_GENERIC)
  x = (vt<float>)vmaxq_f32((float32x4_t)x, vdupq_n_f32(-87.0f));
#else
  x = x > bc(-87.0f) ? x : bc(-87.0f);
#endif
  vt<float> t = x * 1.44269504088896341f + 12582912.0f;
  vt<float> n = t - 12582912.0f;
  vi ni = (vi)t - (vi)bc(12582912.0f);
  vt<float> r = x - n * 0.693359375f;
  r = r - n * (-2.12194440e-4f);
  vt<float> p = bc(1.9875691500E-4f);
  p = p * r + 1.3981999507E-3f;
  p = p * r + 8.3334519073E-3f;
  p = p * r + 4.1665795894E-2f;
  p = p * r + 1.6666665459E-1f;
  p = p * r + 5.0000001201E-1f;
  p = p * (r * r) + r + 1.0f;
  return (vt<float>)((vi)p + (ni << 23));
}
inline vt<double> vexp_neg(vt<double> x) { return vexp(x); }

__attribute__((always_inline)) inline vt<float> verf(vt<float> x) {  // rational minimax, |err| ~ few ulp (Eigen/XLA float erf)
  x = x < bc(4.0f) ? x : bc(4.0f);
  x = x > bc(-4.0f) ? x : bc(-4.0f);
  vt<float> x2 = x * x;
  vt<float> p = x2 * -2.72614225801306e-10f + 2.77068142495902e-08f;
  p = p * x2 + -2.10102402082508e-06f;
  p = p * x2 + -5.69250639462346e-05f;
  p = p * x2 + -7.34990630326855e-04f;
  p = p * x2 + -2.95459980854025e-03f;
  p = p * x2 + -1.60960333262415e-02f;
  p = p * x;
  vt<float> q = x2 * -1.45660718464996e-05f + -2.13374055278905e-04f;
  q = q * x2 + -1.68282697438203e-03f;
  q = q * x2 + -7.37332916720468e-03f;
  q = q * x2 + -1.42647390514189e-02f;
  return p / q;
}
inline vt<double> verf(vt<double> x) { for (int i = 0; i < LANES<double>; ++i) x[i] = std::erf(x[i]); return x; }

// GELU and, if need_d, its derivative. With the derivative, Phi shares the
// exp(-z^2/2) of phi via Zelen-Severo (A&S 26.2.17, |abs err| < 7.5e-8, i.e.
// below float32 resolution of Phi); otherwise the rational erf above is used.
template <class V> __attribute__((always_inline)) inline void gelu(V z, V& g, V& dg, bool need_d) {
  using T = std::remove_cv_t<std::remove_reference_t<decltype(z[0])>>;
  if (!need_d) {
    g = z * ((verf(z * T(0.70710678118654752440)) + T(1)) * T(0.5));
    return;
  }
  V phi = vexp_neg(z * z * T(-0.5)) * T(0.39894228040143267794);
  V phi_cdf;
  if constexpr (sizeof(T) == 4) {
    V a = z < T(0) ? -z : z;
    V t = T(1) / (a * T(0.2316419) + T(1));
    V p = t * T(1.330274429) + T(-1.821255978);
    p = p * t + T(1.781477937);
    p = p * t + T(-0.356563782);
    p = p * t + T(0.319381530);
    V tail = phi * (p * t);
    phi_cdf = z < T(0) ? tail : T(1) - tail;
  } else {
    phi_cdf = (verf(z * T(0.70710678118654752440)) + T(1)) * T(0.5);
  }
  g = z * phi_cdf;
  dg = phi_cdf + z * phi;
}

// -------------------------------------------------------------------- GEMM
// Y[m][0:32] = A[m][0:K] @ W[0:K][0:32] (+bias), row stride lda/ldw/ldy.
template <class T>
static void gemm32(const T* A, int lda, int M, int K, const T* W, int ldw, T* Y, int ldy, const T* bias) {
  constexpr int L = LANES<T>, NV = 32 / L;
  int m = 0;
  for (; m + 3 <= M; m += 3) {
    const T *a0 = A + (size_t)m * lda, *a1 = a0 + lda, *a2 = a1 + lda;
    vt<T> c0[NV], c1[NV], c2[NV];
    for (int v = 0; v < NV; ++v) c0[v] = c1[v] = c2[v] = bias ? ld(bias + v * L) : vt<T>{};
    for (int k = 0; k < K; ++k) {
      const T* w = W + (size_t)k * ldw;
      T x0 = a0[k], x1 = a1[k], x2 = a2[k];
      for (int v = 0; v < NV; ++v) {
        vt<T> b = ld(w + v * L);
        c0[v] += x0 * b; c1[v] += x1 * b; c2[v] += x2 * b;
      }
    }
    T *y0 = Y + (size_t)m * ldy, *y1 = y0 + ldy, *y2 = y1 + ldy;
    for (int v = 0; v < NV; ++v) { st(y0 + v * L, c0[v]); st(y1 + v * L, c1[v]); st(y2 + v * L, c2[v]); }
  }
  for (; m < M; ++m) {
    const T* a0 = A + (size_t)m * lda;
    vt<T> c0[NV];
    for (int v = 0; v < NV; ++v) c0[v] = bias ? ld(bias + v * L) : vt<T>{};
    for (int k = 0; k < K; ++k) {
      const T* w = W + (size_t)k * ldw;
      for (int v = 0; v < NV; ++v) c0[v] += a0[k] * ld(w + v * L);
    }
    for (int v = 0; v < NV; ++v) st(Y + (size_t)m * ldy + v * L, c0[v]);
  }
}
// General N (multiple of 32).
template <class T>
static void gemm(const T* A, int lda, int M, int K, const T* W, int N, T* Y, int ldy, const T* bias) {
  for (int n = 0; n < N; n += 32) gemm32(A, lda, M, K, W + n, N, Y + n, ldy, bias ? bias + n : nullptr);
}
// Small dense C[M][N] = A[M][K] B[K][N], vectorized over N (any N).
template <class T> static void matmul(const T* A, const T* B, T* C, int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    T* c = C + (size_t)i * N;
    std::fill(c, c + N, T(0));
    for (int k = 0; k < K; ++k) {
      T a = A[(size_t)i * K + k];
      const T* b = B + (size_t)k * N;
#pragma GCC ivdep
      for (int j = 0; j < N; ++j) c[j] += a * b[j];
    }
  }
}

// ---------------------------------------------------------- per-pixel norms
// LayerNorm over 32 channels for nc dual components; tangents have no bias.
// If act: out[c] += GELU(LN(y)) (dual); else out[c] = LN(y).
template <class T>
inline void ln32(int nc, T* const* y, const T* g, const T* b, T eps, T* const* out, bool act) {
  constexpr int L = LANES<T>, NV = 32 / L;
  vt<T> n[NV], z[NV], dg[NV];
  vt<T> acc{};
  for (int v = 0; v < NV; ++v) acc += ld(y[0] + v * L);
  T mu = hsum(acc) / T(32);
  acc = vt<T>{};
  for (int v = 0; v < NV; ++v) { n[v] = ld(y[0] + v * L) - mu; acc += n[v] * n[v]; }
  T r = T(1) / std::sqrt(hsum(acc) / T(32) + eps);
  for (int v = 0; v < NV; ++v) {
    n[v] *= r;
    z[v] = n[v] * ld(g + v * L) + ld(b + v * L);
    if (act) {
      vt<T> gv;
      gelu(z[v], gv, dg[v], nc > 1);
      st(out[0] + v * L, ld(out[0] + v * L) + gv);
    } else {
      st(out[0] + v * L, z[v]);
    }
  }
  for (int c = 1; c < nc; ++c) {
    vt<T> s1{}, s2{}, d[NV];
    for (int v = 0; v < NV; ++v) { d[v] = ld(y[c] + v * L); s1 += d[v]; }
    T m1 = hsum(s1) / T(32);
    for (int v = 0; v < NV; ++v) { d[v] -= m1; s2 += n[v] * d[v]; }
    T proj = hsum(s2) / T(32);
    for (int v = 0; v < NV; ++v) {
      vt<T> zd = (d[v] - n[v] * proj) * r * ld(g + v * L);
      if (act) st(out[c] + v * L, ld(out[c] + v * L) + dg[v] * zd);
      else st(out[c] + v * L, zd);
    }
  }
}

// ------------------------------------------------------------- small duals
struct Dual {
  double v = 0, d[MAXT] = {0, 0, 0, 0};
};
inline Dual dd(double v) { Dual r; r.v = v; return r; }
inline Dual operator+(Dual a, const Dual& b) { a.v += b.v; for (int t = 0; t < MAXT; ++t) a.d[t] += b.d[t]; return a; }
inline Dual operator-(Dual a, const Dual& b) { a.v -= b.v; for (int t = 0; t < MAXT; ++t) a.d[t] -= b.d[t]; return a; }
inline Dual operator*(const Dual& a, const Dual& b) {
  Dual r; r.v = a.v * b.v; for (int t = 0; t < MAXT; ++t) r.d[t] = a.d[t] * b.v + a.v * b.d[t]; return r;
}
inline Dual operator*(Dual a, double s) { a.v *= s; for (int t = 0; t < MAXT; ++t) a.d[t] *= s; return a; }
inline Dual chain(const Dual& a, double f, double df) { Dual r; r.v = f; for (int t = 0; t < MAXT; ++t) r.d[t] = df * a.d[t]; return r; }
inline Dual operator/(const Dual& a, const Dual& b) {
  Dual r; r.v = a.v / b.v; for (int t = 0; t < MAXT; ++t) r.d[t] = (a.d[t] - r.v * b.d[t]) / b.v; return r;
}

// ------------------------------------------------------------------ weights
template <class T> struct Weights {
  // center / readout / scalars
  double sigma = 2, max_shift = 3, gain = 1, alpha = 0.25, nl_beta = 10, nl_threshold = 20;
  int iters = 3, npad = 127, gate_mode = 0;  // 0 tanh MLP, 1 odd GELU half, 2 odd GELU difference
  T eps1 = 1e-5, eps2 = 1e-5, eps_a = 1e-5, eps_f = 1e-5;
  std::vector<T> wb;                                   // [63*63] brightness weight
  std::vector<T> l1;                                   // [7][32]: c, e, k, bias, gamma, beta, skip*res
  std::vector<T> W2, b2, g2, beta2;                    // W2 [64][32]: [x, 4*cross + diag]
  T res2 = 0.25;
  std::vector<T> W3, b3;                               // [96][32] incl. residual
  std::vector<T> Wtok, tokbias;                        // [32][32], [256][32]
  std::vector<T> an_g, an_b, Wqkv, bqkv, Wo, bo, fn_g, fn_b, Wf1, bf1, Wf2, bf2, wk;  // [in][out]
  T bk = 0;
  std::vector<T> M, MT, sup;                           // [63][16], [16][63], [63*63]
  std::vector<T> B;                                    // [2][25]
  std::vector<double> common, fscale, gW1, gb1, gW2, gb2;  // gate W1 [64][32], W2 [32][64] (torch layout)
  int pb0[NB], pb1[NB];
};

// -------------------------------------------------------------------- worker
// Planar feature layout: plane (comp c, channel o) = 65 x RS, one zero border
// pixel; interior pixel (y, x) at [(y+1)*RS + x+1]. Rows are processed as
// NX = 64 vector lanes (lane 63 lands on the right zero border and is reset).
constexpr int RS = 68, PS = PW * RS, NX = 64;

// Y[o][n] = sum_k W[k*ldw + o] A[k*lda + n] for o < O (multiple of 8), n < N
// (multiple of 2 vectors).
template <class T>
static void gemm_planar(const T* A, int lda, int N, int K, const T* W, int ldw, int O, T* Y, int ldy) {
  constexpr int L = LANES<T>, MR = 8;
  for (int n0 = 0; n0 < N; n0 += 2 * L)
    for (int o0 = 0; o0 < O; o0 += MR) {
      vt<T> c[MR][2];
#pragma GCC unroll 8
      for (int i = 0; i < MR; ++i) c[i][0] = c[i][1] = vt<T>{};
      const T* a = A + n0;
      const T* w = W + o0;
      for (int k = 0; k < K; ++k, a += lda, w += ldw) {
        vt<T> a0 = ld(a), a1 = ld(a + L);
#pragma GCC unroll 8
        for (int i = 0; i < MR; ++i) { T wi = w[i]; c[i][0] += a0 * wi; c[i][1] += a1 * wi; }
      }
#pragma GCC unroll 8
      for (int i = 0; i < MR; ++i) { st(Y + (o0 + i) * ldy + n0, c[i][0]); st(Y + (o0 + i) * ldy + n0 + L, c[i][1]); }
    }
}

#if defined(__aarch64__) && !defined(D4_GENERIC)
// NEON micro-kernel: 8 output channels x 4*NV pixels, W broadcast by lane.
template <int NV>
static inline void gemm_tile_f32(const float* A, int lda, int K, const float* W, int ldw, float* Y, int ldy) {
  float32x4_t c[8][NV];
  for (int i = 0; i < 8; ++i)
    for (int v = 0; v < NV; ++v) c[i][v] = vdupq_n_f32(0.f);
  for (int k = 0; k < K; ++k, A += lda, W += ldw) {
    const float32x4_t w0 = vld1q_f32(W), w1 = vld1q_f32(W + 4);
    float32x4_t a[NV];
    for (int v = 0; v < NV; ++v) a[v] = vld1q_f32(A + 4 * v);
    for (int v = 0; v < NV; ++v) {
      c[0][v] = vfmaq_laneq_f32(c[0][v], a[v], w0, 0); c[1][v] = vfmaq_laneq_f32(c[1][v], a[v], w0, 1);
      c[2][v] = vfmaq_laneq_f32(c[2][v], a[v], w0, 2); c[3][v] = vfmaq_laneq_f32(c[3][v], a[v], w0, 3);
      c[4][v] = vfmaq_laneq_f32(c[4][v], a[v], w1, 0); c[5][v] = vfmaq_laneq_f32(c[5][v], a[v], w1, 1);
      c[6][v] = vfmaq_laneq_f32(c[6][v], a[v], w1, 2); c[7][v] = vfmaq_laneq_f32(c[7][v], a[v], w1, 3);
    }
  }
  for (int i = 0; i < 8; ++i)
    for (int v = 0; v < NV; ++v) vst1q_f32(Y + i * ldy + 4 * v, c[i][v]);
}
template <>
void gemm_planar<float>(const float* A, int lda, int N, int K, const float* W, int ldw, int O, float* Y, int ldy) {
  int n0 = 0;
  for (; n0 + 12 <= N; n0 += 12)
    for (int o0 = 0; o0 < O; o0 += 8) gemm_tile_f32<3>(A + n0, lda, K, W + o0, ldw, Y + o0 * ldy + n0, ldy);
  for (; n0 + 8 <= N; n0 += 8)
    for (int o0 = 0; o0 < O; o0 += 8) gemm_tile_f32<2>(A + n0, lda, K, W + o0, ldw, Y + o0 * ldy + n0, ldy);
  for (; n0 < N; n0 += 4)
    for (int o0 = 0; o0 < O; o0 += 8) gemm_tile_f32<1>(A + n0, lda, K, W + o0, ldw, Y + o0 * ldy + n0, ldy);
}
#endif

template <class T> struct Worker {
  const Weights<T>& w;
  int nc = 1;
  std::vector<double> img;       // [nc][63][64] input (double, for centering), col 63 = 0
  std::vector<T> imgT, A1, sh;   // [nc][64][64] transposed input, I K^T, shifted image
  std::vector<T> xp;             // [nc][PS]
  std::vector<T> s1, s2;         // [nc][32][PS]
  std::vector<T> pack, Yb;       // [96][nc*64], [32][nc*64]
  std::vector<T> Vb, A3;         // pooling
  std::vector<T> field, X, Aln, QKV, KT, VR, ctx, hid, tmp;
  std::vector<T> logit16, t16, L63, Kp, G;
  std::vector<T> Sb, Pb, Wb;
  Worker(const Weights<T>& w_, int T_) : w(w_), nc(T_ + 1) {
    img.assign(nc * S * 64, 0.0); imgT.assign(nc * 4096, T(0)); A1.assign((nc + 1) * 4096, T(0)); sh.assign((nc + 2) * 4096, T(0));
    xp.assign(nc * PS, T(0));
    s1.assign((size_t)nc * CH * PS, T(0));
    s2.assign((size_t)nc * CH * PS, T(0));
    pack.resize(64 * nc * NX); Yb.resize(CH * nc * NX);
    Vb.resize(2 * NB * RS); A3.resize(NTOK * 96);
    field.resize(nc * NTOK * CH); X.resize(nc * NTOK * AD); Aln.resize(nc * NTOK * AD);
    QKV.resize(nc * NTOK * 3 * AD); KT.resize(nc * NH * HD * NTOK); VR.resize(nc * NH * NTOK * HD);
    ctx.resize(nc * NTOK * AD); hid.resize(nc * NTOK * FF); tmp.resize(nc * NTOK * FF);
    logit16.resize(nc * NTOK); t16.resize(NB * S); L63.resize(nc * NPIX);
    Kp.assign(nc * KW * KW, T(0)); G.assign(2 * nc * S * NX, T(0));
    Sb.resize(4 * NTOK); Pb.resize(4 * NTOK); Wb.resize(4 * NTOK);
  }

  // Dirichlet kernel of the odd-N periodic band-limited shift and d/du.
  static void dirichlet(double u, int N, double& D, double& dD) {
    const double pi = M_PI;
    if (std::abs(u) < 1e-3) {
      double a = pi * pi * (1.0 - 1.0 / ((double)N * N)) / 6.0;
      D = 1.0 - a * u * u; dD = -2.0 * a * u; return;
    }
    double s = std::sin(pi * u), s2 = std::sin(pi * u / N), c = std::cos(pi * u), c2 = std::cos(pi * u / N);
    D = s / (N * s2);
    dD = pi * (c * N * s2 - s * c2) / ((N * s2) * (N * s2));
  }

  // Row sums over 64-padded double rows: s0 = sum I*a, s1 = sum I*b.
  static inline void rowdot2(const double* I, const double* a, const double* b, double& s0, double& s1) {
    constexpr int L = LANES<double>;
    vt<double> p0{}, p1{}, q0{}, q1{};
    for (int j = 0; j < 64; j += 2 * L) {
      vt<double> x0 = ld(I + j), x1 = ld(I + j + L);
      p0 += x0 * ld(a + j); p1 += x1 * ld(a + j + L);
      q0 += x0 * ld(b + j); q1 += x1 * ld(b + j + L);
    }
    s0 = hsum(p0 + p1); s1 = hsum(q0 + q1);
  }

  __attribute__((noinline)) void center() {
    const int T_ = nc - 1;
    const double sig2 = w.sigma * w.sigma;
    alignas(64) double xs[64], gx[64], gxx[64], hx[64], hxx[64], gy[S], hy[S];
    for (int j = 0; j < 64; ++j) xs[j] = j < S ? j - 31 : 0;
    Dual cx, cy;
    double u[(MAXT + 1) * S], ux[(MAXT + 1) * S], uh[S], uhx[S];
    for (int it = 0; it < w.iters; ++it) {
      for (int j = 0; j < S; ++j) {
        double a = xs[j] - cx.v, b = xs[j] - cy.v;
        gx[j] = std::exp(-a * a / (2 * sig2)); hx[j] = gx[j] * a / sig2;
        gy[j] = std::exp(-b * b / (2 * sig2)); hy[j] = gy[j] * b / sig2;
        gxx[j] = gx[j] * xs[j]; hxx[j] = hx[j] * xs[j];
      }
      gx[63] = gxx[63] = hx[63] = hxx[63] = 0;
      for (int c = 0; c < nc; ++c)
        for (int i = 0; i < S; ++i) rowdot2(img.data() + (c * S + i) * 64, gx, gxx, u[c * S + i], ux[c * S + i]);
      if (T_)
        for (int i = 0; i < S; ++i) rowdot2(img.data() + i * 64, hx, hxx, uh[i], uhx[i]);
      Dual flux, sx, sy;
      for (int i = 0; i < S; ++i) {
        flux.v += gy[i] * u[i]; sx.v += gy[i] * ux[i]; sy.v += gy[i] * xs[i] * u[i];
      }
      for (int t = 0; t < T_; ++t) {
        double f = 0, a = 0, b = 0;
        for (int i = 0; i < S; ++i) {
          double ut = gy[i] * u[(t + 1) * S + i] + gy[i] * cx.d[t] * uh[i] + cy.d[t] * hy[i] * u[i];
          double uxt = gy[i] * ux[(t + 1) * S + i] + gy[i] * cx.d[t] * uhx[i] + cy.d[t] * hy[i] * ux[i];
          f += ut; a += uxt; b += xs[i] * ut;
        }
        flux.d[t] = f; sx.d[t] = a; sy.d[t] = b;
      }
      Dual one = dd(1.0);
      Dual f2 = flux * flux + one;
      Dual den = chain(f2, std::sqrt(f2.v), 0.5 / std::sqrt(f2.v));
      Dual nx = sx / den, ny = sy / den;
      Dual q = (nx * nx + ny * ny) * (1.0 / (w.max_shift * w.max_shift));
      Dual q2 = q * q + one;
      Dual bound = chain(q2, std::pow(q2.v, 0.25), 0.25 * std::pow(q2.v, -0.75));
      cx = nx / bound; cy = ny / bound;
    }
    // Shift: out = Ky I Kx^T with K[n][m] = D(n - m + c), as planar GEMMs on
    // 64-padded matrices: A_c = I_c Kx^T uses W = I_c^T, out = Ky A uses W = Ky^T.
    double Dx[2 * S - 1], dDx[2 * S - 1], Dy[2 * S - 1], dDy[2 * S - 1];
    for (int t = -(S - 1); t <= S - 1; ++t) {
      dirichlet(t + cx.v, w.npad, Dx[t + S - 1], dDx[t + S - 1]);
      dirichlet(t + cy.v, w.npad, Dy[t + S - 1], dDy[t + S - 1]);
    }
    alignas(64) T KxT[4096] = {}, dKxT[4096] = {}, KyT[4096] = {}, dKyT[4096] = {};
    for (int j = 0; j < S; ++j)
      for (int b = 0; b < S; ++b) {
        KxT[j * 64 + b] = T(Dx[b - j + S - 1]); dKxT[j * 64 + b] = T(dDx[b - j + S - 1]);  // Kx[b][j]
        KyT[j * 64 + b] = T(Dy[b - j + S - 1]); dKyT[j * 64 + b] = T(dDy[b - j + S - 1]);  // Ky[b][j]
      }
    for (int c = 0; c < nc; ++c)
      for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) imgT[c * 4096 + j * 64 + i] = T(img[(c * S + i) * 64 + j]);
    for (int c = 0; c < nc; ++c)
      gemm_planar(KxT, 64, 64, S, imgT.data() + c * 4096, 64, 64, A1.data() + c * 4096, 64);
    T* out = sh.data();
    gemm_planar(A1.data(), 64, 64, S, KyT, 64, 64, out, 64);
    if (T_) {
      T* Ap = A1.data() + nc * 4096;                  // I_0 dKx^T
      gemm_planar(dKxT, 64, 64, S, imgT.data(), 64, 64, Ap, 64);
      T* KdA = sh.data() + nc * 4096;                 // dKy A_0
      gemm_planar(A1.data(), 64, 64, S, dKyT, 64, 64, KdA, 64);
      for (int t = 1; t <= T_; ++t) {
        T* At = A1.data() + t * 4096;
        const T ccx = T(cx.d[t - 1]), ccy = T(cy.d[t - 1]);
        for (int i = 0; i < 4096; ++i) At[i] += ccx * Ap[i];
        T* o = out + t * 4096;
        gemm_planar(At, 64, 64, S, KyT, 64, 64, o, 64);
        for (int i = 0; i < 4096; ++i) o[i] += ccy * KdA[i];
      }
    }
  }

  __attribute__((noinline)) void normalize() {
    // brightness = 100 sum(x w); x /= 1 + softplus_beta(brightness - 1)
    double bsum[MAXT + 1] = {0, 0, 0, 0, 0};
    for (int c = 0; c < nc; ++c) {
      const T* x = sh.data() + c * 4096;
      double s = 0;
      for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) s += double(x[i * 64 + j]) * double(w.wb[i * S + j]);
      bsum[c] = 100.0 * s;
    }
    double z = bsum[0] - 1.0, bz = w.nl_beta * z, sp, dsp;
    if (bz > w.nl_threshold) { sp = z; dsp = 1.0; }
    else { sp = std::log1p(std::exp(bz)) / w.nl_beta; dsp = 1.0 / (1.0 + std::exp(-bz)); }
    double sc = 1.0 / (1.0 + sp);
    const T* x0 = sh.data();
    for (int c = 0; c < nc; ++c) {
      const T* x = sh.data() + c * 4096;
      T* o = xp.data() + c * PS;
      T a = T(sc), bcoef = c ? T(-sc * sc * dsp * bsum[c]) : T(0);
      for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j)
          o[(i + 1) * RS + j + 1] = a * x[i * 64 + j] + bcoef * x0[i * 64 + j];
    }
  }

  // Dual LayerNorm over the 32 channel rows of Yb (vectorized over pixels),
  // GELU, plus rs[o] * residual, written to row y of the `outp` planes.
  // Residual of (comp c, channel o) row starts at res + c*res_cs + o*res_os.
  template <bool D>
  __attribute__((noinline)) void ln_gelu_rows(int y, const T* bias, const T* g, const T* b, T eps, T* outp,
                                               const T* res, size_t res_cs, size_t res_os, const T* rs) {
    constexpr int L = LANES<T>;
    const int lw = nc * NX;
    alignas(64) T nbuf[CH * L], dbuf[CH * L];
    const size_t row = (size_t)(y + 1) * RS + 1;
    const T inv_ch = T(1) / T(CH);
    for (int v = 0; v < NX; v += L) {
      const T* y0 = Yb.data() + v;
      vt<T> s0{}, s1_{}, s2_{}, s3{};
      for (int o = 0; o < CH; o += 4) {
        vt<T> a0 = ld(y0 + o * lw) + bias[o], a1 = ld(y0 + (o + 1) * lw) + bias[o + 1];
        vt<T> a2 = ld(y0 + (o + 2) * lw) + bias[o + 2], a3 = ld(y0 + (o + 3) * lw) + bias[o + 3];
        st(nbuf + o * L, a0); st(nbuf + (o + 1) * L, a1); st(nbuf + (o + 2) * L, a2); st(nbuf + (o + 3) * L, a3);
        s0 += a0; s1_ += a1; s2_ += a2; s3 += a3;
      }
      vt<T> mu = ((s0 + s1_) + (s2_ + s3)) * inv_ch;
      s0 = s1_ = s2_ = s3 = vt<T>{};
      for (int o = 0; o < CH; o += 4) {
        vt<T> a0 = ld(nbuf + o * L) - mu, a1 = ld(nbuf + (o + 1) * L) - mu;
        vt<T> a2 = ld(nbuf + (o + 2) * L) - mu, a3 = ld(nbuf + (o + 3) * L) - mu;
        s0 += a0 * a0; s1_ += a1 * a1; s2_ += a2 * a2; s3 += a3 * a3;
      }
      vt<T> r = ((s0 + s1_) + (s2_ + s3)) * inv_ch + eps;
      for (int l = 0; l < L; ++l) r[l] = T(1) / std::sqrt(r[l]);
      vt<T> mr = mu * r;
      const T* r0 = res + row + v;
      T* o0 = outp + row + v;
      for (int o = 0; o < CH; o += 4) {  // 4 independent GELU chains
        vt<T> n[4], z[4], G_[4], dG[4];
        for (int k = 0; k < 4; ++k) { n[k] = ld(nbuf + (o + k) * L) * r - mr; z[k] = n[k] * g[o + k] + b[o + k]; }
        for (int k = 0; k < 4; ++k) gelu(z[k], G_[k], dG[k], D);
        for (int k = 0; k < 4; ++k) {
          st(o0 + (size_t)(o + k) * PS, G_[k] + ld(r0 + (o + k) * res_os) * rs[o + k]);
          if (D) { st(nbuf + (o + k) * L, n[k]); st(dbuf + (o + k) * L, dG[k] * (r * g[o + k])); }
        }
      }
      if (!D) continue;
      for (int c = 1; c < nc; ++c) {
        const T* yt = Yb.data() + c * NX + v;
        vt<T> a0{}, a1{}, p0{}, p1{};
        for (int o = 0; o < CH; o += 2) {
          vt<T> d0 = ld(yt + o * lw), d1 = ld(yt + (o + 1) * lw);
          a0 += d0; a1 += d1;
          p0 += ld(nbuf + o * L) * d0; p1 += ld(nbuf + (o + 1) * L) * d1;
        }
        // sum_o n_o = 0, hence mean(n (y - m)) = mean(n y).
        vt<T> m = (a0 + a1) * inv_ch, p = (p0 + p1) * inv_ch;
        T* oc = outp + (size_t)c * CH * PS + row + v;
        const T* rc = res + c * res_cs + row + v;
        for (int o = 0; o < CH; ++o)
          st(oc + (size_t)o * PS,
             ld(dbuf + o * L) * (ld(yt + o * lw) - m - ld(nbuf + o * L) * p) + ld(rc + o * res_os) * rs[o]);
      }
    }
    for (int c = 0; c < nc; ++c)
      for (int o = 0; o < CH; ++o) outp[(size_t)(c * CH + o) * PS + row + S] = T(0);  // right border
  }
  void ln_gelu(int y, const T* bias, const T* g, const T* b, T eps, T* outp, const T* res, size_t res_cs,
               size_t res_os, const T* rs) {
    if (nc > 1) ln_gelu_rows<true>(y, bias, g, b, eps, outp, res, res_cs, res_os, rs);
    else ln_gelu_rows<false>(y, bias, g, b, eps, outp, res, res_cs, res_os, rs);
  }

  __attribute__((noinline)) void layer1() {
    constexpr int L = LANES<T>;
    const T* l1 = w.l1.data();
    const int lw = nc * NX;
    for (int y = 0; y < S; ++y) {
      for (int c = 0; c < nc; ++c) {
        const T* x = xp.data() + c * PS + (y + 1) * RS + 1;
        for (int v = 0; v < NX; v += L) {
          vt<T> xc = ld(x + v);
          vt<T> cr = ld(x + v - RS) + ld(x + v + RS) + ld(x + v - 1) + ld(x + v + 1);
          vt<T> dg = ld(x + v - RS - 1) + ld(x + v - RS + 1) + ld(x + v + RS - 1) + ld(x + v + RS + 1);
          for (int o = 0; o < CH; ++o)
            st(Yb.data() + o * lw + c * NX + v, xc * l1[o] + cr * l1[CH + o] + dg * l1[2 * CH + o]);
        }
      }
      ln_gelu(y, l1 + 3 * CH, l1 + 4 * CH, l1 + 5 * CH, w.eps1, s1.data(), xp.data(), PS, 0, l1 + 6 * CH);
    }
  }

  __attribute__((noinline)) void layer2() {
    constexpr int L = LANES<T>;
    const int lw = nc * NX;
    T rs[CH];
    for (int o = 0; o < CH; ++o) rs[o] = w.res2;
    for (int y = 0; y < S; ++y) {
      for (int c = 0; c < nc; ++c)
        for (int i = 0; i < CH; ++i) {
          const T* s = s1.data() + (size_t)(c * CH + i) * PS + (y + 1) * RS + 1;
          T* a0 = pack.data() + (size_t)i * lw + c * NX;
          T* a1 = a0 + (size_t)CH * lw;
          for (int v = 0; v < NX; v += L) {
            st(a0 + v, ld(s + v));
            vt<T> cr = (ld(s + v - RS) + ld(s + v + RS)) + (ld(s + v - 1) + ld(s + v + 1));
            vt<T> dg = (ld(s + v - RS - 1) + ld(s + v - RS + 1)) + (ld(s + v + RS - 1) + ld(s + v + RS + 1));
            st(a1 + v, cr * T(4) + dg);
          }
        }
      gemm_planar(pack.data(), lw, lw, 64, w.W2.data(), 32, 32, Yb.data(), lw);
      ln_gelu(y, w.b2.data(), w.g2.data(), w.beta2.data(), w.eps2, s2.data(), s1.data(), (size_t)CH * PS, PS, rs);
    }
  }

  // Adaptive-avg-pooled F8 inputs [x, cross, diag] per 16x16 token. With
  // A0 = sum over the bin's rows, A1 = sum over rows shifted by -1 and +1,
  // and box0/boxpm = sums over the bin's columns (unshifted / shifted -1 + +1):
  //   center = box0(A0), cross = box0(A1) + boxpm(A0), diag = boxpm(A1).
  // Columns are vectorized over x; rows are transposed and vectorized over by.
  __attribute__((noinline)) void pooled_f8() {
    constexpr int L = LANES<T>;
    constexpr int NR = 2 * NB;  // rows: A0 for 16 by, then A1 for 16 by
    alignas(64) T Vt[RS * NR];  // [x][row]
    T* Ab = Vb.data();          // [row][RS]
    for (int c = 0; c < nc; ++c) {
      T* A3c = A3.data();
      for (int o = 0; o < CH; ++o) {
        const T* P = s2.data() + (size_t)(c * CH + o) * PS;
        for (int by = 0; by < NB; ++by) {
          const int y0 = w.pb0[by], y1 = w.pb1[by];  // padded rows [y0+1, y1]
          T* a0 = Ab + by * RS;
          T* a1 = Ab + (NB + by) * RS;
          std::memcpy(a0, P + (y0 + 1) * RS, RS * sizeof(T));
          for (int Y = y0 + 2; Y <= y1; ++Y) {
            const T* p = P + Y * RS;
            for (int x = 0; x < RS; x += L) st(a0 + x, ld(a0 + x) + ld(p + x));
          }
          const T *pa = P + y0 * RS, *pb = P + y1 * RS, *pc = P + (y1 + 1) * RS, *pd = P + (y0 + 1) * RS;
          for (int x = 0; x < RS; x += L) {
            vt<T> s0 = ld(a0 + x);
            st(a1 + x, s0 + s0 + ((ld(pa + x) - ld(pb + x)) + (ld(pc + x) - ld(pd + x))));
          }
        }
        if constexpr (L == 4) {  // 4x4 block transposes
          for (int r = 0; r < NR; r += 4)
            for (int x = 0; x < RS; x += 4) {
              vt<T> a = ld(Ab + r * RS + x), b = ld(Ab + (r + 1) * RS + x);
              vt<T> c_ = ld(Ab + (r + 2) * RS + x), d = ld(Ab + (r + 3) * RS + x);
              vt<T> t0 = __builtin_shufflevector(a, b, 0, 4, 1, 5), t1 = __builtin_shufflevector(a, b, 2, 6, 3, 7);
              vt<T> t2 = __builtin_shufflevector(c_, d, 0, 4, 1, 5), t3 = __builtin_shufflevector(c_, d, 2, 6, 3, 7);
              st(Vt + x * NR + r, __builtin_shufflevector(t0, t2, 0, 1, 4, 5));
              st(Vt + (x + 1) * NR + r, __builtin_shufflevector(t0, t2, 2, 3, 6, 7));
              st(Vt + (x + 2) * NR + r, __builtin_shufflevector(t1, t3, 0, 1, 4, 5));
              st(Vt + (x + 3) * NR + r, __builtin_shufflevector(t1, t3, 2, 3, 6, 7));
            }
        } else {
          for (int r = 0; r < NR; ++r)
            for (int x = 0; x < RS; ++x) Vt[x * NR + r] = Ab[r * RS + x];
        }
        for (int bx = 0; bx < NB; ++bx) {
          const int x0 = w.pb0[bx], x1 = w.pb1[bx];
          vt<T> b0[NR / L], bp[NR / L];
          for (int u = 0; u < NR / L; ++u) {
            const T* v = Vt + u * L;
            vt<T> s0 = ld(v + (x0 + 1) * NR);
            for (int X = x0 + 2; X <= x1; ++X) s0 += ld(v + X * NR);
            b0[u] = s0;
            bp[u] = s0 + s0 + ((ld(v + x0 * NR) - ld(v + x1 * NR)) + (ld(v + (x1 + 1) * NR) - ld(v + (x0 + 1) * NR)));
          }
          for (int u = 0; u < NB / L; ++u) {
            vt<T> cen = b0[u], cro = b0[u + NB / L] + bp[u], dia = bp[u + NB / L];
            for (int l = 0; l < L; ++l) {
              T* a = A3c + ((u * L + l) * NB + bx) * 96 + o;
              a[0] = cen[l]; a[CH] = cro[l]; a[2 * CH] = dia[l];
            }
          }
        }
      }
      for (int by = 0; by < NB; ++by)
        for (int bx = 0; bx < NB; ++bx) {
          T inv = T(1) / T((w.pb1[by] - w.pb0[by]) * (w.pb1[bx] - w.pb0[bx]));
          T* a = A3c + (by * NB + bx) * 96;
          for (int k = 0; k < 96; ++k) a[k] *= inv;
        }
      gemm32(A3c, 96, NTOK, 96, w.W3.data(), 32, field.data() + c * NTOK * CH, 32, c ? nullptr : w.b3.data());
    }
  }

#if defined(__aarch64__) && !defined(D4_GENERIC)
  // NEON float path of attn_block: queries pre-transposed to [e][row] so one
  // vector feeds 4 rows by lane; P/W loaded 4 keys at a time (lane FMA).
  void softmax_rows4(float* Sb_, float* Pb_) {
    for (int r = 0; r < 4; ++r) {
      float* s = Sb_ + r * NTOK; float* P = Pb_ + r * NTOK;
      float32x4_t m0 = vld1q_f32(s), m1 = vld1q_f32(s + 4), m2 = vld1q_f32(s + 8), m3 = vld1q_f32(s + 12);
      for (int j = 16; j < NTOK; j += 16) {
        m0 = vmaxq_f32(m0, vld1q_f32(s + j)); m1 = vmaxq_f32(m1, vld1q_f32(s + j + 4));
        m2 = vmaxq_f32(m2, vld1q_f32(s + j + 8)); m3 = vmaxq_f32(m3, vld1q_f32(s + j + 12));
      }
      const float m = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
      vt<float> a0{}, a1{}, a2{}, a3{};
      for (int j = 0; j < NTOK; j += 16) {
        vt<float> p0 = vexp_neg(ld(s + j) - m), p1 = vexp_neg(ld(s + j + 4) - m);
        vt<float> p2 = vexp_neg(ld(s + j + 8) - m), p3 = vexp_neg(ld(s + j + 12) - m);
        st(P + j, p0); st(P + j + 4, p1); st(P + j + 8, p2); st(P + j + 12, p3);
        a0 += p0; a1 += p1; a2 += p2; a3 += p3;
      }
      const float32x4_t inv = vdupq_n_f32(1.f / hsum((a0 + a1) + (a2 + a3)));
      for (int j = 0; j < NTOK; j += 4) vst1q_f32(P + j, vmulq_f32(vld1q_f32(P + j), inv));
    }
  }
  __attribute__((noinline)) void attn_block_neon(int h, int i0) {
    const int T_ = nc - 1;
    float *Sb_ = (float*)Sb.data(), *Pb_ = (float*)Pb.data(), *Wb_ = (float*)Wb.data();
    const float* K0 = (const float*)KT.data() + h * HD * NTOK;
    const float* V0 = (const float*)VR.data() + h * NTOK * HD;
    const float* Q = (const float*)QKV.data();
    alignas(16) float q0T[HD][4], qtT[HD][4];
    for (int r = 0; r < 4; ++r)
      for (int e = 0; e < HD; ++e) q0T[e][r] = Q[(i0 + r) * 3 * AD + h * HD + e];
    // primal scores: 4 rows x 16 keys
    for (int j = 0; j < NTOK; j += 16) {
      float32x4_t a[4][4];
      for (int r = 0; r < 4; ++r)
        for (int u = 0; u < 4; ++u) a[r][u] = vdupq_n_f32(0.f);
      for (int e = 0; e < HD; ++e) {
        const float32x4_t q = vld1q_f32(q0T[e]);
        float32x4_t k[4];
        for (int u = 0; u < 4; ++u) k[u] = vld1q_f32(K0 + e * NTOK + j + 4 * u);
        for (int u = 0; u < 4; ++u) {
          a[0][u] = vfmaq_laneq_f32(a[0][u], k[u], q, 0); a[1][u] = vfmaq_laneq_f32(a[1][u], k[u], q, 1);
          a[2][u] = vfmaq_laneq_f32(a[2][u], k[u], q, 2); a[3][u] = vfmaq_laneq_f32(a[3][u], k[u], q, 3);
        }
      }
      for (int r = 0; r < 4; ++r)
        for (int u = 0; u < 4; ++u) vst1q_f32(Sb_ + r * NTOK + j + 4 * u, a[r][u]);
    }
    softmax_rows4(Sb_, Pb_);
    // context = P V: per 4 keys, P rows as vectors, V rows (8 = 2 vectors) by lane;
    // two accumulator sets (even/odd key groups) for 16 independent chains.
#define D4_PV_STEP(C_, L_)                                                          \
    {                                                                               \
      const float* vj = V0 + (j + L_) * HD;                                         \
      const float32x4_t v0 = vld1q_f32(vj), v1 = vld1q_f32(vj + 4);                 \
      for (int r = 0; r < 4; ++r) {                                                 \
        C_[r][0] = vfmaq_laneq_f32(C_[r][0], v0, p[r], L_);                         \
        C_[r][1] = vfmaq_laneq_f32(C_[r][1], v1, p[r], L_);                         \
      }                                                                             \
    }
    {
      float32x4_t c[4][2], c2[4][2];
      for (int r = 0; r < 4; ++r) c[r][0] = c[r][1] = c2[r][0] = c2[r][1] = vdupq_n_f32(0.f);
      for (int j = 0; j < NTOK; j += 4) {
        float32x4_t p[4];
        for (int r = 0; r < 4; ++r) p[r] = vld1q_f32(Pb_ + r * NTOK + j);
        D4_PV_STEP(c, 0) D4_PV_STEP(c2, 1) D4_PV_STEP(c, 2) D4_PV_STEP(c2, 3)
      }
      float* out = (float*)ctx.data();
      for (int r = 0; r < 4; ++r) {
        vst1q_f32(out + (i0 + r) * AD + h * HD, vaddq_f32(c[r][0], c2[r][0]));
        vst1q_f32(out + (i0 + r) * AD + h * HD + 4, vaddq_f32(c[r][1], c2[r][1]));
      }
    }
#undef D4_PV_STEP
    for (int t = 1; t <= T_; ++t) {
      const float* Kt = (const float*)KT.data() + (t * NH + h) * HD * NTOK;
      const float* Vt = (const float*)VR.data() + (t * NH + h) * NTOK * HD;
      for (int r = 0; r < 4; ++r)
        for (int e = 0; e < HD; ++e) qtT[e][r] = Q[(t * NTOK + i0 + r) * 3 * AD + h * HD + e];
      for (int j = 0; j < NTOK; j += 8) {  // tangent scores Qt K0^T + Q0 Kt^T
        float32x4_t a[4][2], b[4][2];
        for (int r = 0; r < 4; ++r) a[r][0] = a[r][1] = b[r][0] = b[r][1] = vdupq_n_f32(0.f);
        for (int e = 0; e < HD; ++e) {
          const float32x4_t qa = vld1q_f32(qtT[e]), qb = vld1q_f32(q0T[e]);
          const float32x4_t k0 = vld1q_f32(K0 + e * NTOK + j), k1 = vld1q_f32(K0 + e * NTOK + j + 4);
          const float32x4_t t0 = vld1q_f32(Kt + e * NTOK + j), t1 = vld1q_f32(Kt + e * NTOK + j + 4);
          a[0][0] = vfmaq_laneq_f32(a[0][0], k0, qa, 0); a[0][1] = vfmaq_laneq_f32(a[0][1], k1, qa, 0);
          a[1][0] = vfmaq_laneq_f32(a[1][0], k0, qa, 1); a[1][1] = vfmaq_laneq_f32(a[1][1], k1, qa, 1);
          a[2][0] = vfmaq_laneq_f32(a[2][0], k0, qa, 2); a[2][1] = vfmaq_laneq_f32(a[2][1], k1, qa, 2);
          a[3][0] = vfmaq_laneq_f32(a[3][0], k0, qa, 3); a[3][1] = vfmaq_laneq_f32(a[3][1], k1, qa, 3);
          b[0][0] = vfmaq_laneq_f32(b[0][0], t0, qb, 0); b[0][1] = vfmaq_laneq_f32(b[0][1], t1, qb, 0);
          b[1][0] = vfmaq_laneq_f32(b[1][0], t0, qb, 1); b[1][1] = vfmaq_laneq_f32(b[1][1], t1, qb, 1);
          b[2][0] = vfmaq_laneq_f32(b[2][0], t0, qb, 2); b[2][1] = vfmaq_laneq_f32(b[2][1], t1, qb, 2);
          b[3][0] = vfmaq_laneq_f32(b[3][0], t0, qb, 3); b[3][1] = vfmaq_laneq_f32(b[3][1], t1, qb, 3);
        }
        for (int r = 0; r < 4; ++r) {
          vst1q_f32(Sb_ + r * NTOK + j, vaddq_f32(a[r][0], b[r][0]));
          vst1q_f32(Sb_ + r * NTOK + j + 4, vaddq_f32(a[r][1], b[r][1]));
        }
      }
      for (int r = 0; r < 4; ++r) {
        const float* P = Pb_ + r * NTOK; const float* s = Sb_ + r * NTOK; float* W_ = Wb_ + r * NTOK;
        float32x4_t m0 = vdupq_n_f32(0.f), m1 = m0;
        for (int j = 0; j < NTOK; j += 8) {
          m0 = vfmaq_f32(m0, vld1q_f32(P + j), vld1q_f32(s + j));
          m1 = vfmaq_f32(m1, vld1q_f32(P + j + 4), vld1q_f32(s + j + 4));
        }
        const float32x4_t mb = vdupq_n_f32(vaddvq_f32(vaddq_f32(m0, m1)));
        for (int j = 0; j < NTOK; j += 4) vst1q_f32(W_ + j, vmulq_f32(vld1q_f32(P + j), vsubq_f32(vld1q_f32(s + j), mb)));
      }
      // Ct = W V0 + P Vt
      float32x4_t c[4][2], d[4][2];
      for (int r = 0; r < 4; ++r) c[r][0] = c[r][1] = d[r][0] = d[r][1] = vdupq_n_f32(0.f);
      for (int j = 0; j < NTOK; j += 4) {
        float32x4_t wv[4], pv[4];
        for (int r = 0; r < 4; ++r) { wv[r] = vld1q_f32(Wb_ + r * NTOK + j); pv[r] = vld1q_f32(Pb_ + r * NTOK + j); }
#define D4_TPV_STEP(L_)                                                                   \
        {                                                                                 \
          const float* a_ = V0 + (j + L_) * HD; const float* b_ = Vt + (j + L_) * HD;     \
          const float32x4_t x0 = vld1q_f32(a_), x1 = vld1q_f32(a_ + 4);                   \
          const float32x4_t y0 = vld1q_f32(b_), y1 = vld1q_f32(b_ + 4);                   \
          for (int r = 0; r < 4; ++r) {                                                   \
            c[r][0] = vfmaq_laneq_f32(c[r][0], x0, wv[r], L_);                            \
            c[r][1] = vfmaq_laneq_f32(c[r][1], x1, wv[r], L_);                            \
            d[r][0] = vfmaq_laneq_f32(d[r][0], y0, pv[r], L_);                            \
            d[r][1] = vfmaq_laneq_f32(d[r][1], y1, pv[r], L_);                            \
          }                                                                               \
        }
        D4_TPV_STEP(0) D4_TPV_STEP(1) D4_TPV_STEP(2) D4_TPV_STEP(3)
#undef D4_TPV_STEP
      }
      float* out = (float*)ctx.data() + t * NTOK * AD;
      for (int r = 0; r < 4; ++r) {
        vst1q_f32(out + (i0 + r) * AD + h * HD, vaddq_f32(c[r][0], d[r][0]));
        vst1q_f32(out + (i0 + r) * AD + h * HD + 4, vaddq_f32(c[r][1], d[r][1]));
      }
    }
  }
#endif

  // Softmax attention (primal + tangents) for queries i0..i0+3 of head h.
  __attribute__((noinline)) void attn_block(int h, int i0) {
    constexpr int L = LANES<T>, HV = HD / L;
    const int T_ = nc - 1;
    T *Sb_ = Sb.data(), *Pb_ = Pb.data(), *Wb_ = Wb.data();
    const T* K0 = KT.data() + h * HD * NTOK;
    const T* V0 = VR.data() + h * NTOK * HD;
    T q0[4][HD];
    for (int r = 0; r < 4; ++r)
      for (int e = 0; e < HD; ++e) q0[r][e] = QKV[(i0 + r) * 3 * AD + h * HD + e];
    for (int j = 0; j < NTOK; j += 4 * L) {  // 4 queries x 4 key vectors: 16 chains
      vt<T> a[4][4] = {};
      for (int e = 0; e < HD; ++e) {
        vt<T> k[4];
        for (int u = 0; u < 4; ++u) k[u] = ld(K0 + e * NTOK + j + u * L);
        for (int r = 0; r < 4; ++r) {
          T qe = q0[r][e];
          for (int u = 0; u < 4; ++u) a[r][u] += k[u] * qe;
        }
      }
      for (int r = 0; r < 4; ++r)
        for (int u = 0; u < 4; ++u) st(Sb_ + r * NTOK + j + u * L, a[r][u]);
    }
    for (int r = 0; r < 4; ++r) {
      T* s = Sb_ + r * NTOK; T* P = Pb_ + r * NTOK;
      vt<T> m4[4] = {ld(s), ld(s + L), ld(s + 2 * L), ld(s + 3 * L)};
      for (int j = 4 * L; j < NTOK; j += 4 * L)
        for (int u = 0; u < 4; ++u) { vt<T> x = ld(s + j + u * L); m4[u] = m4[u] > x ? m4[u] : x; }
      for (int u = 1; u < 4; ++u) m4[0] = m4[0] > m4[u] ? m4[0] : m4[u];
      T m = m4[0][0];
      for (int l = 1; l < L; ++l) m = std::max(m, m4[0][l]);
      vt<T> acc0{}, acc1{}, acc2{}, acc3{};
      for (int j = 0; j < NTOK; j += 4 * L) {
        vt<T> p0 = vexp(ld(s + j) - m), p1 = vexp(ld(s + j + L) - m);
        vt<T> p2 = vexp(ld(s + j + 2 * L) - m), p3 = vexp(ld(s + j + 3 * L) - m);
        st(P + j, p0); st(P + j + L, p1); st(P + j + 2 * L, p2); st(P + j + 3 * L, p3);
        acc0 += p0; acc1 += p1; acc2 += p2; acc3 += p3;
      }
      T inv = T(1) / hsum((acc0 + acc1) + (acc2 + acc3));
      for (int j = 0; j < NTOK; j += L) st(P + j, ld(P + j) * inv);
    }
    {
      vt<T> acc[2][4][HV] = {};  // parity sets: 16 independent chains
      for (int j = 0; j < NTOK; j += 2) {
        vt<T> va[HV], vb[HV];
        for (int u = 0; u < HV; ++u) { va[u] = ld(V0 + j * HD + u * L); vb[u] = ld(V0 + (j + 1) * HD + u * L); }
        for (int r = 0; r < 4; ++r) {
          T pa = Pb_[r * NTOK + j], pb = Pb_[r * NTOK + j + 1];
          for (int u = 0; u < HV; ++u) { acc[0][r][u] += va[u] * pa; acc[1][r][u] += vb[u] * pb; }
        }
      }
      for (int r = 0; r < 4; ++r)
        for (int u = 0; u < HV; ++u) st(ctx.data() + (i0 + r) * AD + h * HD + u * L, acc[0][r][u] + acc[1][r][u]);
    }
    for (int t = 1; t <= T_; ++t) {
      const T* Kt = KT.data() + (t * NH + h) * HD * NTOK;
      const T* Vt = VR.data() + (t * NH + h) * NTOK * HD;
      T qt[4][HD];
      for (int r = 0; r < 4; ++r)
        for (int e = 0; e < HD; ++e) qt[r][e] = QKV[(t * NTOK + i0 + r) * 3 * AD + h * HD + e];
      for (int j = 0; j < NTOK; j += 2 * L) {
        vt<T> a[4][2][2] = {};  // rows x key vectors x (Qt K0, Q0 Kt)
        for (int e = 0; e < HD; ++e) {
          vt<T> k0[2], k1[2];
          for (int u = 0; u < 2; ++u) { k0[u] = ld(K0 + e * NTOK + j + u * L); k1[u] = ld(Kt + e * NTOK + j + u * L); }
          for (int r = 0; r < 4; ++r) {
            T a_ = qt[r][e], b_ = q0[r][e];
            for (int u = 0; u < 2; ++u) { a[r][u][0] += k0[u] * a_; a[r][u][1] += k1[u] * b_; }
          }
        }
        for (int r = 0; r < 4; ++r)
          for (int u = 0; u < 2; ++u) st(Sb_ + r * NTOK + j + u * L, a[r][u][0] + a[r][u][1]);
      }
      for (int r = 0; r < 4; ++r) {
        const T* P = Pb_ + r * NTOK; const T* s = Sb_ + r * NTOK; T* W_ = Wb_ + r * NTOK;
        vt<T> mm[2] = {};
        for (int j = 0; j < NTOK; j += 2 * L) { mm[0] += ld(P + j) * ld(s + j); mm[1] += ld(P + j + L) * ld(s + j + L); }
        T mbar = hsum(mm[0] + mm[1]);
        for (int j = 0; j < NTOK; j += L) st(W_ + j, ld(P + j) * (ld(s + j) - mbar));
      }
      vt<T> acc[2][4][HV] = {};  // separate chains for W V0 and P Vt
      for (int j = 0; j < NTOK; ++j) {
        vt<T> v0[HV], v1[HV];
        for (int u = 0; u < HV; ++u) { v0[u] = ld(V0 + j * HD + u * L); v1[u] = ld(Vt + j * HD + u * L); }
        for (int r = 0; r < 4; ++r) {
          T wv = Wb_[r * NTOK + j], p = Pb_[r * NTOK + j];
          for (int u = 0; u < HV; ++u) { acc[0][r][u] += v0[u] * wv; acc[1][r][u] += v1[u] * p; }
        }
      }
      for (int r = 0; r < 4; ++r)
        for (int u = 0; u < HV; ++u)
          st(ctx.data() + (t * NTOK + i0 + r) * AD + h * HD + u * L, acc[0][r][u] + acc[1][r][u]);
    }
  }

  __attribute__((noinline)) void attention() {
    constexpr int L = LANES<T>, HV = HD / L;
    const int T_ = nc - 1;
    T* yp[MAXT + 1]; T* op[MAXT + 1];
    for (int c = 0; c < nc; ++c)
      gemm32(field.data() + c * NTOK * CH, 32, NTOK, 32, w.Wtok.data(), 32, X.data() + c * NTOK * AD, AD, (const T*)nullptr);
    for (int i = 0; i < NTOK * AD; ++i) X[i] += w.tokbias[i];
    for (int n = 0; n < NTOK; ++n) {
      for (int c = 0; c < nc; ++c) { yp[c] = X.data() + (c * NTOK + n) * AD; op[c] = Aln.data() + (c * NTOK + n) * AD; }
      ln32(nc, yp, w.an_g.data(), w.an_b.data(), w.eps_a, op, false);
    }
    for (int c = 0; c < nc; ++c)
      gemm(Aln.data() + c * NTOK * AD, AD, NTOK, AD, w.Wqkv.data(), 3 * AD, QKV.data() + c * NTOK * 3 * AD, 3 * AD,
           c ? nullptr : w.bqkv.data());
    const T scale = T(1) / std::sqrt(T(HD));
    for (int c = 0; c < nc; ++c)
      for (int h = 0; h < NH; ++h) {
        T* kt = KT.data() + (c * NH + h) * HD * NTOK;
        T* vr = VR.data() + (c * NH + h) * NTOK * HD;
        for (int n = 0; n < NTOK; ++n) {
          const T* q = QKV.data() + (c * NTOK + n) * 3 * AD + h * HD;
          for (int e = 0; e < HD; ++e) { kt[e * NTOK + n] = q[AD + e] * scale; vr[n * HD + e] = q[2 * AD + e]; }
        }
      }
    for (int h = 0; h < NH; ++h)
      for (int i0 = 0; i0 < NTOK; i0 += 4) {
#if defined(__aarch64__) && !defined(D4_GENERIC)
        if constexpr (std::is_same_v<T, float>) { attn_block_neon(h, i0); continue; }
#endif
        attn_block(h, i0);
      }
    for (int c = 0; c < nc; ++c) {
      gemm32(ctx.data() + c * NTOK * AD, AD, NTOK, AD, w.Wo.data(), AD, tmp.data(), AD, c ? nullptr : w.bo.data());
      T* x = X.data() + c * NTOK * AD;
      for (int i = 0; i < NTOK * AD; ++i) x[i] += tmp[i];
    }
    for (int n = 0; n < NTOK; ++n) {
      for (int c = 0; c < nc; ++c) { yp[c] = X.data() + (c * NTOK + n) * AD; op[c] = Aln.data() + (c * NTOK + n) * AD; }
      ln32(nc, yp, w.fn_g.data(), w.fn_b.data(), w.eps_f, op, false);
    }
    for (int c = 0; c < nc; ++c)
      gemm(Aln.data() + c * NTOK * AD, AD, NTOK, AD, w.Wf1.data(), FF, hid.data() + c * NTOK * FF, FF,
           c ? nullptr : w.bf1.data());
    for (int i = 0; i < NTOK * FF; i += L) {
      vt<T> g, dg;
      gelu(ld(hid.data() + i), g, dg, T_ > 0);
      for (int t = 1; t <= T_; ++t) st(hid.data() + t * NTOK * FF + i, dg * ld(hid.data() + t * NTOK * FF + i));
      st(hid.data() + i, g);
    }
    for (int c = 0; c < nc; ++c) {
      gemm32(hid.data() + c * NTOK * FF, FF, NTOK, FF, w.Wf2.data(), AD, tmp.data(), AD, c ? nullptr : w.bf2.data());
      T* x = X.data() + c * NTOK * AD;
      for (int i = 0; i < NTOK * AD; ++i) x[i] += tmp[i];
    }
    for (int c = 0; c < nc; ++c)
      for (int n = 0; n < NTOK; ++n) {
        const T* x = X.data() + (c * NTOK + n) * AD;
        T a = c ? T(0) : w.bk;
        for (int k = 0; k < AD; ++k) a += x[k] * w.wk[k];
        logit16[c * NTOK + n] = a;
      }
  }

  __attribute__((noinline)) void pool_kernel() {
    constexpr int L = LANES<T>;
    const int T_ = nc - 1;
    for (int c = 0; c < nc; ++c) {
      matmul(logit16.data() + c * NTOK, w.MT.data(), t16.data(), NB, NB, S);  // [16][63]
      matmul(w.M.data(), t16.data(), L63.data() + c * NPIX, S, NB, S);       // [63][63]
    }
    T* l0 = L63.data();
    T m = -1e30;
    for (int i = 0; i < NPIX; ++i) { l0[i] -= w.sup[i]; m = std::max(m, l0[i]); }
    double z = 0;
    int i = 0;
    for (; i + L <= NPIX; i += L) { vt<T> e = vexp(ld(l0 + i) - m); st(l0 + i, e); z += hsum(e); }
    for (; i < NPIX; ++i) { l0[i] = std::exp(l0[i] - m); z += l0[i]; }
    T inv = T(1.0 / z);
    for (int k = 0; k < NPIX; ++k) l0[k] *= inv;
    for (int t = 1; t <= T_; ++t) {
      T* lt = L63.data() + t * NPIX;
      double mb = 0;
      for (int k = 0; k < NPIX; ++k) mb += double(l0[k]) * double(lt[k]);
      T mbt = T(mb);
      for (int k = 0; k < NPIX; ++k) lt[k] = l0[k] * (lt[k] - mbt);
    }
    // G_j(q) = sum_d B_j(d) K(q - d), K zero-padded by 2; rows stored with stride NX.
    for (int c = 0; c < nc; ++c) {
      T* kp = Kp.data() + c * KW * KW;
      for (int y = 0; y < S; ++y) std::memcpy(kp + (y + 2) * KW + 2, L63.data() + c * NPIX + y * S, S * sizeof(T));
      for (int j = 0; j < 2; ++j) {
        T* g = G.data() + (j * nc + c) * S * NX;
        std::fill(g, g + S * NX, T(0));
        for (int dy = -2; dy <= 2; ++dy)
          for (int dx = -2; dx <= 2; ++dx) {
            T b = w.B[j * 25 + (dy + 2) * 5 + dx + 2];
            if (b == T(0)) continue;
            for (int y = 0; y < S; ++y) {
              const T* src = kp + (y - dy + 2) * KW + 2 - dx;
              T* dst = g + y * NX;
#pragma GCC ivdep
              for (int x = 0; x < S; ++x) dst[x] += b * src[x];
            }
          }
      }
    }
  }

  __attribute__((noinline)) void head(double* e, double* R) {
    constexpr int L = LANES<T>;
    const int T_ = nc - 1;
    Dual vj[2][CH];
    for (int o = 0; o < CH; ++o) {
      vt<T> a[2][MAXT + 1] = {};
      const T* f0 = s2.data() + (size_t)o * PS + RS + 1;
      const T* g0 = G.data();
      const T* g1 = G.data() + nc * S * NX;
      for (int y = 0; y < S; ++y)
        for (int x = 0; x < NX; x += L) {
          vt<T> f = ld(f0 + y * RS + x);
          a[0][0] += f * ld(g0 + y * NX + x);
          a[1][0] += f * ld(g1 + y * NX + x);
        }
      for (int t = 1; t <= T_; ++t) {
        const T* ft = s2.data() + (size_t)(t * CH + o) * PS + RS + 1;
        const T* g0t = G.data() + t * S * NX;
        const T* g1t = G.data() + (nc + t) * S * NX;
        for (int y = 0; y < S; ++y)
          for (int x = 0; x < NX; x += L) {
            vt<T> f = ld(f0 + y * RS + x), d = ld(ft + y * RS + x);
            a[0][t] += d * ld(g0 + y * NX + x) + f * ld(g0t + y * NX + x);
            a[1][t] += d * ld(g1 + y * NX + x) + f * ld(g1t + y * NX + x);
          }
      }
      for (int j = 0; j < 2; ++j) {
        vj[j][o].v = hsum(a[j][0]);
        for (int t = 1; t <= T_; ++t) vj[j][o].d[t - 1] = hsum(a[j][t]);
      }
    }
    // spin gate
    Dual hcap[CH];
    for (int ch = 0; ch < CH; ++ch) {
      Dual r2 = (vj[0][ch] * vj[0][ch] + vj[1][ch] * vj[1][ch]) * (1.0 / (w.fscale[ch] * w.fscale[ch]));
      Dual r4 = r2 * r2;
      hcap[ch] = chain(r4, std::log1p(r4.v), 1.0 / (1.0 + r4.v));
    }
    auto act = [&](const Dual& z) -> Dual {
      if (w.gate_mode == 0) { double t = std::tanh(z.v); return chain(z, t, 1 - t * t); }
      double Phi = 0.5 * (1 + std::erf(z.v * M_SQRT1_2));
      double phi = std::exp(-0.5 * z.v * z.v) * 0.39894228040143267794;
      return chain(z, z.v * Phi, Phi + z.v * phi);
    };
    Dual a[GH];
    for (int k = 0; k < GH; ++k) {
      Dual zp = dd(w.gb1[k]), zm = dd(w.gb1[k]);
      for (int ch = 0; ch < CH; ++ch) { zp = zp + hcap[ch] * w.gW1[k * CH + ch]; zm = zm - hcap[ch] * w.gW1[k * CH + ch]; }
      if (w.gate_mode == 0) a[k] = act(zp);
      else a[k] = (act(zp) - act(zm)) * (w.gate_mode == 1 ? 0.5 : 1.0);
    }
    Dual out[2];
    for (int ch = 0; ch < CH; ++ch) {
      Dual u = dd(w.gate_mode == 0 ? w.gb2[ch] : 0.0);
      for (int k = 0; k < GH; ++k) u = u + a[k] * w.gW2[ch * GH + k];
      double th = std::tanh(u.v);
      Dual gate = chain(u, 1.0 + 0.5 * th, 0.5 * (1 - th * th));
      Dual cg = gate * w.common[ch];
      out[0] = out[0] + vj[0][ch] * cg;
      out[1] = out[1] + vj[1][ch] * cg;
    }
    Dual q = (out[0] * out[0] + out[1] * out[1]) * w.alpha + dd(1.0);
    Dual rs = chain(q, w.gain / std::sqrt(q.v), -0.5 * w.gain / (q.v * std::sqrt(q.v)));
    for (int j = 0; j < 2; ++j) {
      Dual o = out[j] * rs;
      e[j] = o.v;
      for (int t = 0; t < T_; ++t) R[j * T_ + t] = o.d[t];
    }
  }

  template <class In>
  void run(const In* q, double* e, double* R) {
    for (int c = 0; c < nc; ++c)
      for (int i = 0; i < S; ++i)
        for (int j = 0; j < S; ++j) img[(c * S + i) * 64 + j] = double(q[(size_t)c * 4096 + (i + 1) * 64 + j + 1]);
#ifdef D4_PROFILE
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    auto tick = [&](int k) { auto t1 = clk::now(); g_prof[k] += std::chrono::duration<double>(t1 - t0).count(); t0 = t1; };
    center(); tick(0); normalize(); tick(1); layer1(); tick(2); layer2(); tick(3); pooled_f8(); tick(4);
    attention(); tick(5); pool_kernel(); tick(6); head(e, R); tick(7);
#else
    center();
    normalize();
    layer1();
    layer2();
    pooled_f8();
    attention();
    pool_kernel();
    head(e, R);
#endif
  }
};

// ------------------------------------------------------------------- binding
template <class T> std::vector<T> getv(const py::dict& d, const char* k, size_t n) {
  auto a = py::array_t<double, py::array::c_style | py::array::forcecast>(d[k]);
  if ((size_t)a.size() != n) throw std::runtime_error(std::string("bad size for ") + k);
  std::vector<T> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = T(a.data()[i]);
  return v;
}
inline double gets(const py::dict& d, const char* k) { return py::float_(d[k]); }

template <class T> Weights<T> make_weights(const py::dict& d) {
  Weights<T> w;
  w.sigma = gets(d, "center_sigma"); w.max_shift = gets(d, "center_max_shift");
  w.iters = (int)gets(d, "center_iterations"); w.npad = (int)gets(d, "center_pad");
  w.gain = gets(d, "gain"); w.alpha = gets(d, "alpha");
  w.nl_beta = gets(d, "nl_beta"); w.nl_threshold = gets(d, "nl_threshold");
  w.gate_mode = (int)gets(d, "gate_mode");
  w.eps1 = T(gets(d, "eps1")); w.eps2 = T(gets(d, "eps2")); w.eps_a = T(gets(d, "eps_a")); w.eps_f = T(gets(d, "eps_f"));
  w.res2 = T(gets(d, "res2")); w.bk = T(gets(d, "bk"));
  w.wb = getv<T>(d, "wb", NPIX);
  w.l1 = getv<T>(d, "l1", 7 * CH);
  w.W2 = getv<T>(d, "W2", 64 * CH); w.b2 = getv<T>(d, "b2", CH); w.g2 = getv<T>(d, "g2", CH); w.beta2 = getv<T>(d, "beta2", CH);
  w.W3 = getv<T>(d, "W3", 96 * CH); w.b3 = getv<T>(d, "b3", CH);
  w.Wtok = getv<T>(d, "Wtok", CH * AD); w.tokbias = getv<T>(d, "tokbias", NTOK * AD);
  w.an_g = getv<T>(d, "an_g", AD); w.an_b = getv<T>(d, "an_b", AD);
  w.Wqkv = getv<T>(d, "Wqkv", AD * 3 * AD); w.bqkv = getv<T>(d, "bqkv", 3 * AD);
  w.Wo = getv<T>(d, "Wo", AD * AD); w.bo = getv<T>(d, "bo", AD);
  w.fn_g = getv<T>(d, "fn_g", AD); w.fn_b = getv<T>(d, "fn_b", AD);
  w.Wf1 = getv<T>(d, "Wf1", AD * FF); w.bf1 = getv<T>(d, "bf1", FF);
  w.Wf2 = getv<T>(d, "Wf2", FF * AD); w.bf2 = getv<T>(d, "bf2", AD);
  w.wk = getv<T>(d, "wk", AD);
  w.M = getv<T>(d, "M", S * NB); w.MT.resize(NB * S);
  for (int i = 0; i < S; ++i) for (int a = 0; a < NB; ++a) w.MT[a * S + i] = w.M[i * NB + a];
  w.sup = getv<T>(d, "sup", NPIX); w.B = getv<T>(d, "B", 50);
  w.common = getv<double>(d, "common", CH); w.fscale = getv<double>(d, "fscale", CH);
  w.gW1 = getv<double>(d, "gW1", GH * CH); w.gb1 = getv<double>(d, "gb1", GH);
  w.gW2 = getv<double>(d, "gW2", CH * GH); w.gb2 = getv<double>(d, "gb2", CH);
  for (int i = 0; i < NB; ++i) { w.pb0[i] = (i * S) / NB; w.pb1[i] = ((i + 1) * S + NB - 1) / NB; }
  return w;
}

struct ModelBase {
  virtual ~ModelBase() = default;
  virtual py::tuple measure(py::array q, int T, int nthreads) = 0;
};

template <class T> struct Model : ModelBase {
  Weights<T> w;
  explicit Model(const py::dict& d) : w(make_weights<T>(d)) {}

  template <class In>
  void run_all(const In* q, size_t N, int C, int T_, int nthreads, double* e, double* R) {
    auto work = [&](size_t a, size_t b) {
      Worker<T> wk(w, T_);
      for (size_t n = a; n < b; ++n) wk.run(q + n * C * 4096, e + 2 * n, R + 2 * T_ * n);
    };
    nthreads = std::max(1, std::min<int>(nthreads, (int)N));
    if (nthreads == 1) { work(0, N); return; }
    std::vector<std::thread> th;
    for (int k = 0; k < nthreads; ++k) th.emplace_back(work, N * k / nthreads, N * (k + 1) / nthreads);
    for (auto& t : th) t.join();
  }

  py::tuple measure(py::array q, int T_, int nthreads) override {
    if (q.ndim() != 4 || q.shape(2) != 64 || q.shape(3) != 64) throw std::invalid_argument("q must be [N,C,64,64]");
    size_t N = q.shape(0); int C = (int)q.shape(1);
    if (T_ < 0 || T_ > MAXT || T_ + 1 > C) throw std::invalid_argument("need 0 <= T <= 4 and T+1 <= C");
    py::array_t<double> e({(py::ssize_t)N, (py::ssize_t)2});
    py::array_t<double> R({(py::ssize_t)N, (py::ssize_t)2, (py::ssize_t)T_});
    double* ep = e.mutable_data(); double* rp = R.mutable_data();
    if (py::isinstance<py::array_t<float>>(q) && (q.flags() & py::array::c_style)) {
      auto a = py::array_t<float, py::array::c_style>::ensure(q);
      const float* p = a.data();
      py::gil_scoped_release nogil;
      run_all(p, N, C, T_, nthreads, ep, rp);
    } else {
      auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(q);
      const double* p = a.data();
      py::gil_scoped_release nogil;
      run_all(p, N, C, T_, nthreads, ep, rp);
    }
    return py::make_tuple(e, R);
  }
};

}  // namespace

PYBIND11_MODULE(_d4inv, m) {
  m.doc() = "D4InvCNN CPU forward-mode inference";
  py::class_<ModelBase>(m, "Model")
      .def("measure", &ModelBase::measure, py::arg("q"), py::arg("n_tangent") = 2, py::arg("nthreads") = 1);
  m.def("create", [](const py::dict& d, const std::string& precision) -> ModelBase* {
    if (precision == "float32") return new Model<float>(d);
    if (precision == "float64") return new Model<double>(d);
    throw std::invalid_argument("precision must be float32 or float64");
  }, py::return_value_policy::take_ownership);
  m.attr("VECTOR_BYTES") = VB;
#ifdef D4_PROFILE
  m.def("profile", []() { std::vector<double> v(g_prof, g_prof + 8); std::fill(g_prof, g_prof + 8, 0.0); py::list l; for (double x : v) l.append(x); return l; });
#endif
}
