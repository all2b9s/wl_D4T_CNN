// CPU forward-mode inference for cropped F8_noTanh checkpoints
// (src/architecture/forward8_CNN.py: F8_RECIPES 'f8c_parab10_d{D}c16_truth', e.g.
// models/legacy/f8_3c16_s43.pth), wrapped in ObservableCenter(disabled) +
// InputCrop63 + TrainableRadial.
//
// One galaxy = one 64x64 q stamp [v, t_1..t_T]; output e[2] and R[2][T] = de/dt.
// The primal and all T tangents are propagated together (dual numbers, T <= 4).
//
// Network (per galaxy): central 39x39 crop -> brightness normalization ->
// 8 D4 views -> D x [valid 3x3 conv 16ch -> WindowedChannelNorm (one scale per
// object, Gaussian window, shared over the 8 views) -> exact GELU -> + 0.25 residual]
// -> per-view pooling with the fixed parabola kernel -> scalar-conditioned MLP readout.
//
// Exact reductions / C=16-specific layout (no approximation beyond float rounding):
//  * Channel centring of the norm is linear, so it is folded into the conv
//    weights and bias at export (W - mean_out W); the conv emits centred values.
//  * Features are channels-last [view][t][H][W][16]: 16 channels = 4 NEON float
//    vectors. The conv micro-kernel keeps (1+T) x NPX pixels x 16 channels in
//    registers, so the primal and every tangent share each weight load; the
//    whole 16x16x3x3 weight block (9 KB) stays in L1.
//  * The norm variance and its tangents are accumulated in the conv epilogue;
//    norm + GELU (+ derivative, sharing one exp) + residual run in one second pass,
//    in place.
//  * The last layer's norm/GELU/residual are evaluated only on the support of
//    the pooling kernel (K > 1e-14 max K; dropped weight < 1e-13 of the total)
//    and pooled directly, never stored. Its conv still runs on the full grid,
//    because the norm window sees every pixel.
//  * Pooling in each view's own frame is exact because K is D4 symmetric.
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

namespace {

#if defined(__AVX2__)
constexpr int VB = 32;
constexpr int ACC_REGS = 12;  // 16 ymm registers
#else
constexpr int VB = 16;
constexpr int ACC_REGS = 24;  // 32 NEON registers
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

constexpr int C = 16, MAXT = 4, NV_MAX = 8, RH_MAX = 256;

// ---------------------------------------------------------------- vector math
// exp for x <= 0 with 2^n added to the exponent bits (Cephes polynomial).
__attribute__((always_inline)) inline vt<float> vexp_neg(vt<float> x) {
  typedef int vi __attribute__((vector_size(VB)));
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
  return (vt<float>)((vi)p + (ni << 23));
}
inline vt<double> vexp_neg(vt<double> x) { for (int i = 0; i < LANES<double>; ++i) x[i] = std::exp(x[i]); return x; }

__attribute__((always_inline)) inline vt<float> verf(vt<float> x) {  // rational minimax (Eigen/XLA float erf)
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

// Exact-erf GELU and, if need_d, its derivative. Float32 with the derivative:
// Phi shares the exp(-z^2/2) of phi via Zelen-Severo (|abs err| < 7.5e-8).
template <class V> __attribute__((always_inline)) inline void gelu(V z, V& g, V& dg, bool need_d) {
  using T = std::remove_cv_t<std::remove_reference_t<decltype(z[0])>>;
  if (!need_d) {
    g = z * ((verf(z * T(0.70710678118654752440)) + T(1)) * T(0.5));
    return;
  }
  V phi = vexp_neg(z * z * T(-0.5)) * T(0.39894228040143267794);
  V cdf;
  if constexpr (sizeof(T) == 4) {
    V a = z < T(0) ? -z : z;
    V t = T(1) / (a * T(0.2316419) + T(1));
    V p = t * T(1.330274429) + T(-1.821255978);
    p = p * t + T(1.781477937);
    p = p * t + T(-0.356563782);
    p = p * t + T(0.319381530);
    V tail = phi * (p * t);
    cdf = z < T(0) ? tail : T(1) - tail;
  } else {
    cdf = (verf(z * T(0.70710678118654752440)) + T(1)) * T(0.5);
  }
  g = z * cdf;
  dg = cdf + z * phi;
}

inline double gelu_s(double z, double& d) {
  double cdf = 0.5 * std::erfc(-z * 0.70710678118654752440);
  d = cdf + z * 0.39894228040143267794 * std::exp(-0.5 * z * z);
  return z * cdf;
}

// ------------------------------------------------------------------- weights
struct Layer {
  int cin, hi, ho;
  std::vector<double> W, b, g, be, skip, wn;  // W [3][3][cin][16] centred over output channels
  double eps;
};

template <class T> struct LayerT {
  int cin, hi, ho;
  std::vector<T> W, U, b, g, be, skip, wn;  // U: Winograd F(2,3) weights [16][cin][16]
  double eps;
};

template <class T> struct Weights {
  int depth, n, nf;
  double res, gain, beta, thr;
  std::vector<T> wb;              // [n*n] brightness weight, x100 folded
  std::vector<int> perm;          // [8][n*n] D4 orbit gather indices
  std::vector<LayerT<T>> L;
  std::vector<int> kidx;          // pool-kernel support on the last grid
  std::vector<T> kval;
  double s1[8], s2[8];
  int rh;
  bool winograd = true;
  std::vector<double> R1W, R1b, R2W, R2b, sinv;  // readout in float64 (tiny)
};

template <class T>
std::vector<T> getv(const py::dict& d, const char* key, size_t size) {
  if (!d.contains(key)) throw std::invalid_argument(std::string("missing table ") + key);
  auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(d[key]);
  if (!a || (size_t)a.size() != size) throw std::invalid_argument(std::string("bad size for ") + key);
  return std::vector<T>(a.data(), a.data() + size);
}
inline double gets(const py::dict& d, const char* key) { return getv<double>(d, key, 1)[0]; }

template <class T> Weights<T> make_weights(const py::dict& d) {
  Weights<T> w;
  if ((int)gets(d, "channels") != C) throw std::invalid_argument("native F8_noTanh is compiled for 16 channels");
  w.depth = (int)gets(d, "depth"); w.n = (int)gets(d, "input_size");
  if (w.depth < 1 || w.n % 2 == 0 || w.n - 2 * w.depth < 1 || w.n > 63) throw std::invalid_argument("bad depth/input_size");
  w.nf = w.n - 2 * w.depth;
  w.res = gets(d, "res"); w.gain = gets(d, "gain"); w.beta = gets(d, "nl_beta"); w.thr = gets(d, "nl_threshold");
  w.wb = getv<T>(d, "wb", (size_t)w.n * w.n);
  auto perm = getv<double>(d, "perm", (size_t)8 * w.n * w.n);
  w.perm.assign(perm.begin(), perm.end());
  for (int l = 0; l < w.depth; ++l) {
    LayerT<T> ly;
    ly.cin = l == 0 ? 1 : C; ly.hi = w.n - 2 * l; ly.ho = ly.hi - 2;
    std::string s = std::to_string(l);
    ly.W = getv<T>(d, ("W" + s).c_str(), (size_t)9 * ly.cin * C);
    if (ly.cin == C) {  // U = G w G^T in float64, G = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
      auto Wd = getv<double>(d, ("W" + s).c_str(), (size_t)9 * C * C);
      const double G[4][3] = {{1, 0, 0}, {.5, .5, .5}, {.5, -.5, .5}, {0, 0, 1}};
      ly.U.resize((size_t)16 * C * C);
      for (int a = 0; a < 4; ++a)
        for (int c = 0; c < 4; ++c)
          for (int i = 0; i < C * C; ++i) {
            double s = 0;
            for (int ky = 0; ky < 3; ++ky)
              for (int kx = 0; kx < 3; ++kx) s += G[a][ky] * G[c][kx] * Wd[(ky * 3 + kx) * C * C + i];
            ly.U[(size_t)(a * 4 + c) * C * C + i] = (T)s;
          }
    }
    ly.b = getv<T>(d, ("b" + s).c_str(), C);
    ly.g = getv<T>(d, ("g" + s).c_str(), C);
    ly.be = getv<T>(d, ("be" + s).c_str(), C);
    ly.wn = getv<T>(d, ("wn" + s).c_str(), (size_t)ly.ho * ly.ho);
    ly.eps = gets(d, ("eps" + s).c_str());
    if (l == 0) ly.skip = getv<T>(d, "skip", C);
    w.L.push_back(std::move(ly));
  }
  auto K = getv<double>(d, "K", (size_t)w.nf * w.nf);
  double kmax = *std::max_element(K.begin(), K.end());
  double cut = gets(d, "k_cut") * kmax;
  for (int i = 0; i < w.nf * w.nf; ++i)
    if (K[i] > cut) { w.kidx.push_back(i); w.kval.push_back((T)K[i]); }
  auto s1 = getv<double>(d, "signs_e1", 8), s2 = getv<double>(d, "signs_e2", 8);
  for (int v = 0; v < 8; ++v) { w.s1[v] = s1[v]; w.s2[v] = s2[v]; }
  w.rh = (int)gets(d, "readout_hidden");
  if (w.rh < 1 || w.rh > RH_MAX) throw std::invalid_argument("bad readout_hidden");
  w.R1W = getv<double>(d, "R1W", (size_t)w.rh * C); w.R1b = getv<double>(d, "R1b", w.rh);
  w.R2W = getv<double>(d, "R2W", (size_t)C * w.rh); w.R2b = getv<double>(d, "R2b", C);
  auto sc = getv<double>(d, "scalar_scale", C);
  for (double x : sc) w.sinv.push_back(1.0 / x);
  return w;
}

// ------------------------------------------------------------ conv kernels
constexpr int npx_for(int nt1, int nv) { int p = ACC_REGS / (nt1 * nv); return p < 1 ? 1 : (p > 6 ? 6 : p); }

// Pass 1 for NPX output pixels (x..x+NPX-1, row y) of one view, all 1+NT images.
// in/out: view bases; istr/ostr: per-image strides. Stores centred conv output;
// var[0] += wn * c*c, var[t] += wn * c*dc_t (vector partial sums).
template <class T, int CIN, int NT1, int NPX>
__attribute__((always_inline)) inline void conv_tile(const T* __restrict in, size_t istr, int wi, const T* __restrict W,
                                                     const T* __restrict b, T* __restrict out, size_t ostr, int wo,
                                                     int y, int x, const T* __restrict wn, vt<T>* var) {
  constexpr int L = LANES<T>, NV = C / L;
  vt<T> acc[NT1][NPX][NV];
  for (int p = 0; p < NPX; ++p)
    for (int j = 0; j < NV; ++j) {
      acc[0][p][j] = ld(b + j * L);
      for (int t = 1; t < NT1; ++t) acc[t][p][j] = vt<T>{};
    }
  for (int ky = 0; ky < 3; ++ky)
    for (int kx = 0; kx < 3; ++kx) {
      const T* w = W + (ky * 3 + kx) * CIN * C;
      const T* ip = in + ((size_t)(y + ky) * wi + x + kx) * CIN;
      for (int ic = 0; ic < CIN; ++ic) {
        vt<T> wv[NV];
        for (int j = 0; j < NV; ++j) wv[j] = ld(w + ic * C + j * L);
        for (int t = 0; t < NT1; ++t)
          for (int p = 0; p < NPX; ++p) {
            T a = ip[t * istr + p * CIN + ic];
            for (int j = 0; j < NV; ++j) acc[t][p][j] += a * wv[j];
          }
      }
    }
  for (int p = 0; p < NPX; ++p) {
    T wp = wn[y * wo + x + p];
    T* op = out + ((size_t)y * wo + x + p) * C;
    vt<T> s[NT1];
    for (int t = 0; t < NT1; ++t) {
      s[t] = vt<T>{};
      for (int j = 0; j < NV; ++j) { st(op + t * ostr + j * L, acc[t][p][j]); s[t] += acc[0][p][j] * acc[t][p][j]; }
      var[t] += wp * s[t];
    }
  }
}

template <class T, int CIN, int NT1>
void conv_view(const T* in, size_t istr, int wi, const T* W, const T* b, T* out, size_t ostr, int wo,
               const T* wn, double* scale) {
  constexpr int NV = C / LANES<T>, NPX = npx_for(NT1, NV);
  vt<T> var[NT1];
  for (int t = 0; t < NT1; ++t) var[t] = vt<T>{};
  for (int y = 0; y < wo; ++y) {
    int x = 0;
    for (; x + NPX <= wo; x += NPX) conv_tile<T, CIN, NT1, NPX>(in, istr, wi, W, b, out, ostr, wo, y, x, wn, var);
    for (; x < wo; ++x) conv_tile<T, CIN, NT1, 1>(in, istr, wi, W, b, out, ostr, wo, y, x, wn, var);
  }
  for (int t = 0; t < NT1; ++t) scale[t] = (double)hsum(var[t]);
}

// Winograd F(2x2,3x3) for the 16->16 layers: U = G w G^T per (ic,oc) [16 pos][ic][oc],
// V = B^T d B per input 4x4 tile, M = sum_ic V U, Y = A^T M A. NTL tiles x (1+T)
// images share each U load. The last tile row/column may read one row/column past
// the valid input; by the structure of B and A this only reaches discarded outputs
// (buffers carry zeroed slack), and those outputs are neither stored nor counted.
template <class T, int NT1, int NTL>
__attribute__((always_inline)) inline void wino_tiles(const T* __restrict in, size_t istr, int wi, const T* __restrict U,
                                                      const T* __restrict b, T* __restrict out, size_t ostr, int wo,
                                                      int y0, int x0, const T* __restrict wn, vt<T>* var) {
  constexpr int L = LANES<T>, NV = C / L;
  alignas(64) T V[NTL][NT1][16][C];
  for (int tl = 0; tl < NTL; ++tl)
    for (int t = 0; t < NT1; ++t) {
      const T* base = in + t * istr + ((size_t)y0 * wi + x0 + 2 * tl) * C;
      for (int j = 0; j < NV; ++j) {
        vt<T> d[4][4], r[4][4];
        for (int a = 0; a < 4; ++a)
          for (int c = 0; c < 4; ++c) d[a][c] = ld(base + ((size_t)a * wi + c) * C + j * L);
        for (int c = 0; c < 4; ++c) {  // B^T d
          r[0][c] = d[0][c] - d[2][c]; r[1][c] = d[1][c] + d[2][c];
          r[2][c] = d[2][c] - d[1][c]; r[3][c] = d[1][c] - d[3][c];
        }
        for (int a = 0; a < 4; ++a) {  // (B^T d) B
          st(&V[tl][t][a * 4 + 0][j * L], r[a][0] - r[a][2]); st(&V[tl][t][a * 4 + 1][j * L], r[a][1] + r[a][2]);
          st(&V[tl][t][a * 4 + 2][j * L], r[a][2] - r[a][1]); st(&V[tl][t][a * 4 + 3][j * L], r[a][1] - r[a][3]);
        }
      }
    }
  alignas(64) T M[NTL][NT1][16][C];
  for (int k = 0; k < 16; ++k) {
    vt<T> acc[NTL][NT1][NV];
    for (int tl = 0; tl < NTL; ++tl) for (int t = 0; t < NT1; ++t) for (int j = 0; j < NV; ++j) acc[tl][t][j] = vt<T>{};
    const T* u = U + (size_t)k * C * C;
    for (int ic = 0; ic < C; ++ic) {
      vt<T> uv[NV];
      for (int j = 0; j < NV; ++j) uv[j] = ld(u + ic * C + j * L);
      for (int tl = 0; tl < NTL; ++tl)
        for (int t = 0; t < NT1; ++t) {
          T a = V[tl][t][k][ic];
          for (int j = 0; j < NV; ++j) acc[tl][t][j] += a * uv[j];
        }
    }
    for (int tl = 0; tl < NTL; ++tl) for (int t = 0; t < NT1; ++t) for (int j = 0; j < NV; ++j) st(&M[tl][t][k][j * L], acc[tl][t][j]);
  }
  for (int tl = 0; tl < NTL; ++tl) {
    const int xb = x0 + 2 * tl;
    vt<T> Y[NT1][2][2][NV];
    for (int t = 0; t < NT1; ++t)
      for (int j = 0; j < NV; ++j) {
        vt<T> m[4][4], s[2][4];
        for (int k = 0; k < 16; ++k) m[k / 4][k % 4] = ld(&M[tl][t][k][j * L]);
        for (int c = 0; c < 4; ++c) { s[0][c] = m[0][c] + m[1][c] + m[2][c]; s[1][c] = m[1][c] - m[2][c] - m[3][c]; }
        vt<T> bias = t ? vt<T>{} : ld(b + j * L);
        for (int a = 0; a < 2; ++a) {
          Y[t][a][0][j] = s[a][0] + s[a][1] + s[a][2] + bias;
          Y[t][a][1][j] = s[a][1] - s[a][2] - s[a][3] + bias;
        }
      }
    for (int a = 0; a < 2; ++a)
      for (int c = 0; c < 2; ++c) {
        const int yy = y0 + a, xx = xb + c;
        if (yy >= wo || xx >= wo) continue;
        T wp = wn[yy * wo + xx];
        T* op = out + ((size_t)yy * wo + xx) * C;
        for (int t = 0; t < NT1; ++t) {
          vt<T> s = vt<T>{};
          for (int j = 0; j < NV; ++j) { st(op + t * ostr + j * L, Y[t][a][c][j]); s += Y[0][a][c][j] * Y[t][a][c][j]; }
          var[t] += wp * s;
        }
      }
  }
}

template <class T, int NT1>
void wino_view(const T* in, size_t istr, int wi, const T* U, const T* b, T* out, size_t ostr, int wo,
               const T* wn, double* scale) {
  constexpr int NV = C / LANES<T>, NTL = npx_for(NT1, NV);
  vt<T> var[NT1];
  for (int t = 0; t < NT1; ++t) var[t] = vt<T>{};
  const int nt = (wo + 1) / 2;
  for (int ty = 0; ty < nt; ++ty) {
    int tx = 0;
    for (; tx + NTL <= nt; tx += NTL) wino_tiles<T, NT1, NTL>(in, istr, wi, U, b, out, ostr, wo, 2 * ty, 2 * tx, wn, var);
    for (; tx < nt; ++tx) wino_tiles<T, NT1, 1>(in, istr, wi, U, b, out, ostr, wo, 2 * ty, 2 * tx, wn, var);
  }
  for (int t = 0; t < NT1; ++t) scale[t] = (double)hsum(var[t]);
}

// Pass 2 per pixel: y = c*gr + be, dy = dc*gr + c*gdr[t]; GELU; + res * residual.
// POOL: accumulate K-weighted outputs into pool[t][16] instead of storing.
template <class T, int CIN, int NT1, bool POOL>
void act_view(T* z, size_t zstr, int ho, const T* xin, size_t istr, int wi, const T* gr, const T* gdr /*[NT1][C]*/,
              const T* be, const T* skip, T res, const int* idx, const T* kval, int npix, double* pool) {
  constexpr int L = LANES<T>, NV = C / L;
  vt<T> acc[NT1][NV];
  if constexpr (POOL)
    for (int t = 0; t < NT1; ++t) for (int j = 0; j < NV; ++j) acc[t][j] = vt<T>{};
  const int count = POOL ? npix : ho * ho;
  for (int k = 0; k < count; ++k) {
    const int u = POOL ? idx[k] : k;
    const int yy = u / ho, xx = u - yy * ho;
    T* zp = z + (size_t)u * C;
    const T* xp = xin + ((size_t)(yy + 1) * wi + xx + 1) * CIN;
    for (int j = 0; j < NV; ++j) {
      vt<T> c0 = ld(zp + j * L), grv = ld(gr + j * L);
      vt<T> g, dg;
      gelu(c0 * grv + ld(be + j * L), g, dg, NT1 > 1);
      vt<T> o[NT1];
      for (int t = 0; t < NT1; ++t) {
        vt<T> r = CIN == 1 ? ld(skip + j * L) * xp[t * istr] : ld(xp + t * istr + j * L);
        if (t == 0) o[0] = g + res * r;
        else o[t] = dg * (ld(zp + t * zstr + j * L) * grv + c0 * ld(gdr + t * C + j * L)) + res * r;
      }
      if constexpr (POOL) {
        T kv = kval[k];
        for (int t = 0; t < NT1; ++t) acc[t][j] += kv * o[t];
      } else {
        for (int t = 0; t < NT1; ++t) st(zp + t * zstr + j * L, o[t]);
      }
    }
  }
  if constexpr (POOL)
    for (int t = 0; t < NT1; ++t)
      for (int j = 0; j < NV; ++j)
        for (int i = 0; i < L; ++i) pool[t * C + j * L + i] = (double)acc[t][j][i];
}

// ------------------------------------------------------------------- worker
template <class T, int NT1> struct Worker {
  const Weights<T>& w;
  std::vector<T> x0, A, B, xc;
  explicit Worker(const Weights<T>& w_) : w(w_) {
    size_t nn = (size_t)w.n * w.n, h1 = (size_t)(w.n - 2) * (w.n - 2);
    x0.resize(8 * NT1 * nn); xc.resize(NT1 * nn);
    // Slack: Winograd edge tiles may read one row + one pixel past the last image.
    const size_t slack = (size_t)(w.n + 2) * C;
    A.assign(8 * NT1 * h1 * C + slack, T(0)); B.assign(8 * NT1 * h1 * C + slack, T(0));
  }

  template <class In> void run(const In* q, double* e, double* R) {
    const int n = w.n, nn = n * n, off = 32 - n / 2;
    // 1. Crop the central n x n (InputCrop63 [1:,1:] then centred crop) and normalize brightness.
    double m[NT1];
    for (int t = 0; t < NT1; ++t) {
      double s = 0;
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          T v = (T)q[(size_t)t * 4096 + (i + off) * 64 + j + off];
          xc[t * nn + i * n + j] = v;
          s += (double)(w.wb[i * n + j] * v);
        }
      m[t] = s;
    }
    const double zb = w.beta * (m[0] - 1.0);
    const double sp = zb > w.thr ? m[0] - 1.0 : std::log1p(std::exp(zb)) / w.beta;
    const double sig = zb > w.thr ? 1.0 : 1.0 / (1.0 + std::exp(-zb));
    const double nrm = 1.0 + sp;
    const T inv = (T)(1.0 / nrm);
    // 2. D4 orbit [8][1+T][n*n]; dx' = dx/n - x * dn/n^2.
    for (int t = 0; t < NT1; ++t) {
      const T dnn = t ? (T)(sig * m[t] / nrm) : T(0);
      const T* src = xc.data() + t * nn;
      const T* src0 = xc.data();
      for (int v = 0; v < 8; ++v) {
        const int* pm = w.perm.data() + v * nn;
        T* dst = x0.data() + ((size_t)v * NT1 + t) * nn;
        for (int i = 0; i < nn; ++i) dst[i] = t ? (src[pm[i]] - src0[pm[i]] * dnn) * inv : src[pm[i]] * inv;
      }
    }
    // 3. Trunk.
    double pool[8][NT1 * C];
    T* in = x0.data();
    T* bufs[2] = {A.data(), B.data()};
    for (int l = 0; l < w.depth; ++l) {
      const auto& ly = w.L[l];
      const size_t istr = (size_t)ly.hi * ly.hi * ly.cin, ostr = (size_t)ly.ho * ly.ho * C;
      T* out = bufs[l & 1];
      double sc[8][NT1];
      for (int v = 0; v < 8; ++v) {
        const T* vin = in + v * NT1 * istr;
        T* vout = out + v * NT1 * ostr;
        if (l == 0) conv_view<T, 1, NT1>(vin, istr, ly.hi, ly.W.data(), ly.b.data(), vout, ostr, ly.ho, ly.wn.data(), sc[v]);
        else if (w.winograd) wino_view<T, NT1>(vin, istr, ly.hi, ly.U.data(), ly.b.data(), vout, ostr, ly.ho, ly.wn.data(), sc[v]);
        else conv_view<T, C, NT1>(vin, istr, ly.hi, ly.W.data(), ly.b.data(), vout, ostr, ly.ho, ly.wn.data(), sc[v]);
      }
      // Shared orbit scale: S = mean_v sum_u W var_c; dS = 2 mean_v sum_u W cov_c(c, dc).
      double S = 0, dS[NT1] = {};
      for (int v = 0; v < 8; ++v) { S += sc[v][0]; for (int t = 1; t < NT1; ++t) dS[t] += sc[v][t]; }
      S /= 8.0 * C;
      const double r = 1.0 / std::sqrt(S + ly.eps);
      alignas(64) T gr[C], gdr[NT1][C];
      for (int c = 0; c < C; ++c) {
        gr[c] = (T)(ly.g[c] * r);
        gdr[0][c] = 0;
        for (int t = 1; t < NT1; ++t) gdr[t][c] = (T)(ly.g[c] * (-0.5 * r * r * r * (2.0 * dS[t] / (8.0 * C))));
      }
      const bool last = l + 1 == w.depth;
      for (int v = 0; v < 8; ++v) {
        const T* vin = in + v * NT1 * istr;
        T* vout = out + v * NT1 * ostr;
        const T res = (T)w.res;
        if (last) {
          if (l == 0) act_view<T, 1, NT1, true>(vout, ostr, ly.ho, vin, istr, ly.hi, gr, &gdr[0][0], ly.be.data(), ly.skip.data(), res, w.kidx.data(), w.kval.data(), (int)w.kidx.size(), pool[v]);
          else act_view<T, C, NT1, true>(vout, ostr, ly.ho, vin, istr, ly.hi, gr, &gdr[0][0], ly.be.data(), nullptr, res, w.kidx.data(), w.kval.data(), (int)w.kidx.size(), pool[v]);
        } else {
          if (l == 0) act_view<T, 1, NT1, false>(vout, ostr, ly.ho, vin, istr, ly.hi, gr, &gdr[0][0], ly.be.data(), ly.skip.data(), res, nullptr, nullptr, 0, nullptr);
          else act_view<T, C, NT1, false>(vout, ostr, ly.ho, vin, istr, ly.hi, gr, &gdr[0][0], ly.be.data(), nullptr, res, nullptr, nullptr, 0, nullptr);
        }
      }
      in = out;
    }
    // 4. Readout (float64): scalar = mean_v P, v_k = mean_v s_k P,
    //    w = MLP(scalar / scale), e_k = gain * mean_c v_k w.
    double sc_[NT1][C], v1[NT1][C], v2[NT1][C];
    for (int t = 0; t < NT1; ++t)
      for (int c = 0; c < C; ++c) {
        double a = 0, b1 = 0, b2 = 0;
        for (int v = 0; v < 8; ++v) { double p = pool[v][t * C + c]; a += p; b1 += w.s1[v] * p; b2 += w.s2[v] * p; }
        sc_[t][c] = a / 8 * w.sinv[c]; v1[t][c] = b1 / 8; v2[t][c] = b2 / 8;
      }
    double h[NT1][RH_MAX];
    for (int k = 0; k < w.rh; ++k) {
      const double* row = w.R1W.data() + k * C;
      double hk[NT1];
      for (int t = 0; t < NT1; ++t) { double s = t ? 0.0 : w.R1b[k]; for (int c = 0; c < C; ++c) s += row[c] * sc_[t][c]; hk[t] = s; }
      double dg; h[0][k] = gelu_s(hk[0], dg);
      for (int t = 1; t < NT1; ++t) h[t][k] = dg * hk[t];
    }
    double cw[NT1][C];
    for (int c = 0; c < C; ++c) {
      const double* row = w.R2W.data() + c * w.rh;
      for (int t = 0; t < NT1; ++t) { double s = t ? 0.0 : w.R2b[c]; for (int k = 0; k < w.rh; ++k) s += row[k] * h[t][k]; cw[t][c] = s; }
    }
    const double gs = w.gain / C;
    double e1 = 0, e2 = 0;
    for (int c = 0; c < C; ++c) { e1 += v1[0][c] * cw[0][c]; e2 += v2[0][c] * cw[0][c]; }
    e[0] = gs * e1; e[1] = gs * e2;
    for (int t = 1; t < NT1; ++t) {
      double d1 = 0, d2 = 0;
      for (int c = 0; c < C; ++c) {
        d1 += v1[t][c] * cw[0][c] + v1[0][c] * cw[t][c];
        d2 += v2[t][c] * cw[0][c] + v2[0][c] * cw[t][c];
      }
      R[t - 1] = gs * d1; R[(NT1 - 1) + t - 1] = gs * d2;
    }
  }
};

struct ModelBase {
  virtual ~ModelBase() = default;
  virtual py::tuple measure(py::array q, int T, int nthreads) = 0;
  virtual int support() const = 0;
};

template <class T> struct Model : ModelBase {
  Weights<T> w;
  explicit Model(const py::dict& d, bool winograd) : w(make_weights<T>(d)) { w.winograd = winograd; }
  int support() const override { return (int)w.kidx.size(); }

  template <int NT1, class In> void run_t(const In* q, size_t N, int Cq, int nthreads, double* e, double* R) {
    auto work = [&](size_t a, size_t b) {
      Worker<T, NT1> wk(w);
      for (size_t i = a; i < b; ++i) wk.run(q + i * Cq * 4096, e + 2 * i, R + 2 * (NT1 - 1) * i);
    };
    nthreads = std::max(1, std::min<int>(nthreads, (int)N));
    if (nthreads == 1) { work(0, N); return; }
    std::vector<std::thread> th;
    for (int k = 0; k < nthreads; ++k) th.emplace_back(work, N * k / nthreads, N * (k + 1) / nthreads);
    for (auto& t : th) t.join();
  }
  template <class In> void run_all(const In* q, size_t N, int Cq, int T_, int nthreads, double* e, double* R) {
    switch (T_) {
      case 0: run_t<1>(q, N, Cq, nthreads, e, R); break;
      case 1: run_t<2>(q, N, Cq, nthreads, e, R); break;
      case 2: run_t<3>(q, N, Cq, nthreads, e, R); break;
      case 3: run_t<4>(q, N, Cq, nthreads, e, R); break;
      default: run_t<5>(q, N, Cq, nthreads, e, R); break;
    }
  }

  py::tuple measure(py::array q, int T_, int nthreads) override {
    if (q.ndim() != 4 || q.shape(2) != 64 || q.shape(3) != 64) throw std::invalid_argument("q must be [N,C,64,64]");
    size_t N = q.shape(0); int Cq = (int)q.shape(1);
    if (T_ < 0 || T_ > MAXT || T_ + 1 > Cq) throw std::invalid_argument("need 0 <= T <= 4 and T+1 <= C");
    py::array_t<double> e({(py::ssize_t)N, (py::ssize_t)2});
    py::array_t<double> R({(py::ssize_t)N, (py::ssize_t)2, (py::ssize_t)T_});
    double* ep = e.mutable_data(); double* rp = R.mutable_data();
    if (py::isinstance<py::array_t<float>>(q) && (q.flags() & py::array::c_style)) {
      auto a = py::array_t<float, py::array::c_style>::ensure(q);
      const float* p = a.data();
      py::gil_scoped_release nogil;
      run_all(p, N, Cq, T_, nthreads, ep, rp);
    } else {
      auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(q);
      const double* p = a.data();
      py::gil_scoped_release nogil;
      run_all(p, N, Cq, T_, nthreads, ep, rp);
    }
    return py::make_tuple(e, R);
  }
};

}  // namespace

PYBIND11_MODULE(_f8notanh, m) {
  m.doc() = "F8_noTanh (cropped, parabola pool, C=16) CPU forward-mode inference";
  py::class_<ModelBase>(m, "Model")
      .def("measure", &ModelBase::measure, py::arg("q"), py::arg("n_tangent") = 2, py::arg("nthreads") = 1)
      .def_property_readonly("pool_support", &ModelBase::support);
  m.def("create", [](const py::dict& d, const std::string& precision, const std::string& conv) -> ModelBase* {
    if (conv != "winograd" && conv != "direct") throw std::invalid_argument("conv must be winograd or direct");
    if (precision == "float32") return new Model<float>(d, conv == "winograd");
    if (precision == "float64") return new Model<double>(d, conv == "winograd");
    throw std::invalid_argument("precision must be float32 or float64");
  }, py::arg("tables"), py::arg("precision") = "float32", py::arg("conv") = "winograd",
     py::return_value_policy::take_ownership);
  m.attr("VECTOR_BYTES") = VB;
}
