#pragma once
// Optional phase timers (compile with -DD4CNN_PROFILE). Disabled builds
// compile every D4_PROF to nothing. Nested scopes are not additive.
#include <array>
#include <atomic>
#include <chrono>
namespace d4cnn::prof {
enum Phase:int {total,normalize,first_layer,hidden,wino_input,wino_gemm,wino_output,wino_epilogue,
 orbit_gather,attention,attention_core,heads,count};
inline const char* names[count]={"total","normalize","first_layer","hidden","wino_input","wino_gemm",
 "wino_output","wino_epilogue","orbit_gather","attention","attention_core","heads"};
inline std::array<std::atomic<unsigned long long>,count> sums{};
inline thread_local std::array<unsigned long long,count> local{};
inline auto now(){return std::chrono::steady_clock::now();}
inline void add(int k,decltype(now()) start){local[k]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now()-start).count();}
struct Scope{int k;decltype(now()) start;explicit Scope(int k_):k(k_),start(now()){}~Scope(){add(k,start);}};
// Fold thread-local counters into the global sums (call at the end of a task).
inline void commit(){for(int k=0;k<count;++k){sums[k].fetch_add(local[k]);local[k]=0;}}
inline void reset(){for(auto& s:sums)s=0;}
}
#ifdef D4CNN_PROFILE
#define D4_PROF_CAT2(a,b) a##b
#define D4_PROF_CAT(a,b) D4_PROF_CAT2(a,b)
#define D4_PROF(k) d4cnn::prof::Scope D4_PROF_CAT(d4_prof_,__LINE__)(d4cnn::prof::k)
#define D4_PROF_MARK(var) auto var=d4cnn::prof::now()
#define D4_PROF_ADD(k,var) do{d4cnn::prof::add(d4cnn::prof::k,var);var=d4cnn::prof::now();}while(0)
#else
#define D4_PROF(k) do{}while(0)
#define D4_PROF_MARK(var) do{}while(0)
#define D4_PROF_ADD(k,var) do{}while(0)
#endif
