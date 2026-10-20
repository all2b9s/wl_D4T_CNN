#pragma once
#include <vector>
namespace d4cnn {
// F(4x4,3x3). The 1D matrices satisfy
// sum_t A[o,t]*G[t,k]*B[t,i] == (i == o+k); see test_winograd_identity.py.
inline std::vector<float> winograd4_weights(const std::vector<float>& weights){
 constexpr double g[6][3]={{.25,0,0},{-1./6,-1./6,-1./6},
  {-1./6,1./6,-1./6},{1./24,1./12,1./6},{1./24,-1./12,1./6},{0,0,1}};
 std::vector<float> out(36*32*32);
 for(int i=0;i<32;++i)for(int o=0;o<32;++o)for(int a=0;a<6;++a)for(int b=0;b<6;++b){
  double v=0;for(int y=0;y<3;++y)for(int x=0;x<3;++x)
   v+=g[a][y]*double(weights[((y*3+x)*32+i)*32+o])*g[b][x];
  out[((a*6+b)*32+i)*32+o]=float(v);
 }return out;
}
}
