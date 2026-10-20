#pragma once
#include <array>
#include <vector>
namespace d4cnn {
// F(2x2,3x3), cross-correlation with one-pixel zero padding. Only C32.
// Packed weight order is [transform][input channel][output channel].
inline std::vector<float> winograd_weights(const std::vector<float>& weights) {
 constexpr float g[4][3]={{1,0,0},{.5,.5,.5},{.5,-.5,.5},{0,0,1}};
 std::vector<float> out(16*32*32);
 for(int i=0;i<32;++i)for(int o=0;o<32;++o)for(int a=0;a<4;++a)for(int b=0;b<4;++b){
  float v=0;for(int y=0;y<3;++y)for(int x=0;x<3;++x)v+=g[a][y]*weights[((y*3+x)*32+i)*32+o]*g[b][x];
  out[((a*4+b)*32+i)*32+o]=v;
 }return out;
}
}
