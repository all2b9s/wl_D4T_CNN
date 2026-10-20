#pragma once
#include <string>
namespace d4cnn {
// Scalar ABI at dispatch boundaries: callers need no global -mavx flags.
using Epilogue32 = void (*)(float*,int,const float*,const float*,const float*,int,const float*,float,int);
using Winograd32 = void (*)(const float*,const float*,const float*,float*,int,int,const float*,const float*,float,bool);
struct X86Kernels { Epilogue32 epilogue=nullptr; Winograd32 f2=nullptr,f4=nullptr; };
X86Kernels select_x86_kernels(const std::string& isa);
}
