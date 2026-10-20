#pragma once
#include <string>
#include <vector>
namespace d4cnn {
using Conv = void (*)(const float*, const float*, const float*, float*, int, int, int, int);
void conv_scalar(const float*, const float*, const float*, float*, int, int, int, int);
Conv select_conv(const std::string& isa);
std::vector<std::string> available_isas();
std::string resolve_isa(const std::string& isa);
}
