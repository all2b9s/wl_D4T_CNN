#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "d4cnn/model.h"
#include "d4cnn/parallel.h"
#include <memory>
namespace py=pybind11;
using namespace d4cnn;
Config config_from(py::dict m) {
 Config c;
 c.size=py::cast<int>(m["stamp_size"]);auto k=py::cast<py::dict>(m["model"]);
 c.channels=py::cast<int>(k["base_channels"]);c.layers=py::cast<int>(k["num_layers"]);
 c.head_depth=k.contains("head_num_hidden_layers")?py::cast<int>(k["head_num_hidden_layers"]):3;
 c.residual=py::cast<double>(k["res_factor"]);c.nl_sigma=py::cast<double>(k["nl_sigma"]);
 c.shared=py::cast<bool>(k["shared_head"]);c.suppress=py::cast<bool>(k["suppress"]);
 c.attention=py::cast<std::string>(m["architecture"])=="Forward8_AttnW_CNN";
 if(c.attention){c.grid=py::cast<int>(k["attn_grid_size"]);c.dim=py::cast<int>(k["attn_dim"]);c.heads=py::cast<int>(k["attn_heads"]);
 c.r_cut=py::cast<double>(k["r_cut"]);c.order=py::cast<int>(k["suppress_order"]);}
 else c.gw_sigma=py::cast<double>(k["gw_sigma"]);
 if(c.size<1||c.size>4096||c.channels<1||c.layers<1||c.head_depth<1||c.grid<1||c.dim<1||c.heads<1||c.dim%c.heads)
  throw std::invalid_argument("Invalid model dimensions");
 return c;
}
Weights weights_from(py::dict weights){
 Weights out;
 for(auto item:weights){auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(item.second);
  if(!a) { throw std::invalid_argument("Parameter must be numeric"); }
  Weight v;
  v.values.assign(a.data(),a.data()+a.size());for(int j=0;j<a.ndim();++j)v.shape.push_back(int(a.shape(j)));
  out.emplace(py::cast<std::string>(item.first),std::move(v));}
 return out;
}
// Validate native entry points too: malformed shapes must never reach a kernel.
const Weights& checked_weights(const Config& c,const Weights& weights) {
 auto take=[&](const std::string& key,std::vector<int> shape){
  auto it=weights.find(key);
  if(it==weights.end()||it->second.shape!=shape)throw std::invalid_argument("Invalid shape for "+key);
  for(double v:it->second.values)if(!std::isfinite(v))throw std::invalid_argument("Non-finite parameter "+key);
 };
 take("signs_e1",{8});take("signs_e2",{8});
 for(int l=0;l<c.layers;++l){int ci=l==0?1:c.channels;std::string key="blocks."+std::to_string(l);
  take(key+".conv.weight",{c.channels,ci,3,3});take(key+".conv.bias",{c.channels});
  take(key+".norm.weight",{1,c.channels,1,1});take(key+".norm.bias",{1,c.channels,1,1});
  if(ci!=c.channels)take(key+".skip.weight",{c.channels,ci,1,1});
 }
 for(auto key:c.shared?std::vector<std::string>{"head_shared"}:std::vector<std::string>{"head_e1","head_e2"}){
  int ci=c.channels;
  for(int l=0;l<c.head_depth;++l){std::string name=key+".net."+std::to_string(1+2*l)+".weight";
   auto it=weights.find(name);if(it==weights.end()||it->second.shape.size()!=2)throw std::invalid_argument("Invalid head "+name);
   int co=it->second.shape[0];if(co<1)throw std::invalid_argument("Empty head");take(name,{co,ci});ci=co;
  }take(key+".net."+std::to_string(1+2*c.head_depth)+".weight",{1,ci});
 }
 if(c.attention){
  auto lin=[&](const std::string& key,int ci,int co){take(key+".weight",{co,ci});take(key+".bias",{co});};
  lin("token_projection",c.channels+2,c.dim);lin("qkv",c.dim,3*c.dim);lin("attn_output",c.dim,c.dim);
  lin("ffn.0",c.dim,2*c.dim);lin("ffn.2",2*c.dim,c.dim);lin("kernel_head",c.dim,1);
  for(auto name:{"attn_norm","ffn_norm"}){take(std::string(name)+".weight",{c.dim});take(std::string(name)+".bias",{c.dim});}
  take("kernel_blur",{1,1,5,5});
 }return weights;
}
template<class S> class BoundModel {
 Config cfg;Model<S,3> m2;Model<S,5> m4;
 using Array=py::array_t<S,py::array::c_style|py::array::forcecast>;
public:
 BoundModel(Config c,const Weights& w,const std::string& isa):cfg(c),m2(c,checked_weights(c,w),isa),m4(c,w,isa){}
 std::string algorithm()const{return m2.algorithm();}
 py::array_t<S> measure(Array q,int tangent,int threads){
  if(tangent!=2&&tangent!=4)throw std::invalid_argument("n_tangent must be 2 or 4");
  if(threads<1)throw std::invalid_argument("nthreads must be positive");
  if(q.ndim()!=4||q.shape(1)!=5||q.shape(2)!=cfg.size||q.shape(3)!=cfg.size)throw std::invalid_argument("q must be [N,5,H,H] with model stamp size");
  if(q.shape(0)>std::numeric_limits<int>::max())throw std::invalid_argument("Batch too large");
  int n=int(q.shape(0)),d=tangent+1;py::array_t<S> result(std::vector<py::ssize_t>{n,d,2});
  const S* in=q.data();S* out=result.mutable_data();
  {py::gil_scoped_release release;
   parallel_for(n,threads,[&](int i){
    const S* ptr=in+size_t(i)*5*cfg.size*cfg.size;
    if(tangent==2){auto v=m2.measure(ptr);std::copy(v.begin(),v.end(),out+size_t(i)*d*2);}
    else {auto v=m4.measure(ptr);std::copy(v.begin(),v.end(),out+size_t(i)*d*2);}
   });
  }return result;
 }
 py::dict trace(Array q,int tangent){
  if(q.ndim()!=3||q.shape(0)!=5||q.shape(1)!=cfg.size||q.shape(2)!=cfg.size)throw std::invalid_argument("trace expects [5,H,H]");
  if(tangent!=2&&tangent!=4)throw std::invalid_argument("n_tangent must be 2 or 4");
  py::dict out;
  auto run=[&](auto& model,auto tag){constexpr int D=decltype(tag)::value;std::map<std::string,Tensor<S,D>> traces;
   {py::gil_scoped_release release;model.measure(q.data(),&traces);}
   for(auto& [key,t]:traces){py::array_t<S> a(std::vector<py::ssize_t>{D,t.p,t.c});std::copy(t.a.begin(),t.a.end(),a.mutable_data());out[py::str(key)]=a;}};
  if(tangent==2)run(m2,std::integral_constant<int,3>{});else run(m4,std::integral_constant<int,5>{});return out;
 }
};
#ifdef D4CNN_WITH_ANACAL
void bind_anacal(py::module_&);
#endif
PYBIND11_MODULE(_native,m){
 m.doc()="D4CNN forward-mode CPU inference; no libtorch dependency";
 m.attr("api_version")=2;
 m.def("available_isas",&available_isas);m.def("resolve_isa",&resolve_isa);
 py::class_<BoundModel<float>>(m,"FloatModel")
  .def(py::init([](py::dict meta,py::dict weights,std::string isa){return std::make_unique<BoundModel<float>>(config_from(meta),weights_from(weights),isa);}))
  .def("measure",&BoundModel<float>::measure).def("trace",&BoundModel<float>::trace)
  .def_property_readonly("convolution_algorithm",&BoundModel<float>::algorithm);
 py::class_<BoundModel<double>>(m,"DoubleModel")
  .def(py::init([](py::dict meta,py::dict weights,std::string isa){return std::make_unique<BoundModel<double>>(config_from(meta),weights_from(weights),isa);}))
  .def("measure",&BoundModel<double>::measure).def("trace",&BoundModel<double>::trace)
  .def_property_readonly("convolution_algorithm",&BoundModel<double>::algorithm);
 m.def("gaussian_map",[](int size,double sigma,bool suppress,double radius,int order,bool fp64)->py::array{
  if(fp64){auto v=gaussian<double>(size,sigma,suppress,radius,order);py::array_t<double> a({size,size});std::copy(v.begin(),v.end(),a.mutable_data());return a;}
  auto v=gaussian<float>(size,sigma,suppress,radius,order);py::array_t<float> a({size,size});std::copy(v.begin(),v.end(),a.mutable_data());return a;
 });
#ifdef D4CNN_WITH_ANACAL
 bind_anacal(m);m.attr("has_anacal")=true;
#else
 m.attr("has_anacal")=false;
#endif
}
