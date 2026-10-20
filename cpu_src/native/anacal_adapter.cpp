#include "anacal/image.h"
#include "d4cnn/parallel.h"
#include <cmath>
namespace py=pybind11;
namespace {
py::array_t<double> prepare(
 py::array_t<float,py::array::c_style|py::array::forcecast> image,
 py::array_t<double,py::array::c_style|py::array::forcecast> psf,
 py::array_t<double,py::array::c_style|py::array::forcecast> positions,
 py::object noise,int size,double scale,double sigma,double klim,int threads) {
 if(image.ndim()!=2||psf.ndim()!=2||positions.ndim()!=2||positions.shape(1)!=2)
  throw std::invalid_argument("Expected image/psf [H,W] and positions [N,2] in x,y order");
 if(size<2||size%2||threads<1||!std::isfinite(scale)||scale<=0||!std::isfinite(sigma)||sigma<=0||!std::isfinite(klim)||klim<=0)
  throw std::invalid_argument("Invalid q-stamp parameters; size must be positive and even");
 std::optional<py::array_t<float>> noises;
 if(!noise.is_none()) {auto a=py::array_t<float,py::array::c_style|py::array::forcecast>::ensure(noise);
  if(!a||a.ndim()!=2||a.shape(0)!=image.shape(0)||a.shape(1)!=image.shape(1)){ throw std::invalid_argument("Noise shape differs from image"); }
  noises=a;
 }
 // Convert pybind array wrapper types while the GIL is held. Passing a
 // c_style specialization to AnaCal's default array_t would otherwise create
 // temporary Python wrappers inside the released-GIL worker.
 py::array_t<float> image_view=image;
 py::array_t<double> psf_view=psf;
 int n=int(positions.shape(0));std::vector<std::array<int,2>> centers(n);
 for(int i=0;i<n;++i){double x=positions.at(i,0),y=positions.at(i,1);
  if(!std::isfinite(x)||!std::isfinite(y)||x<0||y<0||x>=image.shape(1)||y>=image.shape(0))throw std::invalid_argument("Invalid source position");
  int cx=int(std::floor(x+.5)),cy=int(std::floor(y+.5));
  if(cx-size/2<0||cy-size/2<0||cx+size/2>image.shape(1)||cy+size/2>image.shape(0))throw std::invalid_argument("Source stamp crosses image edge");
  centers[i]={cx,cy};}
 py::array_t<double> result(std::vector<py::ssize_t>{n,5,size,size});double* out=result.mutable_data();
 {py::gil_scoped_release release;
  d4cnn::parallel_for(n,threads,[&](int i){
   anacal::ImageQLease lease(size,size,scale,sigma,klim,true);
   auto q=lease.get().prepare_qnumber_vector_band(image_view,psf_view,0,centers[i][0],centers[i][1],noises);
   double* p=out+size_t(i)*5*size*size;int pixels=size*size;
   for(int k=0;k<pixels;++k){p[k]=q[k].v;p[pixels+k]=q[k].g1;p[2*pixels+k]=q[k].g2;p[3*pixels+k]=q[k].x1;p[4*pixels+k]=q[k].x2;}
  });
 }return result;
}
}
void bind_anacal(py::module_& m){m.def("prepare_qstamps",&prepare,py::arg("image"),py::arg("psf"),py::arg("positions_xy"),py::arg("noise"),py::arg("size"),py::arg("scale"),py::arg("sigma"),py::arg("klim"),py::arg("nthreads"));}
