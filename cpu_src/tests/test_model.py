"""Run: python -m unittest discover -s cpu_src/tests -v (torch needed for reference tests)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import numpy as np

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.torch=torch
        root=Path(__file__).resolve().parents[1]/"runs"/"tests"
        root.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=root)
        cls.root=Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def fixture(self, attention=True, size=8, shared=True, channels=5, suppress=True, depth=2):
        from src.architecture.forward8_CNN import Forward8_AttnW_CNN, Forward8_fixW_CNN
        from cpu_src.export import export_checkpoint
        torch=self.torch
        torch.manual_seed(71)
        name="Forward8_AttnW_CNN" if attention else "Forward8_fixW_CNN"
        kwargs=dict(in_dim=1,out_dim=2,base_channels=channels,head_hidden=7,num_layers=2,
                    res_factor=.1,nl_sigma=4,shared_head=shared,suppress=suppress)
        if attention:
            kwargs.update(r_cut=6.,suppress_order=8,attn_grid_size=3,attn_dim=32 if channels==32 else 8,
                          attn_heads=4 if channels==32 else 2,head_num_hidden_layers=depth)
        else:
            kwargs["gw_sigma"]=4.
        model=(Forward8_AttnW_CNN if attention else Forward8_fixW_CNN)(**kwargs)
        if attention:
            # A zero kernel head would hide missing attention derivatives.
            with torch.no_grad():
                model.kernel_head.weight.normal_(0,.15)
                model.kernel_head.bias.fill_(.1)
        tag=f"{name}_{size}_{shared}_{channels}_{suppress}_{depth}"
        cached=self.root/(tag+".npz")
        if cached.is_file():
            return cached
        checkpoint=self.root/(tag+".pth")
        torch.save(model.state_dict(),checkpoint)
        config=self.root/(tag+".json")
        config.write_text(json.dumps({"model_class":name,"model":kwargs,
                         "image_preprocessing":{"pixel_scale":.2,"sigma_arcsec":.85/2.355,"flim":10.}}))
        exported=self.root/(tag+".npz")
        export_checkpoint(checkpoint,exported,config=config,stamp_size=size)
        return exported

    def test_full_models_against_independent_autograd(self):
        from cpu_src import Model
        from cpu_src.reference import synthetic_q,measure
        from cpu_src import _native
        for attention,size,shared,suppress in ((False,8,False,True),(False,8,True,False),
                                               (True,8,True,True),(True,9,False,False)):
            path=self.fixture(attention,size,shared,suppress=suppress)
            q=synthetic_q(2,size)
            for precision in ("float32","float64"):
                reference=measure(path,q,precision,4)
                for isa in (_native.available_isas() if precision=="float32" else ["scalar"]):
                    with self.subTest(attention=attention,size=size,precision=precision,isa=isa):
                        model=Model.load(path,precision,isa)
                        result=model.measure_qstamps(q,4,2)
                        tol=1e-12 if precision=="float64" else 1e-6
                        np.testing.assert_allclose(result.e,reference.e,atol=tol,rtol=tol if precision=="float64" else 1e-5)
                        np.testing.assert_allclose(result.R,reference.R,atol=tol,rtol=tol if precision=="float64" else 1e-4)
                        np.testing.assert_allclose(result.dE_dx,reference.dE_dx,atol=tol,rtol=tol if precision=="float64" else 1e-4)
                        two=model.measure_qstamps(q,2)
                        np.testing.assert_array_equal(two.e,result.e)
                        np.testing.assert_array_equal(two.R,result.R)

    def test_directional_derivatives_and_reference_shift(self):
        from cpu_src import Model
        from cpu_src.reference import synthetic_q
        model=Model.load(self.fixture(),"float64")
        q=synthetic_q(1,8)
        result=model.measure_qstamps(q,4)
        derivatives=np.concatenate((result.R,result.dE_dx),axis=2)
        for d in range(4):
            plus,minus=q.copy(),q.copy()
            eps=1e-5
            plus[:,0]+=eps*q[:,d+1]
            minus[:,0]-=eps*q[:,d+1]
            diff=(model.measure_qstamps(plus).e-model.measure_qstamps(minus).e)/(2*eps)
            np.testing.assert_allclose(diff,derivatives[:,:,d],atol=1e-8,rtol=1e-5)
        delta=np.array([.12,-.23])
        restored=result.shift_reference(delta).shift_reference(-delta)
        np.testing.assert_allclose(restored.R,result.R,atol=1e-16)
        # AnaCal's exact rule on both shape components.
        expected=result.R.copy()
        expected[:,:,0]+=delta[0]*result.dE_dx[:,:,0]-delta[1]*result.dE_dx[:,:,1]
        expected[:,:,1]+=delta[0]*result.dE_dx[:,:,1]+delta[1]*result.dE_dx[:,:,0]
        np.testing.assert_array_equal(result.shift_reference(delta).R,expected)

    def test_c32_optimized_models_and_workspace_reuse(self):
        from cpu_src import Model, _native
        from cpu_src.reference import synthetic_q, measure
        for attention, size in ((False,8),(True,8),(True,9)):
            path=self.fixture(attention,size,channels=32)
            q=synthetic_q(3,size)
            reference=measure(path,q,"float64",4)
            variants=[(isa,algorithm) for isa in _native.available_isas()
                      for algorithm in (("direct",) if isa=="scalar" else
                                        ("auto","direct","winograd","winograd4"))]
            for isa,algorithm in variants:
                with self.subTest(attention=attention,size=size,isa=isa,algorithm=algorithm), patch.dict(os.environ,{"D4CNN_CONV":algorithm}):
                    model=Model.load(path,isa=isa)
                    self.assertEqual(model._model.convolution_algorithm,{"auto":"winograd_f4","winograd":"winograd_f2","winograd4":"winograd_f4","direct":"direct"}[algorithm])
                    result=model.measure_qstamps(q,4,2)
                    for actual,expected in ((result.e,reference.e),(result.R,reference.R),(result.dE_dx,reference.dE_dx)):
                        np.testing.assert_allclose(actual,expected,atol=2e-6,rtol=2e-4)
                    two=model.measure_qstamps(q,2,2)
                    np.testing.assert_array_equal(two.e,result.e)
                    np.testing.assert_array_equal(two.R,result.R)
                    # Reuse workers with a different model/shape, then check that
                    # accumulators and workspace contain no previous-source data.
                    other=Model.load(self.fixture(attention,10,channels=32),isa=isa)
                    other.measure_qstamps(synthetic_q(2,10),4,2)
                    again=model.measure_qstamps(q,4,2)
                    np.testing.assert_array_equal(again.R,result.R)
                    serial=model.measure_qstamps(q,4,1)
                    np.testing.assert_array_equal(serial.R,result.R)
                    trace=model.trace(q[0],4)
                    self.assertIn("kernel",trace)
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        concurrent=list(pool.map(lambda _:model.measure_qstamps(q,4,2),range(2)))
                    for item in concurrent:np.testing.assert_array_equal(item.R,result.R)

    def test_native_version_guard(self):
        from cpu_src import Model, _native
        with patch.object(_native,"api_version",0):
            with self.assertRaisesRegex(RuntimeError,"out of date"):
                Model.load(self.fixture())

    def test_gaussian_maps(self):
        from cpu_src import _native
        from src.architecture.CNN_toolkit import gaussian_weight_2d
        torch=self.torch
        for size in (64,62,48,63):
            for sigma in (4,8,16):
                for suppress in (False,True):
                    for fp64 in (False,True):
                        with self.subTest(size=size,sigma=sigma,suppress=suppress,fp64=fp64):
                            dtype=torch.float64 if fp64 else torch.float32
                            expected=gaussian_weight_2d(size,sigma,dtype=dtype,suppress=suppress).numpy()
                            actual=_native.gaussian_map(size,float(sigma),suppress,24.,8,fp64)
                            np.testing.assert_allclose(actual,expected,atol=1e-17 if fp64 else 1e-9,rtol=1e-13 if fp64 else 2e-6)

    def test_native_qstamps_and_sources(self):
        import anacal
        from cpu_src import Model,_native
        from cpu_src.reference import measure
        if not _native.has_anacal:
            self.skipTest("Build with ANACAL_SOURCE_DIR for native source tests")
        model=Model.load(self.fixture(size=8),"float32")
        rng=np.random.default_rng(1)
        image=rng.normal(0,.02,(24,24)).astype(np.float32)
        noise=rng.normal(0,.01,image.shape).astype(np.float32)
        psf=np.zeros((8,8),dtype=np.float64)
        psf[4,4]=1
        positions=np.array([[12.1,11.8],[10.2,13.4]])
        centers=np.floor(positions+.5).astype(int)
        for n in (None,noise):
            q=model.prepare_qstamps(image,psf,positions,n,nthreads=2)
            expected=[]
            for cx,cy in centers:
                stamp=image[cy-4:cy+4,cx-4:cx+4].copy()
                iq=anacal.image.ImageQ(8,8,.2,.85/2.355,50.,True)
                if n is None:
                    expected.append(iq.prepare_qnumber_image(stamp,psf,4,4))
                else:
                    expected.append(iq.prepare_qnumber_image(stamp,psf,4,4,n[cy-4:cy+4,cx-4:cx+4].copy()))
            np.testing.assert_allclose(q,expected,atol=1e-13,rtol=1e-13)
            direct=model.measure_qstamps(q,4)
            sources=model.measure_sources(image,psf,positions,n,4,2,1)
            np.testing.assert_array_equal(sources.e,direct.e)
            np.testing.assert_array_equal(sources.R,direct.R)
            np.testing.assert_array_equal(sources.dE_dx,direct.dE_dx)
            np.testing.assert_array_equal(sources.centers_xy,centers)

    def test_runtime_is_torch_free_and_api_failures(self):
        from cpu_src import Model
        from cpu_src.reference import synthetic_q
        path=self.fixture()
        model=Model.load(path)
        empty=model.measure_qstamps(np.empty((0,5,8,8)),4)
        self.assertEqual(empty.R.shape,(0,2,2))
        self.assertEqual(empty.e.dtype,np.float32)
        for q,t,threads in ((np.zeros((1,3,8,8)),2,1),(np.full((1,5,8,8),np.nan),2,1),
                            (np.zeros((1,5,8,8)),3,1),(np.zeros((1,5,8,8)),2,0)):
            with self.assertRaises(ValueError):
                model.measure_qstamps(q,t,threads)
        q=synthetic_q(2,8)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda _:model.measure_qstamps(q,4,2),range(2)))
        np.testing.assert_array_equal(results[0].R,results[1].R)
        # Block even attempted torch imports in a fresh process.
        code='''import sys, importlib.abc, numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","torchvision"): raise RuntimeError("torch import attempted")
sys.meta_path.insert(0,Block())
from cpu_src import Model
import cpu_src.pipeline.runner
import cpu_src.pipeline.calibration
m=Model.load(sys.argv[1]);r=m.measure_qstamps(np.ones((1,5,8,8),dtype=np.float32)*.001,4)
assert np.isfinite(r.R).all()
assert "torch" not in sys.modules
'''
        subprocess.run([sys.executable,"-B","-c",code,str(path)],check=True,cwd=Path(__file__).resolve().parents[2])

    def test_export_and_load_reject_incompatible_metadata(self):
        from cpu_src.model_io import load_model,validate
        from cpu_src.export import export_checkpoint
        path=self.fixture()
        meta,weights=load_model(path)
        broken=dict(weights)
        broken["qkv.weight"]=broken["qkv.weight"][:1]
        with self.assertRaisesRegex(ValueError,"qkv.weight"):
            validate(meta,broken)
        with self.assertRaisesRegex(ValueError,"nl_sigma"):
            export_checkpoint(self.root/"does_not_need_to_exist.pth",self.root/"x.npz",architecture="Forward8_AttnW_CNN")

if __name__=="__main__":
    unittest.main()
