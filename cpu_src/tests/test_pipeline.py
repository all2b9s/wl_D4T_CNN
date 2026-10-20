import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import test_model

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_model.ModelTests.setUpClass()
        cls.fixture=test_model.ModelTests()
        cls.model=cls.fixture.fixture(size=64,channels=4)
        cls.root=test_model.ModelTests.root/"pipeline"
        cls.root.mkdir()
        from astropy.io import fits
        from astropy.table import Table
        psf=np.zeros((48,48),dtype=float)
        psf[24,24]=1
        np.save(cls.root/"psf.npy",psf)
        Table({"i_ab":[23.],"hlr":[.5]}).write(cls.root/"reference.fits")
        for mode,shear in (("sim_mode0",-.02),("sim_mode40",.02)):
            folder=cls.root/mode
            folder.mkdir()
            y,x=np.mgrid[:128,:128]
            # Bright asymmetric source with a nonzero physical-looking response.
            image=(12*np.exp(-((x-64.1)**2/20+(y-64.3)**2/28))+3*np.exp(-((x-67.)**2/6+(y-63.)**2/9))).astype(np.float32)
            hdus=[fits.PrimaryHDU(),fits.ImageHDU(image),fits.ImageHDU(np.zeros_like(image)),fits.ImageHDU(np.ones_like(image)*.01)]
            fits.HDUList(hdus).writeto(folder/"exp-i-00000.fits")
            Table({"dx":[-300.],"dy":[-300.],"gamma1":[shear],"gamma2":[0.],"indices":[0]}).write(folder/"truth-00000.fits")
        cls.config={"model":str(cls.model),"blended_dir":str(cls.root),"psf_path":str(cls.root/"psf.npy"),
                    "cat_ref_path":str(cls.root/"reference.fits"),"output_root":str(cls.root/"output"),
                    "sim_modes":["sim_mode0","sim_mode40"],"exp_range":[0,1],"nthreads":2,
                    "skip_detection":True,"bs_times":3,"n_tangent":4}
        (cls.root/"config.json").write_text(json.dumps(cls.config))

    @classmethod
    def tearDownClass(cls):
        test_model.ModelTests.tearDownClass()

    def test_complete_measurement_without_torch(self):
        code='''import importlib.abc,sys,json
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in ("torch","torchvision"):raise RuntimeError("torch import attempted")
sys.meta_path.insert(0,Block())
from cpu_src.pipeline.config import PipelineConfig
from cpu_src.pipeline.runner import run_measurement
from cpu_src.pipeline.calibration import calibrate
c=PipelineConfig.load(sys.argv[1]);report=run_measurement(c)
assert report["sources"]==2
result=calibrate(c)
assert result["m"][1] is None
assert "torch" not in sys.modules
'''
        subprocess.run([sys.executable,"-B","-c",code,str(self.root/"config.json")],check=True,cwd=Path(__file__).resolve().parents[2])
        for mode in (0,1):
            folder=self.root/"output"/f"g1_{mode}"/"det_ml_cpu"
            with np.load(folder/"catalog_0.npz",allow_pickle=False) as data:
                self.assertEqual(len(data["det_id"]),1)
                self.assertIn("ml_R12",data.files)
            with np.load(folder/"catalog_0_response.npz",allow_pickle=False) as data:
                self.assertEqual(data["dE_dx"].shape,(1,2,2))
        from cpu_src.science_validation import compare_runs
        manifest=self.root/"output"/"det_ml_cpu_manifest.json"
        compared=compare_runs(manifest,manifest,repeats=3)
        self.assertEqual(compared["delta_m"],0.)
        self.assertEqual(compared["delta_c"],0.)
        self.assertTrue(compared["target_met"])
        # Real detection path, rather than only the explicit truth-mode shortcut.
        from cpu_src.pipeline.config import PipelineConfig
        from cpu_src.pipeline.runner import run_measurement
        config=PipelineConfig(**{**self.config,"skip_detection":False,"fname":"detected_cpu"})
        report=run_measurement(config)
        self.assertGreater(report["sources"],0)

    def test_calibration_rejects_missing_shear_pair(self):
        from cpu_src.pipeline.config import PipelineConfig
        from cpu_src.pipeline.calibration import calibrate
        config=PipelineConfig(**{**self.config,"sim_modes":["sim_mode80"]})
        with self.assertRaisesRegex(ValueError,"both"):
            calibrate(config)

if __name__=="__main__":
    unittest.main()
