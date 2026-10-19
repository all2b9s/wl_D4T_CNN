"""Run with: python -m unittest ml_wl.test_blended (anacal_clone environment)."""
import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ml_wl import config as C
from ml_wl.run import build_script, selected_steps, step_jobs
from ml_wl.steps import blended


class BlendedTests(unittest.TestCase):
    def test_product_response_finite_difference(self):
        rng = np.random.default_rng(9)
        e, r = rng.normal(size=(7, 2)), rng.normal(size=(7, 2, 2))
        w, dw = rng.uniform(.1, 1, 7), rng.normal(size=(7, 2))
        x, rx = blended.weighted_inputs(e, r, w, dw)
        for c in range(2):
            h = 1e-6
            plus = (w + h * dw[:, c])[:, None] * (e + h * r[:, :, c])
            minus = (w - h * dw[:, c])[:, None] * (e - h * r[:, :, c])
            np.testing.assert_allclose(rx[:, :, c], (plus-minus)/(2*h), rtol=1e-8, atol=1e-9)
        unit = blended.weighted_inputs(e, r, np.ones(7), np.zeros((7, 2)))
        np.testing.assert_array_equal(unit[0], e)
        np.testing.assert_array_equal(unit[1], r)
        np.testing.assert_allclose(x, w[:, None]*e)

    def test_split_and_legacy_configs(self):
        splits = blended.split_exposures((0, 5000))
        self.assertEqual([len(v) for v in splits.values()], [2000, 1000, 2000])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(5000))
        for k, v in splits.items():
            np.testing.assert_array_equal(v, blended.split_exposures((0, 5000))[k])
        for path in (C.ML_WL_DIR / 'configs').glob('*.yaml'):
            if path.name == 'round_template.yaml':
                continue
            for test in (False, True):
                cfg = C.load(path, test=test)
                steps = selected_steps(cfg, None)
                if cfg.get('blended') is None:
                    self.assertNotIn('evaluate', steps)
                for step in steps:
                    for i, job in enumerate(step_jobs(cfg, step)):
                        result = subprocess.run(['bash', '-n'], input=build_script(cfg, step, i, job),
                                                text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_subset_selection(self):
        with tempfile.TemporaryDirectory() as td:
            with (Path(td)/'measurement_manifest.csv').open('w') as f:
                writer=csv.writer(f);writer.writerow(['exp_id','n_det'])
                writer.writerows((i,15) for i in range(10))
            cfg={'blended':dict(index_range=[0,10],seed=42,old_catalog=td,
                               sample_targets=dict(train=30,val=15,test=30))}
            splits=blended.selected_splits(cfg)
            self.assertEqual([len(v) for v in splits.values()],[2,1,2])
            self.assertEqual(len(np.unique(np.concatenate(list(splits.values())))),5)

    def test_full_response_serialization(self):
        from src.anacal.detection_ml_pipeline import merge_and_save_catalog
        keys = ('y', 'x', 'fpfs_e1', 'fpfs_e2', 'fpfs_R11', 'fpfs_R22',
                'fpfs_w', 'fpfs_dw_dg1', 'fpfs_dw_dg2', 'fpfs_m00',
                'fpfs_dm00_dg1', 'fpfs_dm00_dg2')
        with tempfile.TemporaryDirectory() as td:
            for n in (3, 0):
                det = np.zeros(n, dtype=[(k, float) for k in keys])
                response = np.arange(n*4).reshape(n, 1, 2, 2)
                with patch('src.anacal.detection_ml_pipeline.match_detections_to_truth',
                           return_value=(np.full(n, -1), np.full(n, np.inf))):
                    merge_and_save_catalog(det, np.zeros((n, 2)), response, np.empty((0, 2)),
                                           td, fname=str(n), save_csv=True, verbose=False)
                with np.load(Path(td)/f'{n}.npz') as z:
                    for a in range(2):
                        for b in range(2):
                            np.testing.assert_array_equal(z[f'ml_R{a+1}{b+1}'], response[:, 0, a, b])
                with (Path(td)/f'{n}.csv').open() as f:
                    self.assertIn('ml_R12', next(csv.reader(f)))

    def test_catalog_report_and_mismatch(self):
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as td:
            cfg = {'round_dir': td, 'blended': {'index_range': [0, 10], 'seed': 42,
                                               'old_catalog': str(Path(td)/'old')}}
            for step in ('measure', 'catalog', 'rdsm', 'evaluate'):
                C.step_dir(cfg, step).mkdir()
            old = Path(td)/'old'
            old.mkdir()
            measured = blended.measurement_dir(cfg)
            measured.mkdir(parents=True)
            for exp in range(10):
                n = 15
                e = rng.normal(scale=.1, size=(n, 2))
                r = np.tile(np.array([[1., .2], [-.1, 1.3]]), (n, 1, 1))
                data = dict(det_id=np.arange(n), x=np.arange(n)*2., y=np.zeros(n),
                    fpfs_w=rng.uniform(.1, 1, n), fpfs_dw_dg1=rng.normal(scale=.01, size=n),
                    fpfs_dw_dg2=rng.normal(scale=.01, size=n), ml_e1=e[:, 0], ml_e2=e[:, 1],
                    truth_e1=np.full(n, np.nan))
                data.update({f'ml_R{a+1}{b+1}': r[:, a, b] for a in range(2) for b in range(2)})
                np.savez(measured/f'catalog_{exp}.npz', **data)
                np.savez(old/f'catalog_{exp}.npz', **{k: v for k, v in data.items() if k not in ('ml_R12', 'ml_R21')})
            with contextlib.redirect_stdout(io.StringIO()):
                blended.catalog(cfg)
            with np.load(C.step_dir(cfg, 'catalog')/'catalog.npz') as z:
                data = {k: z[k] for k in z.files}
            self.assertEqual(len(data['shapes']), 150)
            for split, count in ((0, 60), (1, 30), (2, 60)):
                self.assertEqual((data['split']==split).sum(), count)
            self.assertEqual(data['split'].dtype, np.uint8)
            self.assertFalse((C.step_dir(cfg, 'catalog')/'train_catalog.npz').exists())
            data.update(rdsm_e1=2*data['shapes'][:, 0], rdsm_e2=2*data['shapes'][:, 1],
                        rdsm_response=2*data['R_ana'][:, 0], rdsm_valid=np.ones(150, bool))
            np.savez(C.step_dir(cfg, 'rdsm')/'rdsm_catalog.npz', **data)
            blended.evaluate(cfg)
            with (C.step_dir(cfg, 'evaluate')/'metrics.csv').open() as f:
                rows = list(csv.DictReader(f))
            test = data['split'] == 2
            expected = data['raw_e'][test].sum(0) / np.diagonal(data['raw_R'][test], axis1=1, axis2=2).sum(0)
            for c in range(2):
                self.assertAlmostEqual(float(rows[c]['shear']), expected[c])
            for a, b in zip(rows[2:4], rows[4:6]):
                for key in ('shear', 'shape_noise', 'effective_noise'):
                    self.assertAlmostEqual(float(a[key]), float(b[key]))
                self.assertAlmostEqual(float(b['noise_ratio_to_detw']), 1.)
            with (C.step_dir(cfg, 'evaluate')/'offdiagonal_zero.csv').open() as f:
                zero = list(csv.DictReader(f))
            self.assertAlmostEqual(float(zero[0]['mean']), .2)
            self.assertAlmostEqual(float(zero[1]['mean']), -.1)
            path = old/'catalog_0.npz'
            with np.load(path) as z:
                bad = {k: z[k] for k in z.files}
            for key in ('ml_R11', 'ml_R22'):
                original = bad[key].copy()
                bad[key] = original + 9e-4
                np.savez(path, **bad)
                with contextlib.redirect_stdout(io.StringIO()):
                    blended.catalog(cfg)
                bad[key] = original + 2e-3
                np.savez(path, **bad)
                with self.assertRaisesRegex(ValueError, 'old/new mismatch'):
                    blended.catalog(cfg)
                bad[key] = original
            bad['ml_e1'] = bad['ml_e1'] + 9e-5
            np.savez(path, **bad)
            with self.assertRaisesRegex(ValueError, 'old/new mismatch'):
                blended.catalog(cfg)

    def test_norm_chain_rule_and_early_stopping(self):
        code = '''
import tempfile
from pathlib import Path
import numpy as np
import torch
from src.optuna.opt_toolkit import load_dataset
from src.ml_code.toolkits import model_gradients
with tempfile.TemporaryDirectory() as td:
    p = Path(td)/'data.npz'
    e = np.array([[.1,.3],[-.4,.2],[.2,-.1]], dtype=np.float32)
    r = np.tile(np.array([[1.,.4],[-.2,2.]], dtype=np.float32), (3,1,1))
    np.savez(p, shapes=e, R_ana=r[:,None])
    x, rx, norm = load_dataset(p, return_norm=True)
    assert len(load_dataset(p)) == 2
    np.testing.assert_allclose(x, e/norm)
    np.savez(p, shapes=e*7, R_ana=r[:,None]*7)
    xv, rv = load_dataset(p, norm=norm)
    np.testing.assert_allclose(xv, e*7/norm)
    class Linear(torch.nn.Linear):
        def forward(self, x, sigma=None): return super().forward(x)
    model = Linear(2, 2, bias=False)
    A = np.array([[2.,.7],[-.3,1.]], dtype=np.float32)
    with torch.no_grad(): model.weight.copy_(torch.tensor(A))
    y, g1, g2 = model_gradients(model, x, device=torch.device('cpu'), batch_size=2)
    result = np.einsum('nom,nmg->nog', np.stack([g1,g2],axis=1), np.asarray(rx))*norm
    np.testing.assert_allclose(result, np.einsum('om,nmg->nog', A, r), rtol=1e-6)
    np.testing.assert_allclose(y*norm, e@A.T, rtol=1e-6)
'''
        code += """
from unittest.mock import patch
import optuna
import optuna_eDSM as search
train_x = torch.tensor([[1., 2.], [3., 4.]])
train_r = torch.ones((2, 2, 2))
val_x, val_r = train_x + 100, train_r * 7
seen = {}
class Trainer:
    def __init__(self, model, moments, response, **kwargs):
        assert moments is train_x and response is train_r
    def train(self): return 0., []
def evaluate(**kwargs):
    assert kwargs['moments_n'] is val_x and kwargs['response_n'] is val_r
    seen['validation'] = True
    return dict(best_sigma=-1., best_bias=.001, min_bias_seen=.001, min_sn_seen=.2,
                best_sn=.2, scale_ratio=1., sigma_metrics=[], slope_fit=1., mono_spearman_rho=1.)
params = dict(batch_size=5000, lr=.001, weight_decay=1e-5, feat=64, depth=1,
              head_depth=2, noisy_dim='0,1')
with patch.object(search, 'DSMTrainer', Trainer), patch.object(search, 'evaluate_model_no_sigma', evaluate):
    objective = search.build_objective(train_x, train_r, .5, torch.device('cpu'), 32, 0,
                                       .1, 1e-12, eval_moments=val_x, eval_response=val_r)
    objective(optuna.trial.FixedTrial(params))
assert seen['validation']
with tempfile.TemporaryDirectory() as td:
    p = Path(td)/'single.npz'
    e = np.arange(12,dtype=np.float32).reshape(6,2)
    r = np.ones((6,1,2,2),dtype=np.float32)
    np.savez(p,shapes=e,R_ana=r,split=np.array([0,0,1,1,2,2],dtype=np.uint8))
    xt,rt,norm=load_dataset(p,split='train',return_norm=True)
    xv,rv=load_dataset(p,split='val',norm=norm)
    np.testing.assert_allclose(xt,e[:2]/norm)
    np.testing.assert_allclose(xv,e[2:4]/norm)
from src.ml_code.dsm_trainer import DSMConfig, DSMTrainer
class Monitored:
    def __init__(self,loss):
        self.model=torch.nn.Linear(2,2,bias=False)
        self.device=torch.device('cpu')
        self.config=DSMConfig(noisy_dim=(0,1),progress=False)
        self._noise_mask=torch.ones(1,2)
        self.value=loss
    _target_shear_score=staticmethod(DSMTrainer._target_shear_score)
    def _loss_fn(self,*args): return torch.tensor(self.value)
x=torch.randn(16,2);r=torch.eye(2).repeat(16,1,1)
study=optuna.create_study(directions=['minimize','minimize'])
t=study.ask();trainer=Monitored(1.)
monitor,restore=search.validation_monitor(trainer,t,x,r,4,6,0)
for epoch in range(1,7):
    with torch.no_grad(): trainer.model.weight.fill_(epoch)
    stopped=monitor(epoch,1.)
    assert stopped == (epoch==6)
restore()
assert torch.all(trainer.model.weight==1)
assert t.user_attrs['stop_reason']=='validation_plateau'
for _ in range(3):
    study.add_trial(optuna.trial.create_trial(values=[0.,0.],user_attrs={
        'pruning_group':str((0,1)),'validation_loss':[1.]*20}))
t=study.ask();trainer=Monitored(10.)
monitor,_=search.validation_monitor(trainer,t,x,r,20,6,0)
try:
    for epoch in range(1,8): monitor(epoch,10.)
    raise AssertionError('poor trial was not pruned')
except optuna.TrialPruned: pass
assert t.user_attrs['stop_reason']=='poor_validation_loss'
t=study.ask();monitor,_=search.validation_monitor(Monitored(1.),t,x,r,4,6,1e-12)
try:
    monitor(1,1.)
    raise AssertionError('timeout was not applied')
except optuna.TrialPruned: pass
t=study.ask();monitor,_=search.validation_monitor(Monitored(float('nan')),t,x,r,4,6,0)
try:
    monitor(1,1.)
    raise AssertionError('nonfinite loss was not pruned')
except optuna.TrialPruned: pass
model=torch.nn.Linear(2,2)
trainer=DSMTrainer(model,x,r,DSMConfig(epochs=5,batch_size=16,progress=False),device=torch.device('cpu'),
                   epoch_callback=lambda epoch,loss: epoch==1)
assert len(trainer.train()[1])==1
import subprocess,sys,json
from src.ml_code.moment_model import Spin2_ScoreMLP
with tempfile.TemporaryDirectory() as td:
    root=Path(td);p=root/'data.npz';n=80
    e=np.random.default_rng(2).normal(0,.2,(n,2)).astype(np.float32)
    r=np.tile(np.eye(2,dtype=np.float32),(n,1,1))
    np.savez(p,shapes=e,R_ana=r[:,None],split=np.repeat(np.arange(3,dtype=np.uint8),[40,20,20]),
             exposure_id=np.arange(n,dtype=np.int32),det_id=np.arange(n,dtype=np.int32))
    db=root/'DSMsigma0.5_test.db';name='DSMsigma0.5_test'
    study=optuna.create_study(storage=f'sqlite:///{db}',study_name=name,directions=['minimize','minimize'])
    params=dict(batch_size=5000,lr=.001,weight_decay=1e-5,feat=64,depth=1,head_depth=2,noisy_dim='0,1')
    for _ in range(2):study.enqueue_trial(params)
    result=subprocess.run([sys.executable,'optuna_eDSM.py','--data',str(p),'--eval-data',str(p),
        '--data-split','train','--eval-split','val','--storage',f'sqlite:///{db}',
        '--study-name',name,'--n-trials','2','--device','cpu','--n-bootstrap','0',
        '--bias-threshold','1e9','--early-stop-patience','2','--min-epochs','3'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    study=optuna.load_study(storage=f'sqlite:///{db}',study_name=name)
    assert len(study.trials)==2
    assert all('validation_loss' in t.user_attrs and t.user_attrs['epochs_ran']<=20 for t in study.trials)
    assert study.user_attrs['dataset']['data_split']=='train'
    model=Spin2_ScoreMLP(m=2,feat=64,depth=1,head_depth=2,use_sigma=False)
    checkpoint=root/'model.pt'
    torch.save(dict(params=params,trial_number=0,repeat_index=1,model_state_dict=model.state_dict()),checkpoint)
    out=root/'rdsm.npz'
    result=subprocess.run([sys.executable,'build_gauss_train_rdsm_catalog.py','--data',str(p),
        '--checkpoint',str(checkpoint),'--out',str(out),'--device','cpu','--norm','1','--compact'],
        capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    with np.load(out) as z:
        assert 'shapes' not in z.files and 'rdsm_response_normalized' not in z.files
        assert z['rdsm_response'].shape==(n,2,2)
"""
        result = subprocess.run([sys.executable, '-c', code], cwd=C.SG_DIR.parent/'opt_e',
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)


if __name__ == '__main__':
    unittest.main()
