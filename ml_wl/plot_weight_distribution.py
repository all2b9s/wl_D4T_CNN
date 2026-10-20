"""Plot original detection weights for exactly the objects in a round catalog."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('/work/hdd/bfmo/wenyinli/measurement/ml_wl/r1_AttnW_last_detw')
source = Path(json.loads((root/'02_catalog/catalog_audit.json').read_text())['old_catalog'])
out = Path('/projects/bdsp/wenyinli/codes/single_galaxies/output/ml_wl/r1_AttnW_last_detw')
with np.load(root/'02_catalog/catalog.npz') as z:
    exposures, ids = z['exposure_id'], z['det_id']
order = np.argsort(exposures, kind='stable')
unique, starts = np.unique(exposures[order], return_index=True)
ends = np.r_[starts[1:], len(order)]
w = np.empty(len(ids), dtype=np.float64)
for i, (exp, start, end) in enumerate(zip(unique, starts, ends)):
    rows = order[start:end]
    with np.load(source/f'catalog_{exp}.npz') as z:
        det, weight = z['det_id'], z['fpfs_w']
    ix = np.argsort(det)
    pos = np.searchsorted(det[ix], ids[rows])
    assert np.all(pos < len(det))
    assert np.array_equal(det[ix[pos]], ids[rows])
    w[rows] = weight[ix[pos]]
    if (i+1) % 500 == 0:
        print(f'Read {i+1}/{len(unique)} exposures', flush=True)
valid = np.isfinite(w) & (w > 0)
logw = np.log10(w[valid])
thresholds = [1e-5, 1e-4, 1e-3, 1e-2, .1, .5, .9]
stats = dict(n_objects=len(w), source=str(source), nonfinite=int((~np.isfinite(w)).sum()),
             zero=int((w==0).sum()), negative=int((w<0).sum()), above_one=int((w>1).sum()),
             min=float(w[valid].min()), max=float(w[valid].max()),
             quantiles=dict(zip(['p01','p05','p10','p25','p50','p75','p90','p99'],np.quantile(w[valid],[.01,.05,.1,.25,.5,.75,.9,.99]).tolist())),
             below={str(t):dict(count=int((w<t).sum()),fraction=float((w<t).mean())) for t in thresholds})
fig, ax = plt.subplots(1, 3, figsize=(15,4.7), constrained_layout=True)
bins=np.linspace(min(-6, float(logw.min())), max(0,float(logw.max())),121)
counts, _=np.histogram(logw,bins)
for a in ax[:2]:
    a.stairs(counts,bins,color='#3269ab',linewidth=1.5,fill=True,alpha=.65)
    a.set_xlabel(r'$\log_{10}(w)$');a.set_ylabel('Objects per bin');a.grid(alpha=.2)
ax[0].set_title('Detection weight distribution')
ax[1].set_yscale('log');ax[1].set_title('Same histogram, log count axis')
sorted_log=np.sort(logw)
indices=np.unique(np.linspace(0,len(sorted_log)-1,min(20000,len(sorted_log))).astype(int))
ax[2].plot(sorted_log[indices],(indices+1)/len(sorted_log),color='#3269ab')
ax[2].set_xlabel(r'$\log_{10}(w)$');ax[2].set_ylabel('Cumulative fraction');ax[2].set_title('Fraction below a weight threshold');ax[2].grid(alpha=.2)
fig.suptitle(f'AttnW: original detection weights | Same {len(w):,} catalog objects',fontsize=13)
for ext in ['png','pdf']:
    fig.savefig(out/f'log_weight_distribution.{ext}',dpi=180)
(out/'log_weight_distribution.json').write_text(json.dumps(stats,indent=2)+'\n')
print(json.dumps(stats,indent=2),flush=True)
