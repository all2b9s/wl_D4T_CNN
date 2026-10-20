"""Apply matched truth i_ab<24.5 to existing response diagnostics."""
import csv,json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.blending.test_func import truth_magnitude

root=Path('/work/hdd/bfmo/wenyinli/measurement/ml_wl/r1_AttnW_last_detw')
out=root/'diagnostics/magnitude_response';out.mkdir(parents=True,exist_ok=True)
plots=Path('/projects/bdsp/wenyinli/codes/single_galaxies/output/ml_wl/r1_AttnW_last_detw/magnitude_response');plots.mkdir(parents=True,exist_ok=True)
cfg=json.loads((root/'config.resolved.json').read_text());source=Path(cfg['blended']['old_catalog'])
with np.load(root/'diagnostics/response_terms/response_terms.npz') as z:
    d={k:z[k] for k in z.files}
exp,det=d['exposure_id'],d['det_id'];mag=np.full(len(det),np.nan)
order=np.argsort(exp,kind='stable');unique,start=np.unique(exp[order],return_index=True);ends=np.r_[start[1:],len(order)]
def get(task):
    exposure,a,b=task;ix=order[a:b];path=source/f'catalog_{exposure}.npz'
    values=truth_magnitude(path,truth_dir=cfg['blended']['measurement']['blended_dir'])
    with np.load(path) as z:ids=z['det_id']
    sort=np.argsort(ids);p=np.searchsorted(ids[sort],det[ix]);assert np.all(p<len(ids));local=sort[p]
    assert np.array_equal(ids[local],det[ix])
    return ix,values[local]
with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('fork')) as pool:
    for n,(ix,v) in enumerate(pool.map(get,zip(unique,start,ends)),1):
        mag[ix]=v
        if n%250==0:print('Matched magnitudes',n,'/',len(unique),flush=True)
finite=np.isfinite(mag);bright=finite&(mag<24.5);faint=finite&~bright
R,E,W=d['R_norm'],d['ellipticity_term_norm'],d['weight_term_norm'];shift=d['shift'].max(1)
def quant(v):return dict(zip(['p50','p90','p99','p999','max'],np.quantile(v,[.5,.9,.99,.999,1]).tolist())) if len(v) else None
def stats(mask):
    q=dict(n=int(mask.sum()),R=quant(R[mask]),E=quant(E[mask]),W=quant(W[mask]),weight=quant(d['weight'][mask]),dw=quant(np.linalg.norm(d['dw'][mask],axis=1)),large={})
    for t in [1,10]:
        high=mask&(shift>t)
        q['large'][str(t)]=dict(n=int(high.sum()),fraction=float(high.sum()/mask.sum()) if mask.any() else None,
            weight_dominant=int((high&(W>E)).sum()),R2_fraction=float((R[high]**2).sum()/(R[mask]**2).sum()) if mask.any() else None)
    return q
masks={'all':np.ones(len(det),bool),'matched_all':finite,'i_ab_lt_24.5':bright,'i_ab_ge_24.5':faint,'unmatched_or_nonfinite_mag':~finite}
summary={k:stats(v) for k,v in masks.items()}
summary['definition']='Matched truth i_ab<24.5; unmatched/nonfinite excluded; original response and training normalization retained; no retraining or remeasurement.'
summary['bright_splits']={name:stats(bright&(d['split']==i)) for i,name in enumerate(['train','val','test'])}

def row(i):return dict(exposure_id=int(exp[i]),det_id=int(det[i]),split=['train','val','test'][d['split'][i]],i_ab=float(mag[i]),weight=float(d['weight'][i]),R=float(R[i]),ellipticity_term=float(E[i]),weight_term=float(W[i]),dw_dg1=float(d['dw'][i,0]),dw_dg2=float(d['dw'][i,1]),max_normalized_shift=float(shift[i]))
idx=np.flatnonzero(bright);idx=idx[np.argsort(R[idx])[::-1]]
summary['top10_bright']=[row(i) for i in idx[:10]]
keys=exp.astype(np.int64)*1000000+det
lookup={int(k):i for i,k in enumerate(keys)}
summary['previous_outliers']=[row(lookup[a*1000000+b]) for a,b in [(3791,870),(3225,75),(2938,23),(87,160)]]
for name,indices in [('top200_bright.csv',idx[:200]),('all_large_response_bright.csv',idx[shift[idx]>1])]:
    rows=[row(i) for i in indices]
    if rows:
        with (plots/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(out/'matched_magnitude.npz',exposure_id=exp,det_id=det,i_ab=mag,keep_i_ab_lt_24p5=bright)
fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
for name,mask,color in [('All',masks['all'],'#555555'),('Matched i_ab < 24.5',bright,'#2864b4'),('Matched i_ab >= 24.5',faint,'#df7427')]:
    for ax,values in zip(axes[:2],[R,shift]):
        v=np.sort(values[mask]);points=np.unique(np.geomspace(1,len(v),1500).astype(int));ranks=len(v)-points
        ax.loglog(v[ranks],points/len(v),label=f'{name} (N={len(v):,})',color=color)
axes[0].set(xlabel='Response Frobenius norm',ylabel='Fraction above threshold',title='Response tail')
axes[1].set(xlabel='max_j |0.02 R_j| / original norm',ylabel='Fraction above threshold',title='Finite perturbation tail')
axes[1].axvline(1,color='gray',ls=':');axes[0].legend(fontsize=8)
select=np.random.default_rng(3).choice(np.flatnonzero(finite),20000,replace=False)
select=np.unique(np.r_[select,idx[:200]])
axes[2].scatter(mag[select],R[select],s=4,alpha=.4,rasterized=True)
axes[2].axvline(24.5,color='red',ls='--');axes[2].set(yscale='log',xlabel='Matched truth i_ab',ylabel='Response norm',title='Magnitude versus response')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('AttnW: response tails before / after truth magnitude cut (no remeasurement)')
for ext in ['png','pdf']:fig.savefig(plots/f'magnitude_response.{ext}',dpi=180)
for dest in [out,plots]:(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
