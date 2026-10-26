"""Random-labelling null for the fraction of mixed nucleosome/trinucleosome DBSCAN clusters.
Reruns the joint DBSCAN of dbscan_combined_nuc_trinuc.py (eps 100 bin4 px for TS_01/03/16, 75 otherwise, min_samples 3),
then permutes the particle-type labels within each tomogram 2,000 times with the cluster assignments fixed.
Result (26 Oct 2026): 111/136 mixed observed (81.6%), null mean 113.3 (83.3%), 95% 107-119, P(null <= obs) = 0.28."""
import sys, os, glob, importlib.util, numpy as np
import matplotlib; matplotlib.use('Agg')
from sklearn.cluster import DBSCAN
SRC='/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Files_for_ultimate_comparision'
spec=importlib.util.spec_from_file_location('dbm', os.path.join(SRC,'dbscan_combined_nuc_trinuc.py')); dbm=importlib.util.module_from_spec(spec); spec.loader.exec_module(dbm)
rng=np.random.default_rng(0); NPERM=2000
nuc={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(dbm.NUC_DIR,'ts_*.star'))}
tri={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(dbm.TRI_DIR,'ts_*.star'))}
per=[]
for tomo in sorted(set(nuc)|set(tri)):
    n=dbm.load_star_file(nuc[tomo]) if tomo in nuc else np.empty((0,3)); t=dbm.load_star_file(tri[tomo]) if tomo in tri else np.empty((0,3))
    if len(n)+len(t)<3: continue
    X=np.vstack([c for c in (n,t) if len(c)]); pt=np.array([0]*len(n)+[1]*len(t))
    lab=DBSCAN(eps=dbm.EPS_PER_TOMO.get(tomo,dbm.DEFAULT_EPS),min_samples=3).fit_predict(X)
    masks=[lab==k for k in set(lab) if k!=-1]
    mixed=lambda p: sum(1 for mk in masks if p[mk].min()!=p[mk].max())
    obs=mixed(pt); null=np.array([mixed(rng.permutation(pt)) for _ in range(NPERM)])
    per.append((tomo,len(masks),obs,null))
    print(f"{tomo:7} clusters {len(masks):3d} mixed_obs {obs:3d} null {null.mean():6.1f} ({np.percentile(null,2.5):.0f}-{np.percentile(null,97.5):.0f})")
tot_cl=sum(p[1] for p in per); tot_obs=sum(p[2] for p in per); tot_null=sum(p[3] for p in per)
print(f"TOTAL clusters {tot_cl}, mixed {tot_obs} ({100*tot_obs/tot_cl:.1f}%), null {tot_null.mean():.1f} ({100*tot_null.mean()/tot_cl:.1f}%), 95% {np.percentile(tot_null,2.5):.0f}-{np.percentile(tot_null,97.5):.0f}, P(null<=obs)={np.mean(tot_null<=tot_obs):.3f}")
