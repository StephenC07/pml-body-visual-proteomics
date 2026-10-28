"""Radius at which Ripley's L(r) is maximal, per tomogram and analysis, using the functions of ppa_ripleys_L_whole.py
(N_SIM reduced to 2 because only L_obs is needed). Result (26 Oct 2026): L(r) still rising at the 200 nm limit in 38 of 41
curves, so clustering increases over the whole range examined."""
import sys, os, glob, io, contextlib, numpy as np
sys.path.insert(0,'/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Files_for_ultimate_comparision/ppa_ripleys')
import ppa_ripleys_L_whole as m
m.N_SIM=2
nuc={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(m.NUC_DIR,'ts_*.star'))}
tri={os.path.splitext(os.path.basename(f))[0]:f for f in glob.glob(os.path.join(m.TRI_DIR,'ts_*.star'))}
for tomo in sorted(set(nuc)|set(tri)):
    n=m.load_star(nuc[tomo]) if tomo in nuc else np.empty((0,3)); t=m.load_star(tri[tomo]) if tomo in tri else np.empty((0,3))
    if len(n)+len(t)<m.MIN_N: continue
    with contextlib.redirect_stdout(io.StringIO()): res=m.analyse_tomo(tomo,n,t)
    for key in ('nuc','tri','cross'):
        r=res.get(key)
        if r is None: continue
        L=np.asarray(r['L_obs']); i=int(np.argmax(L)); print(f"{tomo:8}{key:7} r_at_Lmax {m.R_NM[i]:6.1f} nm  Lmax {L[i]:6.1f} nm")
