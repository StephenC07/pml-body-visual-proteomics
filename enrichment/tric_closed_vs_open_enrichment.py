"""Inside/outside enrichment of closed and open TRiC separately (25 Oct 2026), same geometry and
inside test as pml_enrichment.py (denominator A from geometry_per_tomogram.csv)."""
import numpy as np, pandas as pd, trimesh
from skimage import io, measure
from scipy import stats
BASE='/Users/stephencarter/Desktop/PMYFPbodypaper_draft'; SEG=f'{BASE}/Tomo3/January_2025'
PX=0.992; DIV=4; PML_LABEL=2
TOMOS=['ts_01','ts_03','ts_08','ts_15','ts_16']
SEGFILES={'ts_01':f'{SEG}/2/2_binnedtc(3).labels.tif','ts_03':f'{SEG}/3/PML_area.labels.tiff','ts_08':f'{SEG}/8/8_binnedtc.labels.tif','ts_15':f'{SEG}/15/darkthreshold0815.labels.tif','ts_16':f'{SEG}/16/darkthreshold0816.obj'}
TYPES={'nucleosome':'split_star_files_nucleosome','tric_closed':'split_star_files_tric_closed','tric_open':'split_star_files_tric_open','pa28':'split_star_files_p28'}
def load_star(fn):
    cols,rows,inblock={},[],False
    for line in open(fn):
        s=line.strip()
        if s.startswith('data_particles'): inblock=True; cols={}
        elif inblock and s.startswith('_rln'): name,idx=s.split('#'); cols[name.strip()]=int(idx)-1
        elif inblock and s and s[0] in '-0123456789':
            p=s.split(); rows.append([float(p[cols['_rlnCoordinateX']]),float(p[cols['_rlnCoordinateY']]),float(p[cols['_rlnCoordinateZ']])])
    return np.array(rows).reshape(-1,3)
def load_mesh(tomo):
    f=SEGFILES[tomo]
    if f.endswith('.obj'):
        mesh=trimesh.load(f); mesh.vertices=mesh.vertices*PX; return mesh
    v=io.imread(f); vol=np.transpose(v==PML_LABEL,(2,1,0))
    verts,faces,normals,_=measure.marching_cubes(vol.astype(np.uint8),level=0.5)
    return trimesh.Trimesh(vertices=verts*PX,faces=faces,normals=normals)
geo=pd.read_csv('geometry_per_tomogram.csv').set_index('tomo')
rec=[]
for tomo in TOMOS:
    mesh=load_mesh(tomo)
    for t,d in TYPES.items():
        xyz=load_star(f'{BASE}/Files_for_ultimate_comparision/{d}/{tomo}.star')/DIV
        ins=mesh.contains(xyz*PX)
        rec.append(dict(tomo=tomo,type=t,n=len(xyz),inside=int(ins.sum()),fA=geo.loc[tomo,'fA']))
df=pd.DataFrame(rec); df['enrich']=(df.inside/df.n)/df.fA
print(df.pivot(index='tomo',columns='type',values=['n','inside']).to_string())
print(df.pivot(index='tomo',columns='type',values='enrich').round(2).to_string())
df.to_csv('tric_closed_vs_open_per_tomogram.csv',index=False)
print('\nPOOLED (denominator A, expected = sum n_i * fA_i)')
P={}
for t in TYPES:
    s=df[df.type==t]; n=int(s.n.sum()); k=int(s.inside.sum()); exp=float((s.n*s.fA).sum()); f=exp/n
    lo,hi=stats.binomtest(k,n,f).proportion_ci(method='wilson'); p=stats.binomtest(k,n,f).pvalue
    P[t]=(k,n); print(f'{t:12s} {k:3d}/{n:3d} = {100*k/n:4.1f}%  expected {100*f:4.1f}%  enrichment {k/n/f:.2f} ({lo/f:.2f} to {hi/f:.2f})  binomial p = {p:.3f}')
def fisher(a,b):
    (k1,n1),(k2,n2)=P[a],P[b]; return stats.fisher_exact([[k1,n1-k1],[k2,n2-k2]])[1]
for a,b in [('tric_closed','tric_open'),('nucleosome','tric_closed'),('nucleosome','tric_open'),('tric_closed','pa28'),('tric_open','pa28')]:
    print(f'Fisher {a} vs {b}: p = {fisher(a,b):.3f}')
