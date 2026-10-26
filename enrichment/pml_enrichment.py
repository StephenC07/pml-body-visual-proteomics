"""Volume-normalised enrichment of particles inside PML bodies.

For each of the five segmented tomograms, compares the observed fraction of each
particle type inside the PML body with the fraction of the searched sample volume
that the body occupies. Three denominators are reported:

  A  tomogram XY footprint x Z-range of all detected particles (pooled types)
  B  3D bounding box of all detected particles (pooled types), no padding
  C  tomogram XY footprint x Z-range of the PML segmentation

The PML volume is clipped to the same region as the denominator in every case.
Enrichment = observed inside fraction / PML volume fraction.
"""
import os, glob
import numpy as np
import pandas as pd
import trimesh
from skimage import io, measure
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'

BASE = '/Users/stephencarter/Desktop/PMYFPbodypaper_draft'
SEG = '/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Tomo3/January_2025'
OUT = os.path.dirname(os.path.abspath(__file__))
PX = 0.992          # nm per bin4 pixel
DIV = 4             # STAR coords are bin1 pixels
PML_LABEL = 2
TOMOS = ['ts_01', 'ts_03', 'ts_08', 'ts_15', 'ts_16']
SEGFILES = {
    'ts_01': f'{SEG}/2/2_binnedtc(3).labels.tif',
    'ts_03': f'{SEG}/3/PML_area.labels.tiff',
    'ts_08': f'{SEG}/8/8_binnedtc.labels.tif',
    'ts_15': f'{SEG}/15/darkthreshold0815.labels.tif',
    'ts_16': f'{SEG}/16/darkthreshold0816.obj',
}
TYPES = {
    'nucleosome': 'split_star_files_nucleosome',
    'trinucleosome': 'split_star_files_7v9s',
    'tric_closed': 'split_star_files_tric_closed',
    'tric_open': 'split_star_files_tric_open',
    'pa28': 'split_star_files_p28',
}
COLORS = {'nucleosome': '#0072B2', 'trinucleosome': '#D55E00', 'tric': '#CC79A7', 'pa28': '#009E73'}


def load_star(fn):
    cols, rows, inblock = {}, [], False
    for line in open(fn):
        s = line.strip()
        if s.startswith('data_particles'):
            inblock = True; cols = {}
        elif inblock and s.startswith('_rln'):
            name, idx = s.split('#'); cols[name.strip()] = int(idx) - 1
        elif inblock and s and s[0] in '-0123456789':
            p = s.split()
            rows.append([float(p[cols['_rlnCoordinateX']]), float(p[cols['_rlnCoordinateY']]), float(p[cols['_rlnCoordinateZ']])])
    return np.array(rows).reshape(-1, 3)


def load_segmentation(tomo):
    f = SEGFILES[tomo]
    if f.endswith('.obj'):
        mesh = trimesh.load(f)
        # voxelise the mesh into a bin4 label volume so all tomograms are treated identically
        vox = mesh.voxelized(pitch=1.0).fill()
        pts = vox.points  # centres of filled voxels, in bin4 pixel XYZ
        vol_xyz = np.zeros((1024, 1024, 350), dtype=bool)
        ijk = np.round(pts).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(vol_xyz.shape)), axis=1)
        ijk = ijk[ok]
        vol_xyz[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        mesh.vertices = mesh.vertices * PX
        return mesh, vol_xyz
    v = io.imread(f)                          # (Z, Y, X)
    vol_xyz = np.transpose(v == PML_LABEL, (2, 1, 0))   # (X, Y, Z)
    verts, faces, normals, _ = measure.marching_cubes(vol_xyz.astype(np.uint8), level=0.5)
    mesh = trimesh.Trimesh(vertices=verts * PX, faces=faces, normals=normals)
    return mesh, vol_xyz


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


records, geometry = [], []
for tomo in TOMOS:
    mesh, vol = load_segmentation(tomo)
    nx, ny, nz = vol.shape
    parts = {t: load_star(f'{BASE}/Files_for_ultimate_comparision/{d}/{tomo}.star') / DIV for t, d in TYPES.items()}
    parts['tric'] = np.vstack([parts.pop('tric_closed'), parts.pop('tric_open')])
    pooled = np.vstack(list(parts.values()))            # bin4 px
    zlo, zhi = pooled[:, 2].min(), pooled[:, 2].max()
    zlo_i, zhi_i = int(np.floor(zlo)), int(np.ceil(zhi))
    bmin, bmax = pooled.min(0), pooled.max(0)
    bmin_i, bmax_i = np.floor(bmin).astype(int), np.ceil(bmax).astype(int)
    segz = np.where(vol.any(axis=(0, 1)))[0]
    # PML voxels and denominators (bin4 voxels)
    pml_A = vol[:, :, zlo_i:zhi_i + 1].sum(); den_A = nx * ny * (zhi_i - zlo_i + 1)
    pml_B = vol[bmin_i[0]:bmax_i[0] + 1, bmin_i[1]:bmax_i[1] + 1, bmin_i[2]:bmax_i[2] + 1].sum()
    den_B = np.prod(bmax_i - bmin_i + 1)
    pml_C = vol[:, :, segz.min():segz.max() + 1].sum(); den_C = nx * ny * (segz.max() - segz.min() + 1)
    geometry.append(dict(tomo=tomo, pml_voxels_total=int(vol.sum()),
                         particle_z_range_nm=(zhi - zlo) * PX, seg_z_range_nm=(segz.max() - segz.min() + 1) * PX,
                         bbox_xy_nm=f'{(bmax[0]-bmin[0])*PX:.0f} x {(bmax[1]-bmin[1])*PX:.0f}',
                         fA=pml_A / den_A, fB=pml_B / den_B, fC=pml_C / den_C))
    for t, xyz in parts.items():
        inside = mesh.contains(xyz * PX)
        records.append(dict(tomo=tomo, type=t, n=len(xyz), inside=int(inside.sum()),
                            fA=pml_A / den_A, fB=pml_B / den_B, fC=pml_C / den_C))
    print(tomo, 'done; inside counts:', {t: r['inside'] for r in records if r['tomo'] == tomo for t in [r['type']]})

geo = pd.DataFrame(geometry)
df = pd.DataFrame(records)
df['obs_frac'] = df.inside / df.n
for k in 'ABC':
    df[f'expected_inside_{k}'] = df.n * df[f'f{k}']
    df[f'enrich_{k}'] = df.obs_frac / df[f'f{k}']
geo.to_csv(f'{OUT}/geometry_per_tomogram.csv', index=False)
df.to_csv(f'{OUT}/enrichment_per_tomogram.csv', index=False)

# pooled across tomograms: expected inside = sum(n_t * f_t)
rows = []
for t in ['nucleosome', 'trinucleosome', 'tric', 'pa28']:
    d = df[df.type == t]
    n, k = int(d.n.sum()), int(d.inside.sum())
    lo, hi = wilson(k, n)
    r = dict(type=t, n=n, inside=k, obs_frac=k / n, ci_lo=lo, ci_hi=hi)
    for dk in 'ABC':
        exp_frac = float((d.n * d[f'f{dk}']).sum() / n)
        r[f'expected_frac_{dk}'] = exp_frac
        r[f'enrichment_{dk}'] = (k / n) / exp_frac
        r[f'enrich_ci_{dk}'] = f'{lo/exp_frac:.2f}-{hi/exp_frac:.2f}'
        r[f'binom_p_{dk}'] = stats.binomtest(k, n, exp_frac).pvalue
    rows.append(r)
pooled = pd.DataFrame(rows)
pooled.to_csv(f'{OUT}/enrichment_pooled.csv', index=False)

# cross-type tests on raw inside/outside counts
tab = np.array([[r['inside'], r['n'] - r['inside']] for r in rows])
chi2, p_chi, _, _ = stats.chi2_contingency(tab)
tab3 = tab[[0, 2, 3]]   # nucleosome, tric, pa28
chi2_3, p_chi3, _, _ = stats.chi2_contingency(tab3)
pair = {}
for a, b in [(0, 2), (0, 3), (2, 3), (1, 3), (1, 2)]:
    pair[f'{rows[a]["type"]} vs {rows[b]["type"]}'] = stats.fisher_exact(tab[[a, b]])[1]
# Cochran-Mantel-Haenszel stratified by tomogram, nucleosome vs pa28
def cmh(t1, t2):
    # Cochran-Mantel-Haenszel test and Mantel-Haenszel pooled odds ratio, stratified by tomogram
    num = den = 0.0; obs = exp = var = 0.0
    for tomo in TOMOS:
        A = df[(df.tomo == tomo) & (df.type == t1)].iloc[0]; B = df[(df.tomo == tomo) & (df.type == t2)].iloc[0]
        a, b, c, d = A.inside, A.n - A.inside, B.inside, B.n - B.inside
        n = a + b + c + d
        if n == 0: continue
        num += a * d / n; den += b * c / n
        r1, r2, c1, c2 = a + b, c + d, a + c, b + d
        obs += a; exp += r1 * c1 / n
        var += r1 * r2 * c1 * c2 / (n * n * (n - 1)) if n > 1 else 0
    orr = num / den if den > 0 else np.inf
    chi = (abs(obs - exp) - 0.5) ** 2 / var if var > 0 else np.nan
    return orr, float(stats.chi2.sf(chi, 1))

with open(f'{OUT}/summary.txt', 'w') as f:
    f.write('Geometry per tomogram (bin4 volumes 1024 x 1024 x 350 px; PX = 0.992 nm)\n')
    f.write(geo.to_string(index=False, float_format=lambda x: f'{x:.3f}') + '\n\n')
    f.write('Per tomogram, per type\n')
    f.write(df.to_string(index=False, float_format=lambda x: f'{x:.3f}') + '\n\n')
    f.write('Pooled across the five segmented tomograms\n')
    f.write(pooled.to_string(index=False, float_format=lambda x: f'{x:.3g}') + '\n\n')
    f.write(f'Chi-squared, inside/outside x 4 types: chi2 = {chi2:.2f}, p = {p_chi:.2g}\n')
    f.write(f'Chi-squared, nucleosome/TRiC/PA28 only: chi2 = {chi2_3:.2f}, p = {p_chi3:.2g}\n')
    for k, v in pair.items():
        f.write(f'Fisher exact {k}: p = {v:.2g}\n')
    for t1, t2 in [('nucleosome', 'pa28'), ('nucleosome', 'tric'), ('tric', 'pa28'), ('trinucleosome', 'pa28')]:
        orr, p = cmh(t1, t2)
        f.write(f'CMH stratified by tomogram {t1} vs {t2}: pooled OR = {orr:.2f}, p = {p:.2g}\n')
print(open(f'{OUT}/summary.txt').read())

# figure: observed vs expected inside fraction (denominator A) and enrichment per tomogram
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axes[0]
x = np.arange(4); w = 0.38
labels = ['Nucleosome', 'Trinucleosome', 'TRiC', 'PA28']
ax.bar(x - w / 2, pooled.expected_frac_A * 100, w, color='lightgrey', label='Expected (PML volume fraction)')
ax.bar(x + w / 2, pooled.obs_frac * 100, w, color=[COLORS[t] for t in pooled.type], label='Observed inside')
ax.errorbar(x + w / 2, pooled.obs_frac * 100, yerr=[(pooled.obs_frac - pooled.ci_lo) * 100, (pooled.ci_hi - pooled.obs_frac) * 100], fmt='none', ecolor='k', capsize=3, lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Particles inside PML body (%)')
ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
ax = axes[1]
for i, t in enumerate(['nucleosome', 'trinucleosome', 'tric', 'pa28']):
    d = df[df.type == t]
    ax.scatter(np.full(len(d), i) + np.linspace(-0.15, 0.15, len(d)), d.enrich_A, color=COLORS[t], s=30, zorder=3)
    ax.plot([i - 0.3, i + 0.3], [pooled.enrichment_A[i]] * 2, color='k', lw=2)
ax.axhline(1, ls='--', color='grey', lw=1)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Enrichment inside PML body\n(observed / expected)')
ax.set_yscale('log'); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(f'{OUT}/pml_enrichment.pdf'); fig.savefig(f'{OUT}/pml_enrichment.png', dpi=300)
print('saved figure')
