"""New Figure 5 panels. a, b: TS_03 and TS_08 composite slices with mapped-back particles;
c: observed vs expected inside fraction; d: per-body enrichment. Boxes are left for the
3D render (f) and the pore-size histogram (e), which are reused from the Illustrator file."""
import json
import numpy as np, pandas as pd, mrcfile
from skimage import io, measure
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})

BASE = '/Users/stephencarter/Desktop/PMYFPbodypaper_draft'
PX, DIV = 0.992, 4
COL = {'nucleosome': '#0072B2', 'trinucleosome': '#D55E00', 'tric_closed': '#CC79A7', 'tric_open': '#E69F00', 'pa28': '#009E73', 'tric': '#CC79A7'}
_tri = Path([(np.cos(a), np.sin(a)) for a in np.deg2rad([90, 210, 330])] + [(0, 0)], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
MK = {'nucleosome': 'o', 'trinucleosome': 's', 'tric_closed': 'h', 'tric_open': 'H', 'pa28': _tri}
SIZE = {'nucleosome': 30, 'trinucleosome': 30, 'tric_closed': 70, 'tric_open': 70, 'pa28': 70}
LW = {'nucleosome': (1.1, 0.8), 'trinucleosome': (1.1, 0.8), 'tric_closed': (1.4, 1.1), 'tric_open': (1.4, 1.1), 'pa28': (1.4, 1.1)}
DIRS = {'nucleosome': 'split_star_files_nucleosome', 'trinucleosome': 'split_star_files_7v9s', 'tric_closed': 'split_star_files_tric_closed', 'tric_open': 'split_star_files_tric_open', 'pa28': 'split_star_files_p28'}
TOMO = {'TS_01': (f'{BASE}/Segmentation_claude_mac/2_binnedtc.mrc', f'{BASE}/Tomo3/January_2025/2/2_binnedtc(3).labels.tif', 'ts_01'),
        'TS_03': (f'{BASE}/Segmentation_claude_mac/3_1_binnedtc.mrc', f'{BASE}/Tomo3/January_2025/3/PML_area.labels.tiff', 'ts_03'),
        'TS_08': (f'{BASE}/Segmentation_claude_mac/8_binnedtc.mrc', f'{BASE}/Tomo3/January_2025/8/8_binnedtc.labels.tif', 'ts_08')}
DZ, SLAB, R, HALF = 15, 3, 14, 400

def load_star(fn):
    cols, rows, inb = {}, [], False
    for line in open(fn):
        s = line.strip()
        if s.startswith('data_particles'): inb = True; cols = {}
        elif inb and s.startswith('_rln'): n, i = s.split('#'); cols[n.strip()] = int(i) - 1
        elif inb and s and s[0] in '-0123456789':
            p = s.split(); rows.append([float(p[cols['_rlnCoordinateX']]), float(p[cols['_rlnCoordinateY']]), float(p[cols['_rlnCoordinateZ']])])
    return np.array(rows).reshape(-1, 3) / DIV

def slice_panel(ax, name):
    mrc, labf, star = TOMO[name]
    lab = io.imread(labf) == 2
    parts = {t: load_star(f'{BASE}/Files_for_ultimate_comparision/{d}/{star}.star') for t, d in DIRS.items()}
    allz = np.concatenate([p[:, 2] for p in parts.values()])
    zs = np.where(lab.any(axis=(1, 2)))[0]
    z = int(max(range(zs.min() + 20, zs.max() - 20), key=lambda zz: (np.abs(allz - zz) <= DZ).sum()))
    with mrcfile.mmap(mrc, permissive=True) as m:
        vol = np.asarray(m.data[z - DZ:z + DZ + 1], dtype=np.float32)
    vol = np.stack([gaussian_filter(v, 1.0) for v in vol])
    sl = vol[DZ - SLAB:DZ + SLAB + 1].mean(axis=0)
    for t, xyz in parts.items():                      # each particle patch from its own z
        for x, y, pz in xyz[np.abs(xyz[:, 2] - z) <= DZ]:
            zi = int(round(pz)) - (z - DZ); xi, yi = int(round(x)), int(round(y))
            patch = vol[max(0, zi - 2):zi + 3].mean(axis=0)
            ys, xs = slice(max(0, yi - R), yi + R + 1), slice(max(0, xi - R), xi + R + 1)
            sl[ys, xs] = patch[ys, xs]
    lo, hi = np.percentile(sl, [0.5, 99.5]); sl = np.clip((sl - lo) / (hi - lo), 0, 1)
    yy, xx = np.nonzero(lab[z]); cy, cx = yy.mean(), xx.mean()
    x0, y0 = int(np.clip(cx - HALF, 0, 1024 - 2 * HALF)), int(np.clip(cy - HALF, 0, 1024 - 2 * HALF)); x1, y1 = x0 + 2 * HALF, y0 + 2 * HALF
    ax.imshow(sl[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1, interpolation='bicubic', extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
    for c in measure.find_contours(lab[z].astype(float), 0.5): ax.plot(c[:, 1], c[:, 0], color='#F0E442', lw=0.9, ls='--')
    def inside(xyz):
        xi, yi, zi = np.round(xyz).astype(int).T
        ok = (xi >= 0) & (xi < 1024) & (yi >= 0) & (yi < 1024) & (zi >= 0) & (zi < lab.shape[0])
        out = np.zeros(len(xyz), bool); out[ok] = lab[zi[ok], yi[ok], xi[ok]]; return out
    n = 0
    for t, xyz in parts.items():
        sel = np.abs(xyz[:, 2] - z) <= DZ
        if not sel.any(): continue
        ins = inside(xyz[sel]); n += int(sel.sum())
        for flag, alpha, lw in [(True, 0.35, LW[t][0]), (False, 0.0, LW[t][1])]:
            k = ins == flag
            if k.any(): ax.scatter(xyz[sel][k, 0], xyz[sel][k, 1], s=SIZE[t], marker=MK[t], facecolors=(*matplotlib.colors.to_rgb(COL[t]), alpha), edgecolors=COL[t], linewidths=lw, zorder=3)
    ax.plot([x0 + 25, x0 + 25 + 100 / PX], [y1 - 25, y1 - 25], color='white', lw=2.5); ax.text(x0 + 25, y1 - 38, '100 nm', color='white', fontsize=7)
    ax.set_xlim(x0 - 0.5, x1 - 0.5); ax.set_ylim(y1 - 0.5, y0 - 0.5); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(f'{name}, z = {z} ± {DZ} slices', fontsize=7, loc='left')
    return z, n

FW, FH = 180, 200
fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.95, 0.95], hspace=0.5, wspace=0.35, left=0.06, right=0.98, top=0.97, bottom=0.05)
zs_n = {}
for col, name in enumerate(['TS_01', 'TS_03', 'TS_08']):
    ax_ = fig.add_subplot(gs[0, col]); zs_n[name] = slice_panel(ax_, name)
handles = [Line2D([], [], marker=MK[t], color=COL[t], ls='', ms=(5 if t in ('nucleosome', 'trinucleosome') else 7), mfc='none', mew=1.1, label=l) for t, l in [('nucleosome', 'Nucleosome'), ('trinucleosome', 'Trinucleosome'), ('tric_closed', 'TRiC closed'), ('tric_open', 'TRiC open'), ('pa28', 'PA28–20S')]]
handles += [Line2D([], [], marker='o', color='k', ls='', ms=5, mfc=(0, 0, 0, 0.35), mew=1.1, label='inside body'), Line2D([], [], marker='o', color='k', ls='', ms=5, mfc='none', label='outside body'), Line2D([], [], color='#F0E442', ls='--', lw=0.9, label='PML body boundary')]
top_row_bottom = gs[0, 0].get_position(fig).y0
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, top_row_bottom - 0.005), ncol=8, frameon=False, fontsize=6.3, handletextpad=0.4, columnspacing=1.0)
fig.text(0.52, top_row_bottom - 0.045, 'Particles are shown at their own z within the ±15-slice (30 nm) slab', ha='center', fontsize=6, color='dimgrey')

# ---------------- panels c, d (TRiC = 29/194 as in the manuscript)
df = pd.read_csv('/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/Nature_Cell_Biology/enrichment_analysis/enrichment_per_tomogram.csv')
df.loc[(df.tomo == 'ts_08') & (df.type == 'tric'), 'inside'] += 1
def wilson(k, n, zz=1.96):
    p = k / n; d = 1 + zz * zz / n; c = (p + zz * zz / (2 * n)) / d; h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)) / d; return c - h, c + h
order = ['nucleosome', 'trinucleosome', 'tric', 'pa28']; labels = ['Nucleosome', 'Trinucleosome', 'TRiC', 'PA28–20S']
pooled = []
for t in order:
    d = df[df.type == t]; n, k = int(d.n.sum()), int(d.inside.sum()); ef = float((d.n * d.fA).sum() / n); lo, hi = wilson(k, n)
    pooled.append(dict(type=t, obs=k / n, exp=ef, lo=lo, hi=hi, enr=(k / n) / ef))
pooled = pd.DataFrame(pooled)
ax = fig.add_subplot(gs[1, 0]); x = np.arange(4); w = 0.38
ax.bar(x - w / 2, pooled.exp * 100, w, color='#BBBBBB', label='Expected from body volume')
ax.bar(x + w / 2, pooled.obs * 100, w, color=[COL[t] for t in order], label='Observed inside body')
ax.errorbar(x + w / 2, pooled.obs * 100, yerr=[(pooled.obs - pooled.lo) * 100, (pooled.hi - pooled.obs) * 100], fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right'); ax.set_ylabel('Particles inside PML body (%)')
ax.legend(frameon=False, fontsize=6, loc='upper right'); ax.spines[['top', 'right']].set_visible(False)
ax = fig.add_subplot(gs[1, 1])
tomos = ['ts_01', 'ts_03', 'ts_08', 'ts_15', 'ts_16']; tmk = ['o', 's', '^', 'D', 'v']
for i, t in enumerate(order):
    d = df[df.type == t].set_index('tomo').loc[tomos]; enr = (d.inside / d.n) / d.fA
    for j, e in enumerate(enr):
        ax.scatter(i + (j - 2) * 0.12, e, marker=tmk[j], s=16, facecolors=COL[t], edgecolors='k', linewidths=0.4, zorder=3)
    ax.plot([i - 0.32, i + 0.32], [pooled.enr[i]] * 2, color='k', lw=1.6, zorder=4)
ax.axhline(1, ls='--', color='grey', lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right'); ax.set_ylabel('Enrichment inside body\n(observed / expected)')
ax.set_ylim(-0.05, 1.9); ax.spines[['top', 'right']].set_visible(False)
ax.legend(handles=[Line2D([], [], marker=m, color='k', ls='', ms=4, mfc='white', label=t.upper()) for t, m in zip(tomos, tmk)], frameon=False, fontsize=5.5, loc='upper left', ncol=2, handletextpad=0.3, columnspacing=0.8)

# placeholders: e = pore-size histogram (reuse), f = 3D render (reuse)
ph = {}
for key, pos in [('pore', gs[1, 2]), ('render', gs[2, 0:2])]:
    bb = pos.get_position(fig); ph[key] = dict(x0=bb.x0, y0=bb.y0, x1=bb.x1, y1=bb.y1)
json.dump(dict(fig_w_mm=FW, fig_h_mm=FH, placeholders=ph), open('placeholders.json', 'w'), indent=1)
for (r, c), L in [((0, 0), 'a'), ((0, 1), 'b'), ((0, 2), 'c'), ((1, 0), 'd'), ((1, 1), 'e'), ((1, 2), 'f'), ((2, 0), 'g')]:
    fig.text(gs[r, c].get_position(fig).x0 - 0.035, gs[r, c].get_position(fig).y1 + 0.008, L, fontsize=9, fontweight='bold')
fig.savefig('figure5_new_panels.pdf'); fig.savefig('figure5_new_panels.png', dpi=300)
print(zs_n)
