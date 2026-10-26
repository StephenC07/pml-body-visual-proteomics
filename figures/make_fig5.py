"""Figure 5 (28 Sept 2026): a to c IsoNet composite slices with mapped-back particles, d observed vs expected,
e per-body enrichment, f 3D render of TS_03 (old Fig. 5c). Pore panel dropped. Output Figure5_assembled.pdf/.png at 600 dpi."""
import numpy as np, pandas as pd, mrcfile
from skimage import io, measure
from scipy.ndimage import gaussian_filter
import matplotlib; matplotlib.use('Agg')
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from PIL import Image
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})

BASE = '/Users/stephencarter/Desktop/PMYFPbodypaper_draft'
ISO = '/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/tomograms/isonet_pml_deconv_2'
PX, DIV = 0.992, 4
# colours matched to the ChimeraX renders of Figs 2a, 3q and 5f (sampled 8 Oct 2026)
COL = {'nucleosome': '#5777C5', 'trinucleosome': '#BD45A6', 'tric_closed': '#E8221C', 'tric_open': '#87CEF8', 'pa28': '#FDDC0B'}
_tri = Path([(np.cos(a), np.sin(a)) for a in np.deg2rad([90, 210, 330])] + [(0, 0)], [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
MK = {'nucleosome': 'o', 'trinucleosome': 's', 'tric_closed': 'h', 'tric_open': 'H', 'pa28': _tri}
SIZE = {'nucleosome': 75, 'trinucleosome': 30, 'tric_closed': 70, 'tric_open': 70, 'pa28': 150}
LW = {'nucleosome': (1.1, 0.8), 'trinucleosome': (1.1, 0.8), 'tric_closed': (1.4, 1.1), 'tric_open': (1.4, 1.1), 'pa28': (1.4, 1.1)}
DIRS = {'nucleosome': 'split_star_files_nucleosome', 'trinucleosome': 'split_star_files_7v9s', 'tric_closed': 'split_star_files_tric_closed', 'tric_open': 'split_star_files_tric_open', 'pa28': 'split_star_files_p28'}
TOMO = {'TS_01': (f'{ISO}/2_binnedtc.mrc', f'{BASE}/Tomo3/January_2025/2/2_binnedtc(3).labels.tif', 'ts_01'),
        'TS_03': (f'{ISO}/3_1_binnedtc.mrc', f'{BASE}/Tomo3/January_2025/3/PML_area.labels.tiff', 'ts_03'),
        'TS_08': (f'{ISO}/8_binnedtc.mrc', f'{BASE}/Tomo3/January_2025/8/8_binnedtc.labels.tif', 'ts_08')}
DZ, SLAB, R, HALF = 15, 3, 14, 400
Z = {'TS_01': 187, 'TS_03': 181, 'TS_08': 157}   # chosen by Stephen 30 Sept 2026 from slice_candidates_isonet.png

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
    z = Z[name]
    with mrcfile.mmap(mrc, permissive=True) as m:
        vol = np.asarray(m.data[z - DZ:z + DZ + 1], dtype=np.float32)
    vol = np.stack([gaussian_filter(v, 0.7) for v in vol])
    sl = vol[DZ - SLAB:DZ + SLAB + 1].mean(axis=0)
    for t, xyz in parts.items():
        for x, y, pz in xyz[np.abs(xyz[:, 2] - z) <= DZ]:
            zi = int(round(pz)) - (z - DZ); xi, yi = int(round(x)), int(round(y))
            patch = vol[max(0, zi - 2):zi + 3].mean(axis=0)
            ys, xs = slice(max(0, yi - R), yi + R + 1), slice(max(0, xi - R), xi + R + 1)
            sl[ys, xs] = patch[ys, xs]
    lo, hi = np.percentile(sl, [0.5, 99.5]); sl = np.clip((sl - lo) / (hi - lo), 0, 1)
    yy, xx = np.nonzero(lab[z]); cy, cx = yy.mean(), xx.mean()
    x0, y0 = int(np.clip(cx - HALF, 0, 1024 - 2 * HALF)), int(np.clip(cy - HALF, 0, 1024 - 2 * HALF)); x1, y1 = x0 + 2 * HALF, y0 + 2 * HALF
    ax.imshow(sl[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1, interpolation='none', extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
    for c in measure.find_contours(lab[z].astype(float), 0.5): ax.plot(c[:, 1], c[:, 0], color='#F0E442', lw=0.9, ls='--')
    def inside(xyz):
        xi, yi, zi = np.round(xyz).astype(int).T
        ok = (xi >= 0) & (xi < 1024) & (yi >= 0) & (yi < 1024) & (zi >= 0) & (zi < lab.shape[0])
        out = np.zeros(len(xyz), bool); out[ok] = lab[zi[ok], yi[ok], xi[ok]]; return out
    for t, xyz in parts.items():
        sel = np.abs(xyz[:, 2] - z) <= DZ
        if not sel.any(): continue
        ins = inside(xyz[sel])
        for flag, alpha, lw in [(True, 0.35, LW[t][0]), (False, 0.0, LW[t][1])]:
            k = ins == flag
            if k.any(): ax.scatter(xyz[sel][k, 0], xyz[sel][k, 1], s=SIZE[t], marker=MK[t], facecolors=(*matplotlib.colors.to_rgb(COL[t]), alpha), edgecolors=COL[t], linewidths=lw, zorder=3)
    ax.plot([x0 + 25, x0 + 25 + 100 / PX], [y1 - 25, y1 - 25], color='white', lw=2.5); ax.text(x0 + 25, y1 - 38, '100 nm', color='white', fontsize=7)
    ax.set_xlim(x0 - 0.5, x1 - 0.5); ax.set_ylim(y1 - 0.5, y0 - 0.5); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(name, fontsize=7, loc='left')
    return z

FW, FH = 180, 135
fig = plt.figure(figsize=(FW / 25.4, FH / 25.4))
gs_top = fig.add_gridspec(1, 3, left=0.035, right=0.99, top=0.97, bottom=0.555, wspace=0.06)
gs_bot = fig.add_gridspec(1, 3, left=0.07, right=0.985, top=0.40, bottom=0.10, wspace=0.42)
zs = {}
for col, name in enumerate(['TS_01', 'TS_03', 'TS_08']):
    zs[name] = slice_panel(fig.add_subplot(gs_top[0, col]), name)
handles = [Line2D([], [], marker=MK[t], color=COL[t], ls='', ms=(8 if t == 'nucleosome' else 5 if t == 'trinucleosome' else 10.5 if t == 'pa28' else 7), mfc='none', mew=1.1, label=l) for t, l in [('nucleosome', 'Nucleosome'), ('trinucleosome', 'Trinucleosome'), ('tric_closed', 'TRiC closed'), ('tric_open', 'TRiC open'), ('pa28', 'PA28–20S')]]
handles += [Line2D([], [], marker='o', color='k', ls='', ms=5, mfc=(0, 0, 0, 0.35), mew=1.1, label='inside body'), Line2D([], [], marker='o', color='k', ls='', ms=5, mfc='none', label='outside body'), Line2D([], [], color='#F0E442', ls='--', lw=0.9, label='PML body boundary')]
yb = gs_top[0, 0].get_position(fig).y0
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, yb - 0.005), ncol=8, frameon=False, fontsize=6.3, handletextpad=0.4, columnspacing=1.0)

df = pd.read_csv('/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/Nature_Cell_Biology/enrichment_analysis/enrichment_per_tomogram.csv')
def wilson(k, n, zz=1.96):
    p = k / n; d = 1 + zz * zz / n; c = (p + zz * zz / (2 * n)) / d; h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)) / d; return c - h, c + h
order = ['nucleosome', 'trinucleosome', 'tric', 'pa28']; labels = ['Nucleosome', 'Trinucleosome', 'TRiC', 'PA28–20S']; CB = {'nucleosome': '#5777C5', 'trinucleosome': '#BD45A6', 'tric': '#E8221C', 'pa28': '#FDDC0B'}
pooled = []
for t in order:
    d = df[df.type == t]; n, k = int(d.n.sum()), int(d.inside.sum()); ef = float((d.n * d.fA).sum() / n); lo, hi = wilson(k, n)
    pooled.append(dict(type=t, obs=k / n, exp=ef, lo=lo, hi=hi, enr=(k / n) / ef))
pooled = pd.DataFrame(pooled)
ax = fig.add_subplot(gs_bot[0, 0]); x = np.arange(4); w = 0.38
ax.bar(x - w / 2, pooled.exp * 100, w, color='#BBBBBB', label='Expected from body volume')
ax.bar(x + w / 2, pooled.obs * 100, w, color=[CB[t] for t in order], label='Observed inside body')
ax.errorbar(x + w / 2, pooled.obs * 100, yerr=[(pooled.obs - pooled.lo) * 100, (pooled.hi - pooled.obs) * 100], fmt='none', ecolor='k', elinewidth=0.7, capsize=2)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right'); ax.set_ylabel('Particles inside PML body (%)')
ax.legend(frameon=False, fontsize=6, loc='upper right'); ax.spines[['top', 'right']].set_visible(False); ax.set_ylim(0, 27)
ax = fig.add_subplot(gs_bot[0, 1])
tomos = ['ts_01', 'ts_03', 'ts_08', 'ts_15', 'ts_16']; tmk = ['o', 's', '^', 'D', 'v']
for i, t in enumerate(order):
    d = df[df.type == t].set_index('tomo').loc[tomos]; enr = (d.inside / d.n) / d.fA
    for j, e in enumerate(enr):
        ax.scatter(i + (j - 2) * 0.12, e, marker=tmk[j], s=16, facecolors=CB[t], edgecolors='k', linewidths=0.4, zorder=3)
    ax.plot([i - 0.32, i + 0.32], [pooled.enr[i]] * 2, color='k', lw=1.6, zorder=4)
ax.axhline(1, ls='--', color='grey', lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha='right'); ax.set_ylabel('Enrichment inside body\n(observed / expected)')
ax.set_ylim(-0.05, 1.9); ax.spines[['top', 'right']].set_visible(False)
ax.legend(handles=[Line2D([], [], marker=m, color='k', ls='', ms=4, mfc='white', label=t.upper()) for t, m in zip(tomos, tmk)], frameon=False, fontsize=5.5, loc='upper left', ncol=2, handletextpad=0.3, columnspacing=0.8)

ax = fig.add_subplot(gs_bot[0, 2]); ax.imshow(np.asarray(Image.open('render_old5c_crop.png')), interpolation='none'); ax.axis('off')
ax.set_title('TS_03', fontsize=7, loc='left')
for g, c, L, dx in [(gs_top, 0, 'a', 0.02), (gs_top, 1, 'b', 0.02), (gs_top, 2, 'c', 0.02), (gs_bot, 0, 'd', 0.045), (gs_bot, 1, 'e', 0.045), (gs_bot, 2, 'f', 0.045)]:
    b = g[0, c].get_position(fig); fig.text(b.x0 - dx, b.y1 + 0.012, L, fontsize=9, fontweight='bold')
fig.savefig('Figure5_assembled.pdf', dpi=600); fig.savefig('Figure5_assembled.png', dpi=600)
print(zs, pooled.round(3).to_dict('records'))
