"""Supplementary Fig. 10: per-body particle enrichment versus fibre network excess (a, b) and sensitivity of the
pooled and per-body enrichment to the body boundary position (c). Inputs: enrichment_analysis/enrichment_per_tomogram.csv,
enrichment_analysis/boundary_sensitivity.csv, pml-network-density-accounted/results_v2/fibre_after_blob_removal_isonet.json."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5})
E = '/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/Nature_Cell_Biology/enrichment_analysis/'
FB = json.load(open('/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/pml-network-density-accounted/results_v2/fibre_after_blob_removal_isonet.json'))
B = ['ts_01', 'ts_03', 'ts_08', 'ts_15', 'ts_16']; MK = dict(ts_01='o', ts_03='s', ts_08='^', ts_15='D', ts_16='v')
fib = {t: FB[t.upper()]['3 >=15 nm removed']['ratio'] for t in B}
df = pd.read_csv(E + 'enrichment_per_tomogram.csv'); bs = pd.read_csv(E + 'boundary_sensitivity.csv')
COL = {'nucleosome': '#5777C5', 'trinucleosome': '#BD45A6', 'tric': '#E8221C', 'pa28': '#FDDC0B'}; LAB = {'nucleosome': 'Nucleosome', 'trinucleosome': 'Trinucleosome', 'tric': 'TRiC', 'pa28': 'PA28–20S'}
fig = plt.figure(figsize=(180 / 25.4, 62 / 25.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.5, left=0.07, right=0.99, top=0.88, bottom=0.2)
axA, axB, axC = [fig.add_subplot(gs[i]) for i in range(3)]
for ax, typ in ((axA, 'pa28'), (axB, 'tric')):
    for t in B:
        r = df[(df.tomo == t) & (df.type == 'nucleosome')].enrich_A.item(); ax.plot(fib[t], r, MK[t], color='#bbbbbb', ms=4, mfc='white', mew=0.7)
        r = df[(df.tomo == t) & (df.type == typ)].enrich_A.item(); ax.plot(fib[t], r, MK[t], color=COL[typ], ms=5, mfc=COL[typ], mec='k', mew=0.5)
    x = [fib[t] for t in B]; y = [df[(df.tomo == t) & (df.type == typ)].enrich_A.item() for t in B]; rho, p = stats.spearmanr(x, y)
    ax.axhline(1, color='k', ls='--', lw=0.7); ax.set_xlim(1.2, 1.65); ax.set_ylim(-0.05, 1.8); ax.set_xlabel('fibre network, body / nucleoplasm'); ax.set_ylabel('enrichment inside body')
    ax.set_title(f'{LAB[typ]} (ρ = {rho:.2f}, p = {p:.2f})', fontsize=7); ax.spines[['top', 'right']].set_visible(False)
axA.plot([], [], 'o', color='#bbbbbb', mfc='white', ms=4, label='nucleosome, same body'); axA.legend(frameon=False, fontsize=6, loc='upper right')
shifts = sorted(bs.shift_nm.unique())
for typ in ['nucleosome', 'trinucleosome', 'tric', 'pa28']:
    pooled = []
    for s in shifts:
        d = bs[(bs.type == typ) & (bs.shift_nm == s)]; pooled.append(d.inside.sum() / (d.n * d.exp_frac).sum())
    for t in B:
        d = bs[(bs.type == typ) & (bs.tomo == t)].sort_values('shift_nm'); axC.plot(d.shift_nm, d.inside / (d.n * d.exp_frac), MK[t], color=COL[typ], ms=2.5, mfc='white', mew=0.6, ls='none')
    axC.plot(shifts, pooled, '-', color=COL[typ], lw=1.8, label=LAB[typ])
axC.axhline(1, color='k', ls='--', lw=0.7); axC.axvline(0, color='k', lw=0.5, alpha=0.5); axC.set_ylim(0, 2.0); axC.set_xlabel('boundary shift (nm)'); axC.set_ylabel('enrichment inside body'); axC.set_xticks([-20, -10, 0, 10, 20]); axC.set_xticklabels(['−20', '−10', '0', '+10', '+20'])
axC.legend(frameon=False, fontsize=6, ncol=2, loc='upper left', handlelength=1.2, columnspacing=0.8); axC.spines[['top', 'right']].set_visible(False)
fig.canvas.draw()
for ax, L in ((axA, 'a'), (axB, 'b'), (axC, 'c')):
    b = ax.get_position(); fig.text(b.x0 - 0.045, b.y1 + 0.02, L, fontsize=11, fontweight='bold', va='bottom', ha='left')
fig.savefig('Supplementary_Figure10.pdf', dpi=600); fig.savefig('Supplementary_Figure10.png', dpi=600); print('written')
for typ in ('pa28', 'tric'):
    x = [fib[t] for t in B]; y = [df[(df.tomo == t) & (df.type == typ)].enrich_A.item() for t in B]; print(typ, stats.spearmanr(x, y))
