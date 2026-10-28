"""Supplementary Fig. 6c: PA28 DBSCAN cluster-size histogram (eps 100 bin4 px, ~99 nm, min_samples 3), label PA28.
Same analysis as split_star_files_p28/nearest_neighbor_analysis.py."""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
sys.path.insert(0, '/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Files_for_ultimate_comparision/split_star_files_p28')
from nearest_neighbor_analysis import load_star_file, analyze_clusters_main, EPS_MAIN, MIN_SAMPLES_MAIN, COLOR_P28
plt.rcParams['font.family'] = 'Arial'; plt.rcParams['font.size'] = 12; plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5; plt.rcParams['ytick.major.width'] = 1.5
star_files = sorted(glob.glob('/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Files_for_ultimate_comparision/split_star_files_p28/ts_*.star'))
cluster_sizes, total_p, total_cl, total_n = analyze_clusters_main(star_files, EPS_MAIN, MIN_SAMPLES_MAIN)
print('star files', len(star_files), 'particles', total_p, 'clusters', len(cluster_sizes), 'sizes', sorted(cluster_sizes, reverse=True))
fig, ax = plt.subplots(figsize=(6, 6))
cutoff = 10
small = cluster_sizes[cluster_sizes <= cutoff]; large = cluster_sizes[cluster_sizes > cutoff]
unique_sizes, counts = np.unique(small, return_counts=True)
ax.bar(unique_sizes, counts, width=0.6, color=COLOR_P28, edgecolor='black', linewidth=1.5, label='PA28')
for pos, count in zip(unique_sizes, counts):
    ax.text(pos, count + 0.3, str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
if len(large) > 0:
    ax.bar(cutoff + 1.5, len(large), width=0.6, color=COLOR_P28, edgecolor='black', linewidth=1.5, hatch='///')
    ax.text(cutoff + 1.5, len(large) + 0.3, f'{len(large)}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(cutoff + 1.5, len(large) + 1.2, '(sizes: ' + ', '.join(str(s) for s in sorted(large, reverse=True)) + ')', ha='center', va='bottom', fontsize=9, color='gray', style='italic')
ax.set_xticks(list(range(3, cutoff + 1)) + [cutoff + 1.5]); ax.set_xticklabels(list(range(3, cutoff + 1)) + ['>10'])
ax.set_xlabel('Cluster size (number of particles)', fontsize=18, fontweight='bold')
ax.set_ylabel('Number of clusters', fontsize=18, fontweight='bold')
ax.set_title(f'Distribution of cluster sizes\n(eps={EPS_MAIN}, min_samples={MIN_SAMPLES_MAIN})', fontsize=20, fontweight='bold')
ax.legend(fontsize=14, frameon=False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', labelsize=16); ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
plt.tight_layout()
out = '/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/Nature_Cell_Biology/FIGURES_CURRENT/Supplementary_Figure6c'
plt.savefig(out + '.png', dpi=300, bbox_inches='tight', facecolor='white'); plt.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
print('saved', out)
