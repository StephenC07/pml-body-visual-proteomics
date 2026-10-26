#!/usr/bin/env python3
"""
Combined figure: 5 tomograms x (bar plot + table) for p28 and TRiC inside/outside PML body.
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 11

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import csv

COLORS = {
    'PA28':           '#009E73',
    'TRiC (closed)':  '#CC79A7',
    'TRiC (open)':    '#E69F00',
}

PARTICLE_MAP = {
    'p28': 'PA28',
    'tric_closed': 'TRiC (closed)',
    'tric_open': 'TRiC (open)',
}

TOMOS = ['ts_01', 'ts_03', 'ts_08', 'ts_15', 'ts_16']


def load_data(csv_path):
    all_data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tomo = row['tomogram']
            ptype = row['particle_type']
            if ptype not in PARTICLE_MAP:
                continue
            display_name = PARTICLE_MAP[ptype]
            if tomo not in all_data:
                all_data[tomo] = {}
            all_data[tomo][display_name] = {
                'inside': int(row['n_inside']),
                'outside': int(row['n_outside']),
                'total': int(row['n_total']),
            }
    return all_data


def make_row(ax_bar, ax_table, tomo, data, panel_label=''):
    names = [n for n in ['PA28', 'TRiC (closed)', 'TRiC (open)'] if n in data]
    if not names:
        ax_bar.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_bar.transAxes)
        ax_table.axis('off')
        return

    inside = [data[n]['inside'] for n in names]
    outside = [data[n]['outside'] for n in names]
    totals = [data[n]['total'] for n in names]
    pct_inside = [100 * data[n]['inside'] / data[n]['total'] if data[n]['total'] > 0 else 0
                  for n in names]

    # --- Bar plot ---
    x = np.arange(len(names))
    bar_width = 0.55
    y_max = max(totals) if totals else 10

    ax_bar.bar(x, inside, bar_width, label='Inside PML body',
               color=[COLORS[n] for n in names], edgecolor='black', linewidth=1.0)
    ax_bar.bar(x, outside, bar_width, bottom=inside, label='Outside PML body',
               color=[COLORS[n] for n in names], edgecolor='black', linewidth=1.0,
               alpha=0.35, hatch='///')

    for i, (n_in, n_out) in enumerate(zip(inside, outside)):
        if n_in > 0:
            ax_bar.text(i, n_in / 2, str(n_in), ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white',
                       path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        if n_out > 0:
            ax_bar.text(i, n_in + n_out / 2, str(n_out), ha='center', va='center',
                       fontsize=10, fontweight='bold', color='#333333')
        ax_bar.text(i, n_in + n_out + y_max * 0.02, f'{pct_inside[i]:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax_bar.set_ylabel('Particles', fontsize=12, fontweight='bold')
    title = f'({panel_label}) {tomo}' if panel_label else tomo
    ax_bar.set_title(title, fontsize=16, fontweight='bold')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.tick_params(axis='y', labelsize=10)
    ax_bar.set_ylim(0, y_max * 1.18)

    # --- Table ---
    ax_table.axis('off')

    cell_text = []
    for n in names:
        d = data[n]
        pct = 100 * d['inside'] / d['total'] if d['total'] > 0 else 0
        cell_text.append([n, str(d['total']), str(d['inside']), str(d['outside']), f'{pct:.1f}%'])

    tric_in = data.get('TRiC (closed)', {}).get('inside', 0) + data.get('TRiC (open)', {}).get('inside', 0)
    tric_out = data.get('TRiC (closed)', {}).get('outside', 0) + data.get('TRiC (open)', {}).get('outside', 0)
    tric_tot = tric_in + tric_out
    tric_pct = 100 * tric_in / tric_tot if tric_tot > 0 else 0
    cell_text.append(['TRiC (combined)', str(tric_tot), str(tric_in), str(tric_out), f'{tric_pct:.1f}%'])

    all_in = sum(inside)
    all_out = sum(outside)
    all_tot = all_in + all_out
    all_pct = 100 * all_in / all_tot if all_tot > 0 else 0
    cell_text.append(['All', str(all_tot), str(all_in), str(all_out), f'{all_pct:.1f}%'])

    col_labels = ['Type', 'Total', 'In', 'Out', '% In']

    table = ax_table.table(cellText=cell_text, colLabels=col_labels,
                           cellLoc='center', loc='center',
                           colWidths=[0.32, 0.14, 0.14, 0.14, 0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)

    for i in range(len(cell_text)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            if i < len(names):
                cell.set_facecolor(mcolors.to_rgba(COLORS[names[i]], alpha=0.15))
            elif i == len(names):
                cell.set_facecolor(mcolors.to_rgba('#8E44AD', alpha=0.15))
            else:
                cell.set_facecolor('#eeeeee')
                cell.set_text_props(fontweight='bold')
            cell.set_edgecolor('#cccccc')


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'segmentation_classification_summary_final.csv')
    output_dir = os.path.join(base_dir, '..', 'FIGURES_CURRENT')
    os.makedirs(output_dir, exist_ok=True)

    all_data = load_data(csv_path)

    fig = plt.figure(figsize=(16, 28))
    gs = GridSpec(5, 2, figure=fig, width_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # Add shared legend at top
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#888888', edgecolor='black', label='Inside PML body'),
        Patch(facecolor='#888888', edgecolor='black', alpha=0.35, hatch='///', label='Outside PML body'),
    ]

    panel_labels = ['a', 'b', 'c', 'd', 'e']
    for row_idx, tomo in enumerate(TOMOS):
        ax_bar = fig.add_subplot(gs[row_idx, 0])
        ax_table = fig.add_subplot(gs[row_idx, 1])

        if tomo in all_data:
            make_row(ax_bar, ax_table, tomo, all_data[tomo], panel_labels[row_idx])
        else:
            ax_bar.text(0.5, 0.5, f'{tomo}: No data', ha='center', va='center',
                       transform=ax_bar.transAxes, fontsize=14)
            ax_table.axis('off')

        # Add legend only to first row
        if row_idx == 0:
            ax_bar.legend(fontsize=11, frameon=False, loc='upper right')

    fig.suptitle('PA28 and TRiC particles inside vs outside PML bodies\nacross five segmented tomograms',
                fontsize=22, fontweight='bold', y=0.995)

    for fmt in ['png', 'pdf']:
        outpath = os.path.join(output_dir, f'Supplementary_Figure4.{fmt}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
