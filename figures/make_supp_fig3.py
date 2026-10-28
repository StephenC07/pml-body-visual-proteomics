#!/usr/bin/env python3
"""
Supplementary Fig. 3: 5-panel 3D PML body renderings (p28 + TRiC only) for all segmented tomograms.
ts_01 and ts_03 are horizontally reflected.
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 11

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import os
import trimesh
from skimage import io, measure

BASE_DIR = '/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Files_for_ultimate_comparision'

SEGMENTATION_FILES = {
    'ts_01': '/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Tomo3/January_2025/2/2_binnedtc(3).labels.tif',
    'ts_03': '/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Tomo3/January_2025/3/PML_area.labels.tiff',
    'ts_08': '/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Tomo3/January_2025/8/8_binnedtc.labels.tif',
    'ts_15': '/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Tomo3/January_2025/15/darkthreshold0815.labels.tif',
    'ts_16': '/Users/stephencarter/Desktop/PMYFPbodypaper_draft/Tomo3/January_2025/16/darkthreshold0816.obj',
}

PARTICLE_DIRS = {
    'p28':         os.path.join(BASE_DIR, 'split_star_files_p28'),
    'tric_closed': os.path.join(BASE_DIR, 'split_star_files_tric_closed'),
    'tric_open':   os.path.join(BASE_DIR, 'split_star_files_tric_open'),
}

DIVIDE_BY = 4
X_TILT, Y_TILT, Z_EXAG = 25, 10, 1.2
VIEW_ELEV, VIEW_AZIM = 75, 265
HFLIP_TOMOS = ('ts_01', 'ts_03')

COLORS = {'p28': '#009E73', 'tric_closed': '#CC79A7', 'tric_open': '#E69F00'}
DISPLAY = {'p28': 'PA28', 'tric_closed': 'TRiC (closed)', 'tric_open': 'TRiC (open)'}
PTYPES = ['p28', 'tric_closed', 'tric_open']
TOMOS = ['ts_01', 'ts_03', 'ts_08', 'ts_15', 'ts_16']
LABELS = ['a', 'b', 'c', 'd', 'e']


def load_star_coords(filename):
    coords = []
    col_x = col_y = col_z = None
    in_data = False
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('data_particles'): in_data = True
            if line.startswith('_rlnCoordinateX'): col_x = int(line.split('#')[-1]) - 1
            elif line.startswith('_rlnCoordinateY'): col_y = int(line.split('#')[-1]) - 1
            elif line.startswith('_rlnCoordinateZ'): col_z = int(line.split('#')[-1]) - 1
            elif in_data and line and not line.startswith(('_', 'data_', 'loop_', '#')):
                if col_x is not None and col_y is not None and col_z is not None:
                    parts = line.split()
                    if len(parts) > max(col_x, col_y, col_z):
                        coords.append([float(parts[col_x]), float(parts[col_y]), float(parts[col_z])])
    return np.array(coords) if coords else np.array([]).reshape(0, 3)


def load_mesh(seg_file):
    if seg_file.endswith('.obj'):
        return trimesh.load(seg_file)
    volume = io.imread(seg_file)
    volume_reoriented = np.transpose(volume == 2, (2, 1, 0)).astype(np.uint8)  # binarise PML label as in enrichment_analysis/pml_enrichment.py
    verts, faces, normals, _ = measure.marching_cubes(volume_reoriented, level=0.5)
    return trimesh.Trimesh(vertices=verts, faces=faces, normals=normals)


def draw_panel(ax, tomo, seg_file, label):
    print(f"  Panel {label}: {tomo}...")
    mesh = load_mesh(seg_file)

    particle_data = {}
    all_coords_list, all_inside_list, all_types_list = [], [], []

    for ptype in PTYPES:
        sf = os.path.join(PARTICLE_DIRS[ptype], f'{tomo}.star')
        if not os.path.exists(sf): continue
        raw = load_star_coords(sf)
        if len(raw) == 0: continue
        px = raw / DIVIDE_BY
        inside = mesh.contains(px)
        n_in = int(np.sum(inside))
        particle_data[ptype] = {'coords': px, 'inside': inside, 'n_inside': n_in, 'n_total': len(px)}
        all_coords_list.append(px)
        all_inside_list.append(inside)
        all_types_list.extend([ptype] * len(px))

    if not all_coords_list:
        ax.text(0.5, 0.5, 0.5, f'{tomo}: No data', ha='center', va='center')
        return

    all_coords = np.vstack(all_coords_list)
    all_inside = np.concatenate(all_inside_list)
    all_types = np.array(all_types_list)

    # Transform mesh
    mc = mesh.bounds.mean(axis=0)
    wv = mesh.vertices.copy()
    wv[:, 2] = mc[2] + (wv[:, 2] - mc[2]) * Z_EXAG
    wv[:, 0] = 1024 - wv[:, 0]

    tx = np.radians(X_TILT); csx, snx = np.cos(tx), np.sin(tx)
    cy = 512; cz = (wv[:, 2].min() + wv[:, 2].max()) / 2
    yc = wv[:, 1] - cy; zc = wv[:, 2] - cz
    wv[:, 1] = yc * csx - zc * snx + cy; wv[:, 2] = yc * snx + zc * csx + cz

    ty = np.radians(Y_TILT); csy, sny = np.cos(ty), np.sin(ty)
    cx = 512; xc = wv[:, 0] - cx; zc2 = wv[:, 2] - cz
    wv[:, 0] = xc * csy + zc2 * sny + cx; wv[:, 2] = -xc * sny + zc2 * csy + cz
    wv[:, 0] = np.clip(wv[:, 0], 0, 1024); wv[:, 1] = np.clip(wv[:, 1], 0, 1024)

    # Transform particles
    pc = all_coords.copy()
    pc[:, 2] = mc[2] + (pc[:, 2] - mc[2]) * Z_EXAG; pc[:, 0] = 1024 - pc[:, 0]
    yp = pc[:, 1] - cy; zp = pc[:, 2] - cz
    pc[:, 1] = yp * csx - zp * snx + cy; pc[:, 2] = yp * snx + zp * csx + cz
    xp = pc[:, 0] - cx; zp2 = pc[:, 2] - cz
    pc[:, 0] = xp * csy + zp2 * sny + cx; pc[:, 2] = -xp * sny + zp2 * csy + cz

    # Horizontal reflection
    if tomo in HFLIP_TOMOS:
        wv[:, 0] = 1024 - wv[:, 0]
        pc[:, 0] = 1024 - pc[:, 0]

    # Render mesh
    ax.xaxis.set_pane_color((1, 1, 1, 0.24))
    ax.yaxis.set_pane_color((1, 1, 1, 0.20))
    ax.zaxis.set_pane_color((1, 1, 1, 0.36))
    ax.grid(True, alpha=0.15, linestyle=':', color='#bbbbbb')

    nf = len(mesh.faces)
    ss = int(max(5000, nf * 0.3))
    np.random.seed(42)
    fidx = np.random.choice(nf, min(ss, nf), replace=False) if nf > ss else np.arange(nf)
    mt = wv[mesh.faces[fidx]]
    mt[:, :, 0] = np.clip(mt[:, :, 0], 0, 1024)
    mt[:, :, 1] = np.clip(mt[:, :, 1], 0, 1024)

    vd = np.array([np.cos(np.radians(VIEW_AZIM)) * np.cos(np.radians(VIEW_ELEV)),
                    np.sin(np.radians(VIEW_AZIM)) * np.cos(np.radians(VIEW_ELEV)),
                    np.sin(np.radians(VIEW_ELEV))])
    fn = np.cross(mt[:, 1] - mt[:, 0], mt[:, 2] - mt[:, 0])
    nn = np.linalg.norm(fn, axis=1, keepdims=True); nn[nn == 0] = 1; fn /= nn
    fres = np.power(1 - np.abs(fn @ vd), 1.2)
    fc = np.zeros((len(fidx), 4)); fc[:, :3] = 0.8; fc[:, 3] = 0.65 + 0.2 * fres

    ax.add_collection3d(Poly3DCollection(mt, facecolors=fc, edgecolors=(0.6, 0.6, 0.6, 0.15), linewidths=0.3))

    # Particles
    handles = []
    n_inside_all = 0; n_total_all = 0
    for ptype in PTYPES:
        if ptype not in particle_data: continue
        pd = particle_data[ptype]
        tmask = all_types == ptype
        tc = pc[tmask]; ti = all_inside[tmask]
        col = COLORS[ptype]; name = DISPLAY[ptype]
        n_in = pd['n_inside']; n_tot = pd['n_total']
        n_inside_all += n_in; n_total_all += n_tot

        inc = tc[ti]
        if len(inc) > 0:
            ax.scatter(inc[:, 0], inc[:, 1], inc[:, 2], c=col, s=180, alpha=0.9,
                      edgecolors='white', linewidths=0.5, marker='o', depthshade=True, zorder=5)
        outc = tc[~ti]
        if len(outc) > 0:
            ax.scatter(outc[:, 0], outc[:, 1], outc[:, 2], c=col, s=40, alpha=0.5,
                      edgecolors='white', linewidths=0.2, marker='o', depthshade=True, zorder=3)
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=col, markersize=8,
                              markeredgecolor='white', markeredgewidth=0.4,
                              label=f'{name} ({n_in}/{n_tot})'))

    pad = 30
    zmin = min(wv[:, 2].min(), pc[:, 2].min()) - pad
    zmax = max(wv[:, 2].max(), pc[:, 2].max()) + pad
    ax.set_xlim(0, 1024); ax.set_ylim(0, 1024); ax.set_zlim(zmin, zmax)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlabel('X', fontsize=8, labelpad=3)
    ax.set_ylabel('Y', fontsize=8, labelpad=3)
    ax.set_zlabel('Z', fontsize=8, labelpad=3)
    ax.tick_params(axis='both', labelsize=7)
    ax.tick_params(axis='z', labelsize=7)

    pct = 100 * n_inside_all / n_total_all if n_total_all > 0 else 0
    ax.set_title(f'({label}) {tomo}\n{n_inside_all}/{n_total_all} inside ({pct:.1f}%)',
                fontsize=13, fontweight='bold', pad=8)

    ax.legend(handles=handles, loc='upper left', fontsize=8, frameon=True,
             fancybox=True, edgecolor='#cccccc', framealpha=0.85,
             handletextpad=0.3, borderpad=0.3)


def main():
    output_dir = '/Users/stephencarter/Desktop/Transfer_to_Mac_PML_NCB/Nature_Cell_Biology/FIGURES_CURRENT'
    os.makedirs(output_dir, exist_ok=True)

    # Layout: 2 rows. Top: 3 panels. Bottom: 2 panels centered.
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 6, figure=fig, hspace=0.22, wspace=0.15)

    # Row 1: A, B, C
    positions = [
        gs[0, 0:2],  # A
        gs[0, 2:4],  # B
        gs[0, 4:6],  # C
        gs[1, 1:3],  # D
        gs[1, 3:5],  # E
    ]

    for idx, (tomo, label) in enumerate(zip(TOMOS, LABELS)):
        ax = fig.add_subplot(positions[idx], projection='3d', computed_zorder=True)
        draw_panel(ax, tomo, SEGMENTATION_FILES[tomo], label)

    fig.text(0.5, 0.01, "Large markers = inside PML body, small markers = outside PML body",
            ha='center', fontsize=11, color='#666666', style='italic')

    fig.suptitle('3D distribution of PA28 and TRiC particles relative to PML bodies',
                fontsize=20, fontweight='bold', y=0.98)

    for fmt in ['png', 'pdf']:
        outpath = os.path.join(output_dir, f'Supplementary_Figure3.{fmt}')
        fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
