#!/usr/bin/env python3
"""
PML body visual proteomics: classify particles inside vs outside a
segmented PML nuclear body, and render a 3D overview.

Given an Amira label TIFF (or a pre-exported .obj mesh) of one PML
body and one or more RELION/PyTOM STAR files of particle coordinates,
the script:

    1. Loads the Amira segmentation (TIFF label file, value matching
       --pml-label) and extracts a triangulated surface via
       `skimage.measure.marching_cubes` at the specified isovalue,
       transposing axis order (Z, Y, X) -> (X, Y, Z) to match the
       RELION XYZ convention of the particle STAR files.  A
       pre-exported .obj mesh can be passed directly instead.
    2. Builds a `trimesh.Trimesh` object and applies standard repair
       routines (fix normals, fill holes, remove degenerate faces).
    3. Loads particle coordinates from each STAR file, scales them
       from unbinned STAR-file pixels to the mesh's bin-N pixel space
       (--unbinned-to-mesh-scale), and then to nanometres via
       --mesh-pixel-size-nm.
    4. Classifies every particle as inside or outside the PML body
       by ray-casting with `Trimesh.contains`.
    5. Writes per-particle-type inside/outside counts and per-particle
       classifications, and renders a 3D figure with the PML body
       mesh as a semi-transparent surface and particles as spheres
       (large = inside, small = outside).

Accompanies:
    Pražák V et al. In situ molecular architecture of eYFP-PML.I
    bodies by cryo-CLEM and cryo-electron tomography reveals
    trinucleosomes in an open state.

Example:
    python pml_body_visual_proteomics.py \\
        --segmentation path/to/pml.tif \\
        --particles p28:path/to/p28.star TRiC_closed:path/to/tric_closed.star \\
        --pml-label 2 \\
        --unbinned-to-mesh-scale 4 \\
        --mesh-pixel-size-nm 0.992 \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import io, measure

DEFAULT_COLOURS = [
    '#009E73', '#CC79A7', '#E69F00', '#0072B2',
    '#D55E00', '#56B4E9', '#F0E442', '#999999',
]


# ────────────────────────────────────────────────────────────────────
# STAR file parsing
# ────────────────────────────────────────────────────────────────────

def load_star_coords(star_path: Path) -> np.ndarray:
    """Extract (x, y, z) rows from a RELION/PyTOM STAR file."""
    coords = []
    col_x = col_y = col_z = None
    with open(star_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith('_rlnCoordinateX'):
                col_x = int(s.split('#')[1]) - 1
            elif s.startswith('_rlnCoordinateY'):
                col_y = int(s.split('#')[1]) - 1
            elif s.startswith('_rlnCoordinateZ'):
                col_z = int(s.split('#')[1]) - 1
            elif (s and not s.startswith('#') and not s.startswith('_')
                  and not s.startswith('data_') and not s.startswith('loop_')
                  and col_x is not None):
                parts = s.split()
                if len(parts) > max(col_x, col_y, col_z):
                    coords.append([float(parts[col_x]),
                                   float(parts[col_y]),
                                   float(parts[col_z])])
    return np.array(coords) if coords else np.empty((0, 3))


# ────────────────────────────────────────────────────────────────────
# Mesh extraction
# ────────────────────────────────────────────────────────────────────

def load_mesh(seg_path: Path, pml_label: int, isovalue: float) -> trimesh.Trimesh:
    """Build a triangulated surface mesh from a segmentation file.

    Accepts either a pre-exported `.obj` (returned as-is after repair) or
    a multi-label Amira TIFF.  For TIFF input, the volume is binarised
    on `pml_label`, axes are transposed (Z, Y, X) -> (X, Y, Z) to match
    the RELION XYZ coordinate order of the particle STAR files, and a
    triangulated surface is extracted by marching cubes at `isovalue`.
    """
    if seg_path.suffix.lower() == '.obj':
        mesh = trimesh.load(seg_path, force='mesh')
    else:
        volume = io.imread(str(seg_path))
        if pml_label is not None:
            volume = (volume == pml_label).astype(np.uint8)
        volume_xyz = np.transpose(volume, (2, 1, 0))
        verts, faces, normals, _ = measure.marching_cubes(volume_xyz,
                                                          level=isovalue)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                               vertex_normals=normals)

    # Repair routines (no-ops if already watertight)
    mesh.fix_normals()
    trimesh.repair.fill_holes(mesh)
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


# ────────────────────────────────────────────────────────────────────
# Classification
# ────────────────────────────────────────────────────────────────────

def classify_particles(star_coords_unbinned: np.ndarray,
                       mesh: trimesh.Trimesh,
                       unbinned_to_mesh_scale: float,
                       mesh_pixel_size_nm: float) -> dict:
    """Return {'coords_nm': (N,3), 'inside': (N,) bool}.

    Particle STAR coords are in unbinned pixels.  They are divided by
    `unbinned_to_mesh_scale` to reach the binned pixel space where the
    mesh lives (e.g. 4 when STAR coords are unbinned and the mesh was
    built from bin-4 segmentations), then multiplied by
    `mesh_pixel_size_nm` to give nanometres.  The mesh vertices are
    likewise rescaled to nm, so ray-casting is performed in nm.
    """
    if len(star_coords_unbinned) == 0:
        return {'coords_nm': np.empty((0, 3)), 'inside': np.empty((0,), dtype=bool)}

    coords_mesh_px = star_coords_unbinned / unbinned_to_mesh_scale
    coords_nm = coords_mesh_px * mesh_pixel_size_nm
    inside = mesh.contains(coords_nm)
    return {'coords_nm': coords_nm, 'inside': inside.astype(bool)}


# ────────────────────────────────────────────────────────────────────
# Rendering
# ────────────────────────────────────────────────────────────────────

def render(mesh: trimesh.Trimesh,
           classified: dict[str, dict],
           colours: dict[str, str],
           out_path: Path,
           title: str) -> None:
    """Render a 3D figure with the PML body mesh and particles."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 0.6))

    mesh_nm = mesh.copy()
    if mesh_nm.vertices.max() < 100:
        # treat as already nm-scaled; otherwise assume the caller did it
        pass

    # Mesh as semi-transparent surface
    face_verts = mesh_nm.vertices[mesh_nm.faces]
    poly = Poly3DCollection(face_verts, alpha=0.08, facecolor='#888888',
                            edgecolor='none')
    ax.add_collection3d(poly)

    for label, data in classified.items():
        if len(data['coords_nm']) == 0:
            continue
        c = colours[label]
        inside = data['inside']
        pts_in = data['coords_nm'][inside]
        pts_out = data['coords_nm'][~inside]
        if len(pts_out):
            ax.scatter(pts_out[:, 0], pts_out[:, 1], pts_out[:, 2],
                       c=c, s=10, alpha=0.55, edgecolor='none')
        if len(pts_in):
            ax.scatter(pts_in[:, 0], pts_in[:, 1], pts_in[:, 2],
                       c=c, s=70, alpha=0.9, edgecolor='black', linewidths=0.5)

    handles = []
    for label, data in classified.items():
        n_in = int(data['inside'].sum())
        n_total = len(data['inside'])
        handles.append(Line2D([0], [0], marker='o', linestyle='',
                              color=colours[label],
                              label=f'{label} ({n_in}/{n_total})',
                              markersize=7, markeredgecolor='black',
                              markeredgewidth=0.3))
    ax.legend(handles=handles, loc='upper left', fontsize=9, frameon=True)
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    ax.set_title(title, fontsize=11)

    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[1],
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--segmentation', required=True, type=Path,
                   help='Amira TIFF label file or pre-exported .obj mesh.')
    p.add_argument('--particles', required=True, nargs='+',
                   metavar='LABEL:PATH',
                   help='Particle STAR files, given as "LABEL:path". '
                        'May be repeated.  LABEL is used in the figure '
                        'legend and output files.')
    p.add_argument('--pml-label', type=int, default=2,
                   help='Integer label for the PML body in the TIFF '
                        '(default: 2).  Ignored for .obj input.')
    p.add_argument('--isovalue', type=float, default=0.5,
                   help='Marching-cubes isovalue (default: 0.5).')
    p.add_argument('--unbinned-to-mesh-scale', type=float, required=True,
                   help='Division factor to bring STAR coordinates from '
                        'unbinned pixels into the mesh coordinate space '
                        '(e.g. 4 if the mesh is from a bin-4 segmentation).')
    p.add_argument('--mesh-pixel-size-nm', type=float, required=True,
                   help='Pixel size of the mesh coordinate space in nm '
                        '(e.g. 0.992 for bin-4 at 2.48 Angstrom unbinned).')
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--title', default=None,
                   help='Figure title (default: segmentation basename).')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    particle_specs = []
    for spec in args.particles:
        if ':' not in spec:
            sys.exit(f'--particles entries must be LABEL:PATH (got {spec!r})')
        label, path = spec.split(':', 1)
        particle_specs.append((label, Path(path)))

    print(f'[mesh] loading {args.segmentation}', flush=True)
    mesh = load_mesh(args.segmentation, args.pml_label, args.isovalue)
    mesh.vertices = mesh.vertices * args.mesh_pixel_size_nm
    print(f'       {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces', flush=True)

    colours = {lbl: DEFAULT_COLOURS[i % len(DEFAULT_COLOURS)]
               for i, (lbl, _) in enumerate(particle_specs)}

    classified = {}
    summary_rows = []
    per_particle_rows = []
    for label, path in particle_specs:
        print(f'[particles] {label}: {path}', flush=True)
        raw = load_star_coords(path)
        data = classify_particles(raw, mesh,
                                  args.unbinned_to_mesh_scale,
                                  args.mesh_pixel_size_nm)
        classified[label] = data
        n_in = int(data['inside'].sum())
        n_total = len(data['inside'])
        pct = (100 * n_in / n_total) if n_total else 0.0
        print(f'            {n_in}/{n_total} inside ({pct:.1f}%)', flush=True)
        summary_rows.append({'label': label, 'total': n_total,
                             'inside': n_in, 'outside': n_total - n_in,
                             'percent_inside': round(pct, 2)})
        for (x, y, z), ins in zip(data['coords_nm'], data['inside']):
            per_particle_rows.append({
                'label': label,
                'x_nm': round(x, 3), 'y_nm': round(y, 3), 'z_nm': round(z, 3),
                'inside': bool(ins),
            })

    summary_path = args.output_dir / 'inside_outside_counts.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['label', 'total', 'inside', 'outside', 'percent_inside'])
        w.writeheader()
        w.writerows(summary_rows)

    per_particle_path = args.output_dir / 'classifications.csv'
    with open(per_particle_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['label', 'x_nm', 'y_nm', 'z_nm', 'inside'])
        w.writeheader()
        w.writerows(per_particle_rows)

    figure_path = args.output_dir / 'rendering.png'
    render(mesh, classified, colours, figure_path,
           title=args.title or args.segmentation.stem)

    meta = {
        'segmentation': str(args.segmentation),
        'pml_label': args.pml_label,
        'isovalue': args.isovalue,
        'unbinned_to_mesh_scale': args.unbinned_to_mesh_scale,
        'mesh_pixel_size_nm': args.mesh_pixel_size_nm,
        'particles': {lbl: str(p) for lbl, p in particle_specs},
        'summary': summary_rows,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(meta, indent=2))

    print(f'\nDone. Outputs in {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
