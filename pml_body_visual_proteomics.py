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
    2. Builds a `trimesh.Trimesh` object from the vertices and faces,
       used as exported without repair.
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
    Pražák V et al. In situ architecture of PML nuclear bodies reveals
    a porous, chromatin-permissive interior (submitted to Nature Cell
    Biology, 2026).

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

def load_mesh(seg_path: Path, pml_label: int, isovalue: float,
              binarise: bool = True) -> trimesh.Trimesh:
    """Build a triangulated surface mesh from a segmentation file.

    Accepts either a pre-exported `.obj` or a multi-label Amira TIFF.
    For TIFF input, axes are transposed (Z, Y, X) -> (X, Y, Z) to match
    the RELION XYZ coordinate order of the particle STAR files and a
    triangulated surface is extracted by marching cubes at `isovalue`.

    When `binarise=True` (default) the volume is first reduced to a
    uint8 mask where only voxels equal to `pml_label` are 1.  This is
    the correct general-purpose behaviour for multi-label segmentations.
    When `binarise=False` the raw multi-value volume is passed to
    marching cubes; for a single-label segmentation this shifts the
    sub-voxel iso-surface position (e.g. 0 <-> 2 at `level=0.5` sits at
    25 % of the gradient rather than 50 %), which can move the mesh
    surface by up to one voxel and reclassify particles that sit on
    the boundary.  Use `--no-binarise` for bit-exact reproduction of
    paper-specific numbers produced with binary segmentations.
    """
    if seg_path.suffix.lower() == '.obj':
        return trimesh.load(seg_path, force='mesh')

    volume = io.imread(str(seg_path))
    if binarise and pml_label is not None:
        volume = (volume == pml_label).astype(np.uint8)
    volume_xyz = np.transpose(volume, (2, 1, 0))
    verts, faces, _, _ = measure.marching_cubes(volume_xyz, level=isovalue)
    return trimesh.Trimesh(vertices=verts, faces=faces)


# ────────────────────────────────────────────────────────────────────
# Classification
# ────────────────────────────────────────────────────────────────────

def classify_particles(star_coords_unbinned: np.ndarray,
                       mesh: trimesh.Trimesh,
                       unbinned_to_mesh_scale: float,
                       mesh_pixel_size_nm: float) -> dict:
    """Return {'coords_nm': (N,3), 'inside': (N,) bool}.

    Particle STAR coords are in unbinned pixels and the mesh is in the
    segmentation's binned pixel space.  Ray-casting via `Trimesh.contains`
    is performed in the mesh's native pixel space (STAR coords divided
    by `unbinned_to_mesh_scale`) because `contains()` has absolute-scale
    numerical tolerances and the result can shift by one particle at
    the mesh boundary if the coordinates are pre-scaled to nanometres.
    The nanometre coordinates returned here are derived post-hoc for
    reporting and rendering only.
    """
    if len(star_coords_unbinned) == 0:
        return {'coords_nm': np.empty((0, 3)), 'inside': np.empty((0,), dtype=bool)}

    coords_mesh_px = star_coords_unbinned / unbinned_to_mesh_scale
    inside = mesh.contains(coords_mesh_px)
    coords_nm = coords_mesh_px * mesh_pixel_size_nm
    return {'coords_nm': coords_nm, 'inside': inside.astype(bool)}


# ────────────────────────────────────────────────────────────────────
# Rendering
# ────────────────────────────────────────────────────────────────────

def render(mesh: trimesh.Trimesh,
           classified: dict[str, dict],
           colours: dict[str, str],
           out_path: Path,
           title: str,
           mesh_pixel_size_nm: float) -> None:
    """Render a 3D figure with the PML body mesh and particles.

    The mesh is passed in mesh-pixel space and scaled to nm locally
    for display; this matches the nm-scale particle coordinates in
    `classified['coords_nm']`.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 0.6))

    mesh_nm = mesh.copy()
    mesh_nm.vertices = mesh_nm.vertices * mesh_pixel_size_nm

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
    p.add_argument('--no-binarise', dest='binarise', action='store_false',
                   help='Do not binarise the volume on --pml-label before '
                        'marching cubes.  Use for bit-exact reproduction of '
                        'paper-specific numbers from already-binary '
                        'segmentations; has no effect on .obj input.')
    p.set_defaults(binarise=True)
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
    mesh = load_mesh(args.segmentation, args.pml_label, args.isovalue,
                     binarise=args.binarise)
    # Mesh is kept in its native (mesh-pixel) space for ray-casting;
    # nanometre scaling is applied only for rendering.
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
           title=args.title or args.segmentation.stem,
           mesh_pixel_size_nm=args.mesh_pixel_size_nm)

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
