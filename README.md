# pml-body-visual-proteomics

Classify template-matched particles as inside vs outside a segmented
PML nuclear body, and render a 3D visual-proteomics overview.

Accompanies:

> Pražák V, Harley I, Falckenhayn J, Boutell C, Thomason PA, Davis BG,
> Kaufmann R, Carter SD. *In situ architecture of PML nuclear bodies reveals a
> porous, chromatin-permissive interior.* (submitted to Nature Cell Biology, 2026)

## What it does

Given one PML body segmentation (Amira TIFF label file or pre-exported
`.obj` mesh) and one or more particle STAR files, the script:

1. Loads the segmentation and extracts a triangulated surface via
   `skimage.measure.marching_cubes`, transposing axes from `(Z, Y, X)`
   to `(X, Y, Z)` to match the RELION XYZ coordinate order of the
   particle STAR files. A pre-exported `.obj` mesh can be passed
   directly instead.
2. Builds a `trimesh.Trimesh` object from the vertices and faces, used as
   exported without repair.
3. Loads particle coordinates from each STAR file, scales them from
   unbinned STAR pixels into the mesh's coordinate space, and then
   into nanometres.
4. Classifies every particle as inside or outside the PML body by
   ray-casting with `Trimesh.contains`.
5. Writes per-type inside/outside counts, per-particle
   classifications, a JSON summary, and a 3D rendering.

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `scikit-image`, `tifffile`, `trimesh`, `matplotlib`; the paper scripts in `enrichment/`, `figures/` and `spatial_checks/` also use `pandas`, `scikit-learn`, `mrcfile` and `Pillow`.

## Usage

```bash
python pml_body_visual_proteomics.py \
    --segmentation path/to/pml_labels.tif \
    --particles p28:path/to/p28.star \
                TRiC_closed:path/to/tric_closed.star \
                TRiC_open:path/to/tric_open.star \
    --pml-label 2 \
    --unbinned-to-mesh-scale 4 \
    --mesh-pixel-size-nm 0.992 \
    --output-dir results/
```

### Required arguments

| Flag | Description |
|------|-------------|
| `--segmentation` | Amira TIFF label file or pre-exported `.obj` mesh. |
| `--particles` | One or more `LABEL:PATH` entries. `LABEL` is used in figure legend and output files. |
| `--unbinned-to-mesh-scale` | Division factor from unbinned STAR pixels to mesh-space pixels (e.g. `4` if mesh is bin-4). |
| `--mesh-pixel-size-nm` | Pixel size of the mesh coordinate space in nm (e.g. `0.992` for bin-4 at 2.48 Å unbinned). |
| `--output-dir` | Directory for outputs (created if absent). |

### Optional arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--pml-label` | `2` | Integer label value marking the PML body in the TIFF. |
| `--isovalue` | `0.5` | Marching-cubes isovalue. |
| `--title` | segmentation name | Figure title override. |

Run `python pml_body_visual_proteomics.py --help` for the full list.

## Input requirements

- **Segmentation**: either a `.tif`/`.tiff` label file (multi-label is
  supported; only the value given to `--pml-label` is treated as PML),
  or a `.obj` mesh file.
- **Particle STAR files**: standard RELION/PyTOM format with
  `_rlnCoordinateX`, `_rlnCoordinateY`, `_rlnCoordinateZ` columns in
  unbinned pixel coordinates.

## Output files

Written to `--output-dir`:

| File | Contents |
|------|----------|
| `inside_outside_counts.csv` | One row per particle type with total / inside / outside counts and percent inside. |
| `classifications.csv` | One row per particle with `(x, y, z)` in nm and the boolean `inside` classification. |
| `summary.json` | All parameters used plus the per-type counts. |
| `rendering.png` | 3D figure with the PML body as a semi-transparent mesh and particles as spheres. Large markers denote inside, small markers outside. |

## Importable API

The underlying functions can be called from another Python program:

```python
from pml_body_visual_proteomics import (
    load_mesh,
    load_star_coords,
    classify_particles,
    render,
)
```

## Licence

MIT (see `LICENSE`).

## Analyses used for the submitted manuscript (October 2026)

`enrichment/`
- `pml_enrichment.py`: volume-normalised enrichment of each particle type inside
  the segmented body (observed inside fraction divided by the fraction of the
  searched volume the body occupies; denominators A, B and C; Wilson intervals,
  exact binomial, Fisher and Cochran-Mantel-Haenszel tests). Inside/outside by
  ray casting against the binary (label == 2) marching-cubes mesh. Source of
  Fig. 5d,e and the numbers in the Results.
- `boundary_sensitivity.py`: enrichment after eroding or dilating the body by
  5, 10 and 20 nm (voxel lookup). Source of Supplementary Fig. 10c.
- `tric_closed_vs_open_enrichment.py`: enrichment of closed and open TRiC
  separately.

`figures/`
- `make_fig5.py`: Fig. 5.
- `make_supp_fig10.py`: Supplementary Fig. 10.
- `make_supp_fig3.py`, `make_supp_fig4.py`, `make_supp_fig6c.py`:
  Supplementary Figs 3, 4 and 6c (`segmentation_classification_summary_final.csv`
  holds the per-tomogram PA28 and TRiC inside/outside counts read by
  `make_supp_fig4.py`; nucleosome and trinucleosome counts come from
  `enrichment/pml_enrichment.py`).
- `make_supp_fig10.py` also reads `fibre_after_blob_removal_isonet.json`, the
  output of the companion repository pml-body-network-analysis.

`spatial_checks/`
- `mixed_cluster_permutation.py`: random-labelling null for the fraction of
  mixed nucleosome/trinucleosome DBSCAN clusters (2,000 label permutations per
  tomogram with the clusters fixed).
- `ripley_r_at_peak.py`: radius at which Ripley's L(r) is maximal per tomogram.

`make_supp_fig6c.py`, `mixed_cluster_permutation.py` and `ripley_r_at_peak.py`
import helper functions from the original analysis scripts
(`nearest_neighbor_analysis.py`, `dbscan_combined_nuc_trinuc.py`,
`ppa_ripleys_L_whole.py`), which are available from the corresponding author.

Paths to the STAR files, segmentations and output folders are hard-coded at the
top of each script and must be edited for another machine. Particle STAR files
and segmentations are not part of this repository. Subtomogram averages are in
the EMDB (EMD-57924, EMD-58330, EMD-57913, EMD-57970, EMD-57971); tilt-series
and tomograms will be deposited in EMPIAR, and particle coordinates and
segmentations are available from the corresponding author.
