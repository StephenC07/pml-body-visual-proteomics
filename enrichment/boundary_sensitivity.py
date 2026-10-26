"""Distance of particles to the PML boundary and sensitivity of enrichment to boundary placement.
Signed distance from the label-volume distance field: positive inside the body, negative outside (nm). The boundary is then shifted by
d nm (dilated for d>0, eroded for d<0); particles with signed distance > -d count as inside and the
expected fraction is recomputed from the shifted label volume within the same slab."""
import numpy as np, pandas as pd
from scipy.ndimage import distance_transform_edt
exec(open('pml_enrichment.py').read().split('records, geometry = [], []')[0])   # loaders/constants

SHIFTS = [-20, -10, -5, 0, 5, 10, 20]
rows, dist_rows = [], []
for tomo in TOMOS:
    mesh, vol = load_segmentation(tomo)
    parts = {t: load_star(f'{BASE}/Files_for_ultimate_comparision/{d}/{tomo}.star') / DIV for t, d in TYPES.items()}
    parts['tric'] = np.vstack([parts.pop('tric_closed'), parts.pop('tric_open')])
    pooled = np.vstack(list(parts.values()))
    zlo, zhi = int(np.floor(pooled[:, 2].min())), int(np.ceil(pooled[:, 2].max()))
    slab = vol[:, :, zlo:zhi + 1]
    # signed distance field of the label volume within the slab (nm)
    din = distance_transform_edt(slab) * PX; dout = distance_transform_edt(~slab) * PX
    sdf = np.where(slab, din, -dout)
    for t, xyz in parts.items():
        ijk = np.round(xyz).astype(int); ijk[:, 2] -= zlo
        ijk = np.clip(ijk, 0, np.array(sdf.shape) - 1)
        sd = sdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]]          # label-volume signed distance, positive inside
        for s in sd: dist_rows.append(dict(tomo=tomo, type=t, sd_nm=float(s)))
        for d in SHIFTS:
            inside = sd > -d
            frac = float((sdf > -d).mean())
            rows.append(dict(tomo=tomo, type=t, shift_nm=d, n=len(xyz), inside=int(inside.sum()), exp_frac=frac))
    print(tomo, 'done', flush=True)

dist = pd.DataFrame(dist_rows); df = pd.DataFrame(rows)
dist.to_csv('particle_boundary_distance.csv', index=False); df.to_csv('boundary_sensitivity.csv', index=False)
with open('boundary_sensitivity.txt', 'w') as f:
    f.write('Signed distance to PML boundary (nm; + inside, - outside), pooled over 5 tomograms\n')
    for t in ['nucleosome', 'trinucleosome', 'tric', 'pa28']:
        s = dist[dist.type == t].sd_nm.values; ins = s > 0
        f.write(f'{t:14s} n={len(s):4d} inside={ins.sum():3d} | of inside: within 10 nm of boundary {(s[ins] < 10).sum():3d}, within 20 nm {(s[ins] < 20).sum():3d}, median depth {np.median(s[ins]) if ins.any() else np.nan:5.1f} nm'
                f' | of outside: within 10 nm {(s[~ins] > -10).sum():3d}, within 20 nm {(s[~ins] > -20).sum():3d}, within 50 nm {(s[~ins] > -50).sum():3d}\n')
    f.write('\nEnrichment (observed/expected) versus boundary shift (negative = eroded, positive = dilated)\n')
    f.write('shift_nm  ' + '  '.join(f'{t:>22s}' for t in ['nucleosome', 'trinucleosome', 'tric', 'pa28']) + '\n')
    for d in SHIFTS:
        line = f'{d:+6d}    '
        for t in ['nucleosome', 'trinucleosome', 'tric', 'pa28']:
            g = df[(df.type == t) & (df.shift_nm == d)]; n, k = g.n.sum(), g.inside.sum(); ef = (g.n * g.exp_frac).sum() / n
            line += f'  {k:3d}/{n:3d} {k/n*100:4.1f}% exp {ef*100:4.1f}% e={k/n/ef:4.2f}'
        f.write(line + '\n')
    f.write('\nTRiC inside counts per tomogram vs shift\n')
    f.write(df[df.type == 'tric'].pivot(index='tomo', columns='shift_nm', values='inside').to_string() + '\n')
print(open('boundary_sensitivity.txt').read())
