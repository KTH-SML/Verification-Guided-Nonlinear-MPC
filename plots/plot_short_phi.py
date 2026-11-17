"""
3D figure of a short-horizon set Phi_k derived from the first long-horizon set (long-horizon set v0), with the
moving obstacles that cross the road during its window.

Short-horizon set k covers [k*T_L, k*T_L + HJ_PREDICTION] (absolute simulation time) and is computed
as in Verifier.run(): safe region = static safe region minus the moving obstacles of the
window, terminal target = long-horizon set v0 sliced at the end of the window (united with the long-horizon
target). It is solved on the whole long-horizon window (crop 'none': the crop methods only restrict
the PDE to a box that contains the set). Short-horizon sets 0 and 1 use long-horizon set v0.

As in plot_long_phi.py, the 4D state is projected onto (x, y) by the minimum over yaw and
vel (or sliced), and the zero level set is drawn in (x, y, t). The moving obstacles are drawn
with their physical footprint as solids in (x, y, t) (slanted: they move along y).

Usage:
    python3 plots/plot_short_phi.py              # short-horizon set 1, window [1, 3] s (crossing car 1)
    python3 plots/plot_short_phi.py --k 0        # short-horizon set 0, window [0, 2] s
    python3 plots/plot_short_phi.py --proj slice --yaw 0 --vel 5

Outputs in results/paper/: short<k>_<scenario>.npz (cache), short<k>_<scenario>_<proj>.html
and .pdf/.png if kaleido is installed. Needs results/paper/long0_<scenario>.npz (computed
by plot_long_phi.py, or here if missing).
"""
import argparse
import os
import time

import numpy as np
import plotly.graph_objects as go

from plot_long_phi import OUT_DIR, box_edges, box_mesh, load_or_compute  # sets up jax and pyspect
from pyspect import *  # noqa: F403

import config
from obstacles import crossing_events
from verifier import Verifier


def compute_short(k, long0):
    """Short-horizon set k on the window of long-horizon set v0 (same construction as Verifier.run)."""
    registry = {}
    config.run(registry)
    Verifier.load_from_file = True                           # do not wipe ./sets
    Verifier.results_dir = os.path.join(OUT_DIR, "_tmp")     # do not overwrite results/crop_*.csv
    ver = Verifier(registry, None, None)

    # activate long-horizon set v0 (as activate_long, from the cached solution)
    window = tuple(tuple(float(v) for v in w) for w in long0['window'])
    for impl in (ver.impl_long, ver.impl, ver.impl_short):
        ver._move_window(impl, window)
    ver.long_version, ver.long_t0, ver.window = 0, 0.0, window
    ver.env = ver.build_env(window)
    ver.target = ver.env['target_loc']
    ver.out1 = long0['out1']

    t_start = k * ver.SHORT_PERIOD
    t_end = t_start + ver.T_PRED
    if t_end > ver.HORIZON + 1e-9:
        raise ValueError(f"short-horizon window [{t_start}, {t_end}] s ends past the horizon of long-horizon set v0")

    moving = ver.build_moving_object_short(t_start, ver.T_PRED)
    safe = Inter(ver.env['safe_region'], Compl(moving))
    target = ver.get_short_terminal_target(ver.out1, t_end)
    task = UNTIL(Set(safe(ver.impl_short)), Set(target(ver.impl_short)))

    t0 = time.time()
    out2 = np.asarray(TLT.construct(task).realize(ver.impl_short))
    print(f"short-horizon set {k} [{t_start:.1f}, {t_end:.1f}] s solved in {time.time() - t0:.1f} s, shape {out2.shape}")

    x, y, yaw, vel = [np.asarray(c) for c in ver.impl_short.grid.coordinate_vectors]
    return dict(out2=out2.astype(np.float32), t=t_start + np.asarray(ver.impl_short.timeline),
                x=x, y=y, yaw=yaw, vel=vel, window=np.asarray(window), x_init=long0['x_init'])


def load_or_compute_short(scenario, k, recompute):
    path = os.path.join(OUT_DIR, f"short{k}_{scenario}.npz")
    if os.path.exists(path) and not recompute:
        print(f"Loading {path}")
        return dict(np.load(path))
    data = compute_short(k, load_or_compute(scenario, False))
    np.savez_compressed(path, **data)
    print(f"Saved {path}")
    return data


def moving_obstacle_solids(t0, t1, window, n=80):
    """
    Region occupied by every moving obstacle in (x, y, t) over [t0, t1], clipped to its band
    and to the window: a convex polytope per crossing, given by points on its boundary
    (Mesh3d with alphahull=0 draws their convex hull). Returns [(name, xs, ys, ts)].
    """
    (wx0, wx1), (wy0, wy1) = window
    solids = []
    for p in config.DYN_OBSTACLES:
        x_lo = max(p['center_x'] - p['width'] / 2, wx0)
        x_hi = min(p['center_x'] + p['width'] / 2, wx1)
        if x_lo >= x_hi:
            continue
        y_bot, y_top = max(p['between_y'][0], wy0), min(p['between_y'][1], wy1)
        for i, v, y_lo0, y_hi0, t_enter, t_exit in crossing_events(p):
            ta, tb = max(t0, t_enter), min(t1, t_exit)
            if ta >= tb:
                continue
            xs, ys, ts = [], [], []
            for t in np.linspace(ta, tb, n):
                ylo = max(y_lo0 + v * (t - t_enter), y_bot)
                yhi = min(y_hi0 + v * (t - t_enter), y_top)
                if ylo >= yhi:
                    continue
                for (a, b) in ((x_lo, ylo), (x_hi, ylo), (x_hi, yhi), (x_lo, yhi)):
                    xs.append(a); ys.append(b); ts.append(t)
            if xs:
                solids.append((p['name'], np.array(xs), np.array(ys), np.array(ts)))
    return solids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--k', type=int, default=1, help="short-horizon set index (window [k*T_L, k*T_L + T_PRED])")
    p.add_argument('--proj', choices=('min', 'slice'), default='min',
                   help="min: min over (yaw, vel); slice: Phi at fixed --yaw, --vel")
    p.add_argument('--yaw', type=float, default=None, help="yaw [rad] for --proj slice (default: X_INIT)")
    p.add_argument('--vel', type=float, default=None, help="vel [m/s] for --proj slice (default: X_INIT)")
    p.add_argument('--recompute', action='store_true', help="recompute the short-horizon set even if cached")
    p.add_argument('--eye', type=float, nargs=3, default=(-0.9, -1.9, 1.6), metavar=('X', 'Y', 'Z'),
                   help="camera position (plotly scene units); default: same view as plot_long_phi.py")
    p.add_argument('--out', default=None, help="output base name (default: short<k>_<scenario>_<proj>)")
    p.add_argument('--dynamic', action=argparse.BooleanOptionalAction, default=True,
                   help="draw the moving obstacles of the window (--no-dynamic to hide them)")
    p.add_argument('--upsample', type=int, default=3,
                   help="linear upsampling of Phi(t, x, y) before the isosurface (smoother surface)")
    p.add_argument('--tick-size', type=int, default=17, help="font size of the axis numbers")
    p.add_argument('--title-size', type=int, default=60, help="font size of the axis titles x, y, t")
    p.add_argument('--width', type=int, default=1100)
    p.add_argument('--height', type=int, default=650)
    args = p.parse_args()

    scenario = config.SCENARIO
    d = load_or_compute_short(scenario, args.k, args.recompute)
    out2, t, x, y = d['out2'], d['t'], d['x'], d['y']
    (wx0, wx1), (wy0, wy1) = d['window']
    t0, t1 = float(t[0]), float(t[-1])

    # --- 4D -> (t, x, y) ----------------------------------------------------
    if args.proj == 'min':
        phi = out2.min(axis=(3, 4))
        label = "min over (yaw, vel)"
    else:
        yaw0 = d['x_init'][2] if args.yaw is None else args.yaw
        vel0 = d['x_init'][3] if args.vel is None else args.vel
        iyaw = int(np.argmin(np.abs(np.angle(np.exp(1j * (d['yaw'] - yaw0))))))
        ivel = int(np.argmin(np.abs(d['vel'] - vel0)))
        phi = out2[:, :, :, iyaw, ivel]
        label = f"yaw = {d['yaw'][iyaw]:.2f} rad, vel = {d['vel'][ivel]:.2f} m/s"
    print(f"Phi_{args.k}(t, x, y): {label}, range [{phi.min():.2f}, {phi.max():.2f}]")

    if args.upsample > 1:
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((t, x, y), phi)
        t, x, y = (np.linspace(c[0], c[-1], args.upsample * (len(c) - 1) + 1) for c in (t, x, y))
        phi = interp(np.stack(np.meshgrid(t, x, y, indexing='ij'), axis=-1))

    # --- figure -------------------------------------------------------------
    X, Y, Tt = np.meshgrid(x, y, t, indexing='ij')
    fig = go.Figure()
    fig.add_trace(go.Isosurface(
        x=X.ravel(), y=Y.ravel(), z=Tt.ravel(),
        value=phi.transpose(1, 2, 0).ravel(),
        isomin=0.0, isomax=0.0, surface_count=1,
        colorscale='Greens', showscale=False, opacity=0.9,
        caps=dict(x_show=False, y_show=False, z_show=False),
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15, roughness=0.6),
        name='Φ = 0', showlegend=True,
    ))

    # static obstacles (physical footprint, clipped to the window), extruded over the window
    first = True
    for (ox, oy, ow, oh) in config.OBSTACLES:
        bx0, bx1 = max(ox, wx0), min(ox + ow, wx1)
        by0, by1 = max(oy, wy0), min(oy + oh, wy1)
        if bx0 >= bx1 or by0 >= by1:
            continue
        fig.add_trace(box_mesh(bx0, bx1, by0, by1, t0, t1, '#7f7f7f', 0.45, 'static obstacle', first))
        fig.add_trace(box_edges(bx0, bx1, by0, by1, t0, t1, '#555555'))
        first = False

    # moving obstacles (physical footprint) during the window: only with --dynamic
    first = True
    for name, xs, ys, ts in (moving_obstacle_solids(t0, t1, d['window']) if args.dynamic else []):
        print(f"moving obstacle in the window: {name}, t in [{ts.min():.2f}, {ts.max():.2f}] s")
        fig.add_trace(go.Mesh3d(x=xs, y=ys, z=ts, alphahull=0, color='#d62728', opacity=0.6,
                                flatshading=True, name='dynamic obstacle', showlegend=first,
                                hoverinfo='skip', lighting=dict(ambient=0.8, diffuse=0.4, specular=0.0)))
        first = False

    Lx, Ly = wx1 - wx0, wy1 - wy0
    ar = dict(x=2.6, y=max(0.75, 2.6 * Ly / Lx), z=1.0)
    font = dict(family='Times New Roman, serif', size=16, color='black')
    # large axis titles, thick black axis lines: x, y, t clearly identified
    axis = dict(showbackground=True, backgroundcolor='rgb(245,245,245)', gridcolor='rgb(210,210,210)',
                zerolinecolor='rgb(210,210,210)', showline=True, linecolor='black', linewidth=10,
                ticks='outside', ticklen=8, tickwidth=2, tickcolor='black',
                tickfont=dict(size=args.tick_size))
    tfont = dict(font, size=args.title_size)
    fig.update_layout(
        width=args.width, height=args.height, font=font,
        margin=dict(l=50, r=0, t=10, b=40), paper_bgcolor='white',
        showlegend=False,
        scene=dict(
            xaxis=dict(axis, linecolor='#1f92e4', linewidth=20, title=dict(text='<i>x</i> [m]', font=tfont), range=[wx0, wx1]),
            yaxis=dict(axis, linecolor='#1f92e4', linewidth=20, title=dict(text='<i>y</i> [m]', font=tfont), tickvals=[0, 2, 4, 6], range=[wy0, wy1]),
            zaxis=dict(axis, title=dict(text='<i>t</i> [s]', font=tfont), range=[0, config.HJ_HORIZON]),   # same as the long-horizon figure
            aspectmode='manual', aspectratio=ar,
            camera=dict(eye=dict(zip('xyz', args.eye)), center=dict(x=0.05, y=0, z=-0.3)),
        ),
    )

    base = args.out or os.path.join(OUT_DIR, f"short{args.k}_{scenario}_{args.proj}")
    fig.write_html(base + ".html", include_plotlyjs='cdn')
    print(f"Saved {base}.html")
    try:
        fig.write_image(base + ".pdf")
        fig.write_image(base + ".png", scale=3)
        print(f"Saved {base}.pdf, {base}.png")
    except Exception as e:  # kaleido missing
        print(f"No static export ({type(e).__name__}: {e}): pip install kaleido, "
              f"or use the camera button in the HTML for a PNG")


if __name__ == '__main__':
    main()
