"""
3D figure of the first long-horizon set Phi (long-horizon set v0) of the selected scenario.

Long-horizon set v0 is computed exactly as in Verifier.solve_long(0, ...): window placed around
X_INIT, task UNTIL(safe_region, target_loc) with static obstacles only. The 4D state
(x, y, yaw, vel) is projected onto (x, y) by the minimum over yaw and vel (the same
projection the verifier saves for the animation), or sliced at a given (yaw, vel), and the
zero level set is drawn as an isosurface in (x, y, t), as impl.plot(..., axes=('x','y','t'))
does in the notebooks. Static obstacles and the target are drawn as boxes over the horizon.

Usage:
    python3 plots/plot_long_phi.py                        # min over (yaw, vel)
    python3 plots/plot_long_phi.py --proj slice --yaw 0 --vel 0
    python3 plots/plot_long_phi.py --recompute            # ignore the cached set
    SCENARIO=corridor python3 plots/plot_long_phi.py

Outputs in results/paper/: long0_<scenario>.npz (cache), long0_<scenario>_<proj>.html,
and .pdf/.png if kaleido is installed (pip install kaleido).
"""
import argparse
import os
import sys
import time

import jax
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import plotly.graph_objects as go

# Run from the repository root (python3 plots/<script>.py): make the simulation modules
# (config, verifier, obstacles, ...) importable from this subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Optional: a local pyspect checkout (otherwise the pip-installed package is used)
PYSPECT_SRC = os.environ.get("PYSPECT_SRC")
if PYSPECT_SRC and PYSPECT_SRC not in sys.path:
    sys.path.append(PYSPECT_SRC)

from pyspect import *  # noqa: E402,F403

import config  # noqa: E402
from verifier import Verifier  # noqa: E402

OUT_DIR = os.path.join("results", "paper")


# ------------------------------------------------------------------
# Long-horizon set v0
# ------------------------------------------------------------------

def compute_long0(cfg_registry):
    """Same computation as Verifier.solve_long(0, 0, X_INIT), without its side effects."""
    Verifier.load_from_file = True                           # do not wipe ./sets
    Verifier.results_dir = os.path.join(OUT_DIR, "_tmp")     # do not overwrite results/crop_*.csv
    ver = Verifier(cfg_registry, None, None)

    state = cfg_registry['config']['x_init']
    window = ver.place_window(state)
    ver._move_window(ver.impl_long, window)
    env = ver.build_env(window)

    t0 = time.time()
    out1 = np.asarray(TLT.construct(UNTIL(env['safe_region'], env['target_loc'])).realize(ver.impl_long))
    print(f"long-horizon set v0 solved in {time.time() - t0:.1f} s, shape {out1.shape}")

    x, y, yaw, vel = [np.asarray(c) for c in ver.impl_long.grid.coordinate_vectors]
    return dict(
        out1=out1.astype(np.float32),
        t=np.asarray(ver.impl_long.timeline), x=x, y=y, yaw=yaw, vel=vel,
        window=np.asarray(window), target_rect=np.asarray(env['target_rect']),
        target_is_goal=env['target_is_goal'], x_init=np.asarray(state),
    )


def load_or_compute(scenario, recompute):
    path = os.path.join(OUT_DIR, f"long0_{scenario}.npz")
    if os.path.exists(path) and not recompute:
        print(f"Loading {path}")
        return dict(np.load(path))
    registry = {}
    config.run(registry)
    data = compute_long0(registry)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"Saved {path}")
    return data


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def box_mesh(x0, x1, y0, y1, z0, z1, color, opacity, name, showlegend=True):
    """Closed box as a Mesh3d (8 vertices, 12 triangles)."""
    vx = [x0, x1, x1, x0, x0, x1, x1, x0]
    vy = [y0, y0, y1, y1, y0, y0, y1, y1]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]
    i = [0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 1, 1]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 2, 6]
    k = [2, 3, 6, 7, 5, 4, 6, 7, 7, 4, 6, 5]
    return go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k, color=color, opacity=opacity,
                     flatshading=True, name=name, showlegend=showlegend, hoverinfo='skip',
                     lighting=dict(ambient=0.8, diffuse=0.4, specular=0.0))


def box_edges(x0, x1, y0, y1, z0, z1, color, width=2):
    """Wireframe of a box as one Scatter3d (None separates the segments)."""
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    xs, ys, zs = [], [], []
    for z in (z0, z1):                                   # bottom and top rectangles
        for (a, b) in c + [c[0]]:
            xs.append(a); ys.append(b); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    for (a, b) in c:                                     # vertical edges
        xs += [a, a, None]; ys += [b, b, None]; zs += [z0, z1, None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', line=dict(color=color, width=width),
                        showlegend=False, hoverinfo='skip')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--proj', choices=('min', 'slice'), default='min',
                   help="min: min over (yaw, vel); slice: Phi at fixed --yaw, --vel")
    p.add_argument('--yaw', type=float, default=None, help="yaw [rad] for --proj slice (default: X_INIT)")
    p.add_argument('--vel', type=float, default=None, help="vel [m/s] for --proj slice (default: X_INIT)")
    p.add_argument('--recompute', action='store_true', help="recompute long-horizon set v0 even if cached")
    p.add_argument('--upsample', type=int, default=3,
                   help="linear upsampling of Phi(t, x, y) before the isosurface (smoother surface)")
    p.add_argument('--tick-size', type=int, default=17, help="font size of the axis numbers")
    p.add_argument('--title-size', type=int, default=60, help="font size of the axis titles x, y, t")
    p.add_argument('--width', type=int, default=1100)
    p.add_argument('--height', type=int, default=650)
    args = p.parse_args()

    scenario = config.SCENARIO
    d = load_or_compute(scenario, args.recompute)
    out1, t, x, y = d['out1'], d['t'], d['x'], d['y']
    (wx0, wx1), (wy0, wy1) = d['window']
    T = float(t[-1])

    # --- 4D -> (t, x, y) ----------------------------------------------------
    if args.proj == 'min':
        phi = out1.min(axis=(3, 4))
        label = r"min over (yaw, vel)"
    else:
        yaw0 = d['x_init'][2] if args.yaw is None else args.yaw
        vel0 = d['x_init'][3] if args.vel is None else args.vel
        iyaw = int(np.argmin(np.abs(np.angle(np.exp(1j * (d['yaw'] - yaw0))))))
        ivel = int(np.argmin(np.abs(d['vel'] - vel0)))
        phi = out1[:, :, :, iyaw, ivel]
        label = f"yaw = {d['yaw'][iyaw]:.2f} rad, vel = {d['vel'][ivel]:.2f} m/s"
    print(f"Phi(t, x, y): {label}, range [{phi.min():.2f}, {phi.max():.2f}]")

    # --- smoother surface: trilinear upsampling of Phi (display only) ---------
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
        name=r'Φ = 0', showlegend=True,
    ))

    # static obstacles (physical footprint, clipped to the window), extruded over the horizon
    first = True
    for (ox, oy, ow, oh) in config.OBSTACLES:
        bx0, bx1 = max(ox, wx0), min(ox + ow, wx1)
        by0, by1 = max(oy, wy0), min(oy + oh, wy1)
        if bx0 >= bx1 or by0 >= by1:
            continue
        fig.add_trace(box_mesh(bx0, bx1, by0, by1, 0, T, '#7f7f7f', 0.45, 'static obstacle', first))
        fig.add_trace(box_edges(bx0, bx1, by0, by1, 0, T, '#555555'))
        first = False


    # window footprint at t = 0 and initial state
    fig.add_trace(box_edges(wx0, wx1, wy0, wy1, 0, 0, '#1f92e4', 20))
    xi = d['x_init']
    fig.add_trace(go.Scatter3d(x=[xi[0]], y=[xi[1]], z=[0.0], mode='markers',
                               marker=dict(size=7, color='#ff7f0e', line=dict(color='black', width=1)),
                               name='initial state', showlegend=False))

    # aspect ratio: true x:y ratio for the road, time axis scaled to be readable
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
            xaxis=dict(axis, linecolor='#1f92e4', title=dict(text='<i>x</i> [m]', font=tfont), range=[wx0, wx1]),
            yaxis=dict(axis, linecolor='#1f92e4', title=dict(text='<i>y</i> [m]', font=tfont), tickvals=[0, 2, 4, 6], range=[wy0, wy1]),
            zaxis=dict(axis, title=dict(text='<i>t</i> [s]', font=tfont), range=[0, T]),
            aspectmode='manual', aspectratio=ar,
            camera=dict(eye=dict(x=-0.9, y=-1.9, z=1.6), center=dict(x=0.05, y=0, z=-0.3)),
        ),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(OUT_DIR, f"long0_{scenario}_{args.proj}")
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
