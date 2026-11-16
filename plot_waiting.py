"""
Paper figure: the vehicle waits in front of a long blocking obstacle (e.g. a long tram).

Four snapshots of the road around the stopped vehicle, each showing only the zero level
of the short-horizon set in use by the MPC (V <= 0, min over yaw and vel):

    (a), (b)  the deadline of the active long-horizon iteration approaches: the set shrinks
    (c)       the next long-horizon iteration becomes active: the set is large again
    (d)       the set shrinks again under the new iteration

The previous snapshot's outline is drawn thin and dashed in each panel, so the change is
visible directly.

Input: results/run_data.pkl (Animation.save_run_data) and the set files it points to
(sets/data_<k>.npy), so run it from the directory the simulation ran in, or pass --root.

Usage:
    python3 plot_waiting.py --list                # per-step table while stopped (choose times)
    python3 plot_waiting.py                       # times picked automatically
    python3 plot_waiting.py --times 14.2 15.5 15.6 18.0 --xlim 55 95
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import zoom

from obstacles import obstacle_rects
from plot_paper import C_SHORT, INK, INK2, GRID, SYM_LONG, SYM_SHORT, tint, _save  # paper style

C_ROAD, C_MARK, C_STATIC, C_DYN = "#ecebe7", "#ffffff", "#9a9893", "#d62728"


class Run:
    """Saved run: trajectory, set used at every MPC step, set files."""

    def __init__(self, path, root):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.cfg = d['config']
        self.path = np.asarray(d['real_path'], float)          # t, x, y, yaw, vel
        self.used = list(d['controller']['used_sets'])
        self.local = {e['set_idx']: e for e in d['local_timeline']}
        self.globals = {g['version']: g for g in d['global_timeline']}
        self.n = min(len(self.path), len(self.used))
        self.root = root
        self.dt_hj = self.cfg['verifier']['timestep']
        self.horizon = self.cfg['verifier']['horizon']
        self.nx, self.ny = [a['points'] for a in self.cfg['verifier']['axes'] if a['name'] in ('x', 'y')]
        self._xy = {}

    def t(self, k):
        return self.path[k, 0]

    def step(self, t):
        return int(np.argmin(np.abs(self.path[:self.n, 0] - t)))

    def set_entry(self, k):
        return self.local.get(self.used[k])

    def gv(self, k):
        e = self.set_entry(k)
        return None if e is None else e.get('global_version')

    def xy(self, sid):
        """(min over yaw, vel) of short-horizon set sid: (NT, nx, ny), and its x, y grids."""
        if sid not in self._xy:
            e = self.local[sid]
            p = e['path'] if os.path.isabs(e['path']) else os.path.join(self.root, e['path'])
            Z = np.load(p, mmap_mode="r")
            Z = np.asarray(Z).min(axis=(3, 4))
            (wx0, wx1), (wy0, wy1) = e['window']
            self._xy[sid] = (Z, np.linspace(wx0, wx1, Z.shape[1]), np.linspace(wy0, wy1, Z.shape[2]))
        return self._xy[sid]

    def slice(self, k):
        """Zero-level data of the set used at step k, at the time of step k."""
        sid = self.used[k]
        Z, gx, gy = self.xy(sid)
        it = int(np.clip(round((self.t(k) - self.local[sid]['interval'][0]) / self.dt_hj), 0, Z.shape[0] - 1))
        return Z[it], gx, gy

    def deadline(self, k):
        g = self.globals.get(self.gv(k))
        return None if g is None else g['t0'] + self.horizon


def stopped_mask(run, v_tol):
    return np.abs(run.path[:run.n, 4]) < v_tol


def auto_times(run, v_tol):
    """Around the first long-horizon switch with the vehicle at rest during the whole
    iteration before it: (mid, end) of that iteration, (start, end) of the next."""
    stop = stopped_mask(run, v_tol)
    gv = [run.gv(k) for k in range(run.n)]
    seg = [0] + [k for k in range(1, run.n) if gv[k] != gv[k - 1]] + [run.n]
    for i in range(1, len(seg) - 2):
        a, b, c = seg[i], seg[i + 1], seg[i + 2]      # [a, b) iteration j, [b, c) iteration j+1
        if stop[a:c].all() and gv[a] is not None and gv[b] is not None:
            return [run.t((a + b - 1) // 2), run.t(b - 1), run.t(b), run.t(c - 1)]
    raise SystemExit("no long-horizon switch found with the vehicle at rest before and after it: "
                     "check with --list and pass --times")


def print_table(run, v_tol, cell):
    stop = stopped_mask(run, v_tol)
    print(f"{'t':>6} {'vel':>5} {'set':>5} {'long':>5} {'deadline':>8} {'area m2':>8} {'V(x)':>7}")
    for k in np.flatnonzero(stop):
        Z, gx, gy = run.slice(k)
        x, y = run.path[k, 1:3]
        ix, iy = np.argmin(np.abs(gx - x)), np.argmin(np.abs(gy - y))
        dl = run.deadline(k)
        print(f"{run.t(k):6.2f} {run.path[k, 4]:5.2f} {SYM_SHORT}{run.used[k]:<4d} {SYM_LONG}{run.gv(k)!s:<4} "
              f"{(dl if dl is not None else np.nan):8.2f} {(Z <= 0).sum() * cell:8.1f} {Z[ix, iy]:+7.3f}")


def smooth(Z, gx, gy, f):
    """Cubic upsampling (same values at the grid nodes, smoother zero level between them)."""
    if f <= 1:
        return Z, gx, gy
    Zs = zoom(Z, f, order=3)
    return Zs, np.linspace(gx[0], gx[-1], Zs.shape[0]), np.linspace(gy[0], gy[-1], Zs.shape[1])


def draw_road(ax, cfg, xlim, t):
    X0, Y0, YW = cfg['X0'], cfg['Y0'], cfg['YW']
    ax.add_patch(mpatches.Rectangle((xlim[0], Y0), xlim[1] - xlim[0], YW, fc=C_ROAD, ec="none", zorder=0))
    for yb in (Y0, Y0 + YW):
        ax.plot(xlim, [yb, yb], color=INK2, lw=0.8, zorder=1)
    ax.plot(xlim, [Y0 + YW / 2] * 2, color=C_MARK, lw=1.0, ls=(0, (6, 4)), zorder=1)
    for ox, oy, ow, oh in cfg['obstacles']:
        if oy + oh <= Y0 or oy >= Y0 + YW:          # curbs, outside the road
            continue
        ax.add_patch(mpatches.Rectangle((ox, oy), ow, oh, fc=C_STATIC, ec="none", zorder=4))
    for dyn in cfg.get('dynamic_obstacles', []):
        for ox, oy, ow, oh in obstacle_rects(t, dyn):
            ax.add_patch(mpatches.Rectangle((ox, oy), ow, oh, fc=C_DYN, ec="none", zorder=4))


def plot_waiting(run, times, out, xlim=None, width=3.5, fs=7, upsample=4, cols=2):
    cfg = run.cfg
    ks = [run.step(t) for t in times]
    if xlim is None:
        xs = run.path[ks, 1]
        xlim = (xs.min() - 22.0, xs.max() + 10.0)
    Y0, YW = cfg['Y0'], cfg['YW']
    rows = int(np.ceil(len(ks) / cols))
    panel_w = width * 0.95 / cols
    panel_h = panel_w * (YW + 0.6) / (xlim[1] - xlim[0])         # equal aspect
    fig, axes = plt.subplots(rows, cols, figsize=(width, rows * (panel_h + 0.30) + 0.20),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(ks):]:
        ax.set_visible(False)

    for i, (ax, k) in enumerate(zip(axes, ks)):
        t, x, y, yaw = run.path[k, :4]
        draw_road(ax, cfg, xlim, t)

        Z, gx, gy = smooth(*run.slice(k), upsample)
        if Z.min() < 0:
            ax.contourf(gx, gy, Z.T, levels=[Z.min() - 1, 0.0], colors=[tint(C_SHORT, 0.30)], zorder=2)
            ax.contour(gx, gy, Z.T, levels=[0.0], colors=[C_SHORT], linewidths=0.5, zorder=3)

        # path so far and the ego vehicle (rear-axle centre: the set lives there)
        ax.plot(run.path[:k + 1, 1], run.path[:k + 1, 2], color=INK2, lw=0.6, zorder=5)
        ax.plot(x, y, "o", ms=2.0, color=INK, zorder=7)

        sid, gv = run.used[k], run.gv(k)
        ax.set_title(f"({chr(97 + i)})", loc="left", fontsize=fs, fontweight="bold", color=INK, pad=1.5)
        ax.set_title(fr"$t = {t:.1f}$ s,  ${SYM_SHORT}_{{{sid}}}$ from ${SYM_LONG}_{{{gv}}}$",
                     loc="right", fontsize=fs, color=INK, pad=1.5)

        ax.set_xlim(xlim)
        ax.set_ylim(Y0 - 0.3, Y0 + YW + 0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_yticks([])
        for sd in ("top", "right", "left"):
            ax.spines[sd].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(axis="x", labelsize=fs - 1, width=0.5, length=2, pad=1.5, colors=INK2)

    for ax in axes[(rows - 1) * cols:len(ks)]:
        ax.set_xlabel("$x$ (m)", fontsize=fs, labelpad=1)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", ms=3, color=INK, label="ego vehicle"),
        mpatches.Patch(fc=tint(C_SHORT, 0.30), ec=C_SHORT, lw=0.5,
                       label=fr"${SYM_SHORT}_k$: area where $V < 0$"),
        mpatches.Patch(fc=C_DYN, ec="none", label="obstacle"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               frameon=False, fontsize=fs, handlelength=1.2, handleheight=0.8, columnspacing=1.0,
               borderaxespad=0.0)
    fig.tight_layout(pad=0.15, h_pad=0.4, w_pad=0.6)
    _save(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="results/run_data.pkl")
    p.add_argument("--root", default=".", help="directory the set paths in the run data are relative to")
    p.add_argument("--out", default="results/paper/waiting")
    p.add_argument("--times", type=float, nargs="+", help="snapshot times (s); default: automatic")
    p.add_argument("--xlim", type=float, nargs=2, help="x range of the panels (m)")
    p.add_argument("--width", type=float, default=3.5, help="figure width (in), 3.5 = one column")
    p.add_argument("--cols", type=int, default=2, help="panels per row (1 = stacked)")
    p.add_argument("--upsample", type=int, default=4, help="cubic upsampling of the set (1 = raw grid)")
    p.add_argument("--v-tol", type=float, default=0.05, help="speed below which the vehicle is at rest")
    p.add_argument("--list", action="store_true", help="print one line per step at rest and exit")
    a = p.parse_args()

    run = Run(a.data, a.root)
    if a.list:
        w, h = run.cfg['verifier']['window_size']
        print_table(run, a.v_tol, (w / (run.nx - 1)) * (h / (run.ny - 1)))
    else:
        times = a.times or auto_times(run, a.v_tol)
        print("snapshot times:", ", ".join(f"{t:.2f}" for t in times))
        plot_waiting(run, times, a.out, xlim=a.xlim, width=a.width, upsample=a.upsample,
                     cols=a.cols)
