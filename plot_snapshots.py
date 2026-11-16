"""
Paper figure: the urban road scenario and snapshots of the closed loop, stacked in rows
with the same axes.

    row 1      scenario (top view, as in TTR.ipynb, "Urban road scenario")
    rows 2..   map panel of the animation at given times, reduced to: border of the
               short-horizon set in use (V = 0, min over yaw and vel), its long-horizon
               window and target, static and dynamic obstacles, vehicle position and
               heading, driven path.
               The last row is the time the goal is reached.

The figure is drawn at the size of the scenario figure in TTR.ipynb (--width, in) so the
scenario looks the same; the fonts and the snapshot lines are sized for the width the
figure has in the paper (--paper-width, in): --fs is the font size in the paper (pt).

Usage:
    python3 plot_snapshots.py --times 6 14            # results/run_data.pkl -> results/paper/snapshots
    python3 plot_snapshots.py --times 6 14 --static   # scenario without the crossing road users
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")      # script: no display (imported from a notebook: keep its backend)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import urban_scene as us
from obstacles import obstacle_rects

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colours of the animation map panel (animation.py), darker where needed on the cream background
C_BG = "#f6f1e3"   # cream
C_OBS, C_DYN, C_V0, C_PATH, C_CUR, C_PRED = "gray", "crimson", "red", "#1a1a1a", us.C_EGO, "#ff8c00"
C_WIN, C_TGT, C_EDGE = "#3182bd", "#2ca02c", "#333333"   # medium blue (window), green (targets)
# vehicle (C_CUR): same colour as the ego car in the scenario row


def nearest(grid, v):
    return int(np.argmin(np.abs(np.asarray(grid) - v)))


def style_axes(ax, fs, show_x):
    ax.set_xlim(us.X0, us.XW)
    ax.set_ylim(-us.SIDEWALK, us.YW + us.SIDEWALK)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(us.X0, us.XW + 1e-9, 10.0))
    ax.set_yticks([0, us.YW])
    ax.tick_params(axis="both", labelsize=fs, width=0.8 * fs / 10, length=4 * fs / 10,
                   labelbottom=show_x)
    ax.set_ylabel("y [m]", fontsize=fs)
    ax.set_xlabel("x [m]" if show_x else "", fontsize=fs)


def time_label(ax, t, fs):
    ax.text(0.004, 0.97, f"$t = {t:.1f}$ s", transform=ax.transAxes, ha="left", va="top",
            fontsize=0.85 * fs, zorder=20,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=0.6 * fs / 10))


def active_window(data, t):
    """Long-horizon (global) set in use at time t, as in the animation: the latest one
    that has started (t >= t0) and is available (computed) by t."""
    gtl = data.get("global_timeline") or []
    if not gtl:
        return None
    active = [g for g in gtl if max(g["t0"], g.get("sim_time_end", g["t0"])) <= t + 1e-9]
    return active[-1] if active else gtl[0]


def draw_snapshot(ax, data, t_query, k_scale, fs):
    cfg = data["config"]
    dt = cfg["controller"]["timestep"]
    real = np.asarray(data["real_path"])
    pred_all = data["pred_path"]
    used = data["controller"]["used_sets"]
    k = int(np.clip(round(t_query / dt), 0, min(len(pred_all), len(used)) - 1))
    t, x, y, yaw, _ = real[k]

    # obstacles (curbs, blockages), dynamic obstacles at time t
    for ox, oy, ow, oh in cfg["obstacles"]:
        ax.add_patch(plt.Rectangle((ox, oy), ow, oh, facecolor=C_OBS, edgecolor=C_EDGE,
                                   alpha=0.6, lw=0.8 * k_scale, zorder=5))
    for dyn in cfg["dynamic_obstacles"]:
        for ox, oy, ow, oh in obstacle_rects(t, dyn):
            ax.add_patch(plt.Rectangle((ox, oy), ow, oh, facecolor=C_DYN, edgecolor=C_EDGE,
                                       alpha=0.7, lw=0.8 * k_scale, zorder=7))
    # goal (outline), long-horizon window in use at time t and its target
    gx, gy, gw, gh = cfg["goal"]
    ax.add_patch(plt.Rectangle((gx, gy), gw, gh, fill=False, edgecolor=C_TGT, ls="--",
                               lw=1.0 * k_scale, zorder=4))
    # the window S^(j) the short-horizon set in use was computed from (the set lives on its
    # grid, so it is always inside it); fallback: the latest window available at time t
    entry = next(e for e in data["local_timeline"] if e["set_idx"] == used[k])
    g = next((g for g in data.get("global_timeline") or []
              if g["version"] == entry.get("global_version")), None) or active_window(data, t)
    if g is not None:
        (wx0, wx1), (wy0, wy1) = g["window"]
        ax.add_patch(plt.Rectangle((wx0, wy0), wx1 - wx0, wy1 - wy0, fill=False, edgecolor=C_WIN,
                                   ls="--", lw=2.0 * k_scale, zorder=12))
        tx, ty, tw, th = g["target"]
        ax.add_patch(plt.Rectangle((tx, ty), tw, th, facecolor=C_TGT, edgecolor=C_TGT, alpha=0.35,
                                   lw=1.0 * k_scale, zorder=4.5))

    # border of the short-horizon set in use at step k (min over yaw, vel)
    Z_all = data["local_xy"][used[k]]
    hj_dt = cfg["verifier"]["timestep"]
    it = nearest(entry["interval"][0] + np.arange(Z_all.shape[0]) * hj_dt, t)
    Z = Z_all[it]
    n_x, n_y = Z.shape
    (wx0, wx1), (wy0, wy1) = entry["window"]
    if Z.min() < 0 < Z.max():
        ax.contour(np.linspace(wx0, wx1, n_x), np.linspace(wy0, wy1, n_y), Z.T, levels=[0.0],
                   colors=C_V0, linewidths=1.6 * k_scale, zorder=8)

    # driven path, current position and heading
    ax.plot(real[:k + 1, 1], real[:k + 1, 2], color=C_PATH, lw=1.4 * k_scale, zorder=9,
            solid_capstyle="round")
    ax.plot(x, y, "o", color=C_CUR, mec="k", mew=0.6 * k_scale, ms=5 * k_scale, zorder=11)
    L = 3.0
    ax.annotate("", xy=(x + L * np.cos(yaw), y + L * np.sin(yaw)), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=C_PATH, lw=1.2 * k_scale,
                                mutation_scale=8 * k_scale, shrinkA=0, shrinkB=0), zorder=11)

    ax.set_facecolor(C_BG)
    ax.grid(True, color="black", alpha=0.10, lw=0.5 * k_scale, zorder=0)
    time_label(ax, t, fs)
    return t


def goal_time(data, v_stop=0.1):
    """First time the rear axle is in the goal box with |vel| < v_stop (None if never)."""
    gx, gy, gw, gh = data["config"]["goal"]
    real = np.asarray(data["real_path"])[:len(data["pred_path"])]
    t, x, y, v = real[:, 0], real[:, 1], real[:, 2], real[:, 4]
    hit = np.where((gx <= x) & (x <= gx + gw) & (gy <= y) & (y <= gy + gh) & (np.abs(v) < v_stop))[0]
    return float(t[hit[0]]) if len(hit) else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="results/run_data.pkl")
    p.add_argument("--out", default="results/paper/snapshots")
    p.add_argument("--times", type=float, nargs="+", required=True,
                   help="snapshot times (s); the goal time is added as the last row")
    p.add_argument("--t-final", type=float, default=None,
                   help="time of the last row (default: when the goal is reached)")
    p.add_argument("--static", action="store_true", help="scenario row without the road users")
    p.add_argument("--width", type=float, default=24.0, help="drawing width (in), as in TTR.ipynb")
    p.add_argument("--paper-width", type=float, default=7.16, help="width in the paper (in)")
    p.add_argument("--fs", type=float, default=10.0, help="font size in the paper (pt)")
    p.add_argument("--no-legend", action="store_true")
    a = p.parse_args()

    with open(a.data, "rb") as f:
        data = pickle.load(f)
    for key in ("pred_path", "local_xy"):
        if key not in data:
            raise SystemExit(f"'{key}' missing in {a.data}: run the simulation again "
                             "(older runs did not save it)")

    done = data.get("mission_done")
    t_final = a.t_final if a.t_final is not None else (done["time"] if done else goal_time(data))
    if t_final is None:
        dt = data["config"]["controller"]["timestep"]
        t_final = (len(data["pred_path"]) - 1) * dt
        print(f"WARNING: goal not reached in this run, last row at the end ({t_final:.1f} s)")
    times = list(a.times) + [t_final]

    k = a.width / a.paper_width        # paper pt -> drawing pt
    fs = a.fs * k
    n = 1 + len(times)
    ar = (us.YW + 2 * us.SIDEWALK) / (us.XW - us.X0)
    # explicit layout (inches): equal axes stacked with a small gap, room below for the
    # x label and above for the legend, as in schedule.pdf (the saved figure is cropped
    # to its content)
    pt = fs / 72.0                      # one font height, in
    left, right = 2.5 * pt, 0.5 * pt
    ax_w = a.width - left - right
    ax_h = ax_w * ar
    gap = 1.2 * pt
    bottom = 2.4 * pt
    top = (0.5 if a.no_legend else 3.4) * pt
    axes_top = bottom + n * ax_h + (n - 1) * gap
    fig_h = axes_top + top
    fig = plt.figure(figsize=(a.width, fig_h))
    axes = []
    for i in range(n):
        y0 = bottom + (n - 1 - i) * (ax_h + gap)
        axes.append(fig.add_axes([left / a.width, y0 / fig_h, ax_w / a.width, ax_h / fig_h]))

    us.draw_scenario(axes[0], road_users=not a.static)
    for txt in axes[0].texts:          # road-user labels: readable in the paper
        if txt.get_bbox_patch() is not None:
            txt.set_fontsize(0.65 * fs)
    style_axes(axes[0], fs, show_x=False)

    for ax, t in zip(axes[1:], times):
        draw_snapshot(ax, data, t, k, fs)
        style_axes(ax, fs, show_x=(ax is axes[-1]))

    if not a.no_legend:
        handles = [
            Line2D([], [], color=C_V0, lw=1.6 * k, label=r"$V_s = 0$"),
            Line2D([], [], color=C_WIN, lw=2.0 * k, ls="--", label=r"$S^{(j)}$"),
            Patch(facecolor=C_TGT, alpha=0.35, edgecolor=C_TGT, label=r"$\mathcal{G}^{j}$"),
            Patch(fill=False, edgecolor=C_TGT, ls="--", lw=1.0 * k, label="goal"),
            Line2D([], [], color=C_PATH, lw=1.4 * k, label="driven path"),
            Line2D([], [], marker="o", ls="", color=C_CUR, mec="k", ms=5 * k, label="vehicle"),
            Patch(facecolor=C_OBS, alpha=0.6, edgecolor=C_EDGE, label="static obstacle"),
            Patch(facecolor=C_DYN, alpha=0.7, edgecolor=C_EDGE, label="dynamic obstacle"),
        ]
        # above the first row, left-aligned with the axes (as in schedule.pdf)
        fig.legend(handles=handles, loc="lower left",
                   bbox_to_anchor=(left / a.width, (axes_top + 0.3 * pt) / fig_h), ncol=4,
                   frameon=False, borderaxespad=0.0,
                   fontsize=fs, handlelength=2.0, columnspacing=2.0)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(a.out + ".png", dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"saved {a.out}.pdf / .png  (rows at t = {', '.join(f'{t:.1f}' for t in times)} s)")


if __name__ == "__main__":
    main()
