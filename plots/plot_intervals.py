"""
Paper figure: definition of the times of a set, as a schematic in the style of the
schedule plot (plot_paper.py). Two long-horizon sets and two short-horizon sets; the
first of each kind is annotated:

    P   computation time         (solid bar)
    C   from the start of the computation to the start of the set's interval
        (computation + wait, dotted line: the computational budget)
    T   length of the interval covered by the set (light bar)

with subscript l (long horizon) or s (short horizon). The durations are illustrative,
not taken from a run.

Usage:
    python3 plots/plot_intervals.py                 # -> results/paper/intervals.pdf / .png
    python3 plots/plot_intervals.py --fs 13 --width 3.5
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plot_paper import C_LONG, C_SHORT, INK, INK2, GRID, SYM_LONG, SYM_SHORT, tint, _save

# Illustrative durations (s): (computation start, P, C, T) of each set
LONG = [(0.0, 1.7, 2.6, 7.0), (2.6, 1.7, 2.6, 7.0)]      # period 2.6 between long-horizon sets
SHORT = [(3.4, 0.6, 1.0, 2.4), (4.4, 0.6, 1.0, 2.4)]     # period 1.0 between short-horizon sets


def plot_intervals(out, width=7.0, fs=11):
    t_end = max(c0 + C + T for c0, P, C, T in LONG + SHORT) + 0.3
    ROW, BAR_H = 1.0, 0.55
    ANN = 0.75                      # room above / below an annotated row for the arrows

    # rows from the top: L_j (annotated), L_j+1, gap, S_k (annotated), S_k+1
    y_l = [0.0, -(ANN + ROW)]
    y_s0 = y_l[1] - ROW - 2 * ANN
    y_s = [y_s0, y_s0 - (ANN + ROW)]
    y_top, y_bot = y_l[0] + ANN + 0.35, y_s[1] - 0.6

    height = (y_top - y_bot) * 0.36 * fs / 11 + 0.6
    fig, ax = plt.subplots(figsize=(width, height))

    def bar(t0, t1, yc, face, edge=None, h=BAR_H):
        ax.add_patch(mpatches.FancyBboxPatch(
            (t0, yc - h / 2), t1 - t0, h, boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=1.0 if edge else 0, edgecolor=edge or "none", facecolor=face,
            zorder=3, mutation_aspect=0.3))

    def dim(t0, t1, y, text, color=INK, above=True):
        """dimension arrow t0 <-> t1 at height y, label above (or below) its middle"""
        ax.annotate("", xy=(t1, y), xytext=(t0, y),
                    arrowprops=dict(arrowstyle="<|-|>", color=color, lw=1.0,
                                    mutation_scale=0.9 * fs, shrinkA=0, shrinkB=0), zorder=5)
        ax.text((t0 + t1) / 2, y + (0.12 if above else -0.12), text, ha="center",
                va="bottom" if above else "top", fontsize=fs, color=color, zorder=5)

    def tick(t, y0, y1):
        """thin vertical guide from the bar to its dimension arrow"""
        ax.plot([t, t], [y0, y1], color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=2)

    def one_set(c0, P, C, T, yc, color, name, sub, annotate):
        h0 = c0 + C
        bar(h0, h0 + T, yc, tint(color), edge=color)                       # interval covered
        bar(c0, c0 + P, yc, color, h=BAR_H * 0.5)                          # computation
        ax.plot([c0 + P, h0], [yc, yc], color=color, lw=1.0, ls=(0, (1, 1.5)), zorder=2)  # wait
        ax.text(h0 + T / 2, yc, name, ha="center", va="center", fontsize=fs, color=INK, zorder=4)
        if not annotate:
            return
        y_up, y_dn = yc + BAR_H / 2 + 0.3, yc - BAR_H / 2 - 0.3
        for t in (c0, h0, h0 + T):
            tick(t, yc + BAR_H / 2, y_up)
        tick(c0 + P, yc - BAR_H / 4, y_dn)
        tick(c0, yc - BAR_H / 4, y_dn)
        dim(c0, h0, y_up, f"$C_{sub}$")
        dim(h0, h0 + T, y_up, f"$T_{sub}$")
        dim(c0, c0 + P, y_dn, f"$P_{sub}$", above=False)

    for k, ((c0, P, C, T), yc) in enumerate(zip(LONG, y_l)):
        name = fr"${SYM_LONG}_{{j}}$" if k == 0 else fr"${SYM_LONG}_{{j+1}}$"
        one_set(c0, P, C, T, yc, C_LONG, name, "l", annotate=(k == 0))
    for k, ((c0, P, C, T), yc) in enumerate(zip(SHORT, y_s)):
        name = fr"${SYM_SHORT}_{{k}}$" if k == 0 else fr"${SYM_SHORT}_{{k+1}}$"
        one_set(c0, P, C, T, yc, C_SHORT, name, "s", annotate=(k == 0))

    # light vertical lines over the whole height where a set starts to be used and the
    # computation of the following set starts (same instant: start of interval = c0 + C)
    # Each line stays inside its own group (long above the separator, short below it).
    y_sep = (y_l[1] + y_s[0]) / 2 + 0.15
    for sets, (ya, yb) in ((LONG, (y_sep, y_top)), (SHORT, (y_bot, y_sep))):
        for (c0, P, C, T), (c0_next, *_) in zip(sets, sets[1:]):
            t_use = c0 + C
            if abs(t_use - c0_next) < 1e-9:
                ax.plot([t_use, t_use], [ya, yb], color=GRID, lw=1.2, zorder=0.5)

    # group labels on the left, separator between the groups
    def group(y0, y1, text):
        ax.text(-0.012, (y0 + y1) / 2, text, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=fs, color=INK, linespacing=1.15)
        ax.plot([-0.006, -0.006], [y1 - BAR_H / 2, y0 + BAR_H / 2], transform=ax.get_yaxis_transform(),
                color=INK2, lw=0.8, clip_on=False)
    group(y_l[0], y_l[1], "long-horizon\nsets")
    group(y_s[0], y_s[1], "short-horizon\nsets")
    ax.axhline(y_sep, color=GRID, lw=0.8, zorder=0)

    # time axis without numbers (durations are illustrative)
    ax.set_xlim(0, t_end)
    ax.set_ylim(y_bot, y_top)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.annotate("", xy=(1.0, 0.0), xytext=(0.985, 0.0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=0.9 * fs))
    ax.set_xlabel("simulation time", fontsize=fs)

    handles = [
        mpatches.Patch(facecolor=INK2, edgecolor="none", label="computation"),
        mpatches.Patch(facecolor=tint(INK2, 0.22), edgecolor=INK2, lw=0.9, label="time interval covered"),
        plt.Line2D([], [], color=INK2, lw=0.9, ls=(0, (1, 1.5)), label="computational budget"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
              fontsize=fs, handlelength=1.6, handleheight=0.9, columnspacing=1.2, borderaxespad=0.2)

    fig.tight_layout(pad=0.3)
    _save(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/paper/intervals")
    p.add_argument("--width", type=float, default=7.0, help="figure width (in)")
    p.add_argument("--fs", type=float, default=11.0, help="font size (pt)")
    a = p.parse_args()
    plot_intervals(a.out, width=a.width, fs=a.fs)
