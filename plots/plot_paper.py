"""
Paper figures from a saved run (results/run_data.pkl, written by Animation.save_run_data).

    1) value.pdf     HJ value V(z(t)) along the closed-loop trajectory, whole simulation
                     (same data as the bottom panel of the animation)
    2) schedule.pdf  computation schedule of the long- and short-horizon sets and the set
                     used by the MPC, over the first T_MAX seconds

Usage:
    python3 plots/plot_paper.py                      # results/run_data.pkl -> results/paper/
    python3 plots/plot_paper.py --data other.pkl --tmax 10
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# Paper style: serif text matching LaTeX (STIX), vector output
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Palette (categorical slots 1-3 of the timeline plot) and text / grid tokens
C_LONG, C_SHORT, C_MPC = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

# Symbols of the sets in the figure (change to match the notation of the paper)
SYM_LONG, SYM_SHORT = "L", "S"


def tint(hex_color, a=0.28):
    """Opaque light tint of a colour (blended on white)."""
    c = np.array(mcolors.to_rgb(hex_color))
    return tuple(a * c + (1 - a))


# ----------------------------------------------------------------------
# 1) V over time
# ----------------------------------------------------------------------
def plot_value(data, out, width=7.0, height=2.0, fs=10):
    dt = data['config']['controller']['timestep']
    # V of the active short-horizon set at the vehicle's full state (x, y, yaw, vel), whatever
    # SAFETY_V_MODE the MPC constraint uses. Runs saved before 'state_cost' was recorded
    # only have the value used by the MPC ('real_cost').
    V = np.asarray(data.get('state_cost', []), dtype=float)
    if len(V) == 0:
        print("WARNING: no 'state_cost' in the run data (older run): plotting the value used "
              "by the MPC ('real_cost'), which is the min over (yaw, vel) in 'min' mode")
        V = np.asarray(data['real_cost'], dtype=float)
    t = np.arange(len(V)) * dt
    done = data.get('mission_done')
    if done is not None:
        keep = t <= done['time'] + 1e-9
        t, V = t[keep], V[keep]

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axhline(0.0, color=INK2, lw=0.8, zorder=1)
    ax.plot(t, V, color=C_LONG, lw=2.0, zorder=3, solid_capstyle="round")

    ax.set_xlim(0, t[-1])
    lo, hi = np.nanmin(V), np.nanmax(V)
    lo, hi = max(lo, -2.0), min(hi, 2.0)          # same clipping as the animation
    pad = 0.08 * max(hi - lo, 1e-3)
    ax.set_ylim(min(lo, 0.0) - pad, max(hi, 0.0) + pad)

    ax.set_xlabel("time (s)", fontsize=fs)
    ax.set_ylabel(r"$V(t, z(t))$", fontsize=fs)
    ax.tick_params(labelsize=fs - 1, width=0.8, length=3.5)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)

    fig.tight_layout(pad=0.3)
    _save(fig, out)


# ----------------------------------------------------------------------
# 2) Schedule of the long- and short-horizon sets
# ----------------------------------------------------------------------
def _rows(items):
    """Greedy row assignment: an item goes on the first row free at its start.
    items: list of (start, end). Returns the row of each item and the number of rows."""
    row_of, row_end = [], []
    for a, b in items:
        r = next((i for i, e in enumerate(row_end) if e <= a + 1e-9), len(row_end))
        if r == len(row_end):
            row_end.append(-np.inf)
        row_end[r] = b
        row_of.append(r)
    return row_of, max(len(row_end), 1)


def plot_schedule(data, out, t_max=10.0, width=7.0, fs=8):
    cfg = data['config']
    H_LONG = cfg['verifier']['horizon']
    gtl = [g for g in data['long_timeline'] if g.get('sim_time_start', g['t0']) < t_max]
    ltl = [e for e in data['short_timeline']
           if e.get('sim_start', e['interval'][0]) < t_max]
    mpc_dt = data['controller']['mpc_dt']
    used = list(data['controller']['used_sets'])
    used_ttr = list(data['controller'].get('used_ttr', []))
    set_gv = {e['set_idx']: e.get('long_version') for e in data['short_timeline']}

    # one row per long-horizon set; short-horizon sets share rows (the next one goes
    # on a row whose previous set has ended, so each row reads left to right)
    g_row, n_g = list(range(len(gtl))), max(len(gtl), 1)
    l_row, n_l = _rows([(e.get('sim_start', e['interval'][0]), e['interval'][1]) for e in ltl])

    ROW, BAR_H, GROUP_GAP = 1.0, 0.62, 0.9
    y = 0.0
    y_mpc = y                      # short-horizon set used by the MPC (safety constraint)
    y_ttr = y + ROW if used_ttr else y   # long-horizon set whose time-to-reach map is the MPC cost
    y = y_ttr + ROW / 2 + GROUP_GAP
    y_l = [y + ROW * (n_l - 1 - r) for r in range(n_l)]
    y += ROW * n_l + GROUP_GAP - ROW / 2
    y_g = [y + ROW * (n_g - 1 - r) for r in range(n_g)]
    y_top = y + ROW * (n_g - 1) + ROW / 2

    height = (0.95 + 0.205 * (y_top - y_mpc + ROW)) * fs / 8   # rows grow with the font
    fig, ax = plt.subplots(figsize=(width, height))
    x_per_in = t_max / (width * 0.80)           # seconds per inch (approx. axes width)
    MIN_W = 0.04

    def text_w(s, size):
        """approximate width of a label in seconds (mathtext subscripts are narrow)"""
        n = len(s.replace("$", "").replace("_", "").replace("{", "").replace("}", "").replace("\\", ""))
        return (n * 0.5 * size / 72 + 0.06) * x_per_in

    def bar(t0, t1, yc, face, edge=None, h=BAR_H, label=None, lsize=fs, lcolor=INK, align="center"):
        t1c = min(t1, t_max)
        if t1c <= t0:
            return
        w = max(t1c - t0, MIN_W)
        ax.add_patch(mpatches.FancyBboxPatch(
            (t0, yc - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.05",
            linewidth=0.9 if edge else 0, edgecolor=edge or "none", facecolor=face,
            zorder=3, clip_on=False, mutation_aspect=0.25))
        if label and w > text_w(label, lsize):
            x = t0 + w / 2 if align == "center" else t0 + 0.08
            ax.text(x, yc, label, ha=align, va="center", fontsize=lsize, color=lcolor, zorder=4)

    def arrow_out(t, yc):
        """the set continues past the plotted range"""
        ax.plot([t + 0.11], [yc], marker=">", ms=4, color=INK2, clip_on=False, zorder=4)

    def wait_line(c1, h0, yc, color):
        """the set is ready (end of its computation) and waits for its interval to start"""
        if h0 - c1 > 0.05:
            ax.plot([c1, min(h0, t_max)], [yc, yc], color=color, lw=0.9, ls=(0, (1, 1.5)), zorder=2)

    def labels_that_fit(cands, a, b, size):
        """first candidate label that fits in [a, b] (None if none does)"""
        return next((c for c in cands if text_w(c, size) < b - a), None)

    # --- long-horizon sets: computation (solid) then time interval covered (light)
    for g, r in zip(gtl, g_row):
        yc = y_g[r]
        c0, c1 = g.get('sim_time_start', g['t0']), g['sim_time_end']
        h0, h1 = g['t0'], g['t0'] + H_LONG
        name = fr"${SYM_LONG}_{{{g['version']}}}$"
        bar(h0, h1, yc, tint(C_LONG), edge=C_LONG)
        if c1 - c0 > 1e-6:   # zero length: computed before the run starts (L0, S0), not drawn
            bar(c0, c1, yc, C_LONG, h=BAR_H * 0.5)
        # budget line up to the start of its interval (or to the end of the plot, if it starts later)
        wait_line(c1, h0, yc, C_LONG)
        a = max(h0, c1) + 0.08                 # label starts after the computation bar
        if h0 < t_max:
            lab = labels_that_fit([name], a, min(h1, t_max), fs)
            if lab:
                ax.text(a, yc, lab, ha="left", va="center", fontsize=fs, color=INK, zorder=4)
        if c0 < t_max < h1:
            arrow_out(t_max, yc)

    # --- short-horizon sets (label: the long-horizon set each one is computed from)
    for e, r in zip(ltl, l_row):
        yc = y_l[r]
        c0, c1 = e.get('sim_start', e['interval'][0]), e.get('sim_end', e['interval'][0])
        h0, h1 = e['interval']
        gv = e.get('long_version')
        name = fr"${SYM_SHORT}_{{{e['set_idx']}}}$"
        bar(h0, h1, yc, tint(C_SHORT), edge=C_SHORT)
        if c1 - c0 > 1e-6:   # zero length: computed before the run starts (L0, S0), not drawn
            bar(c0, c1, yc, C_SHORT, h=BAR_H * 0.5)
        wait_line(c1, h0, yc, C_SHORT)
        a, b = max(h0, c1), min(h1, t_max)
        cands = ([f"{name} from " + fr"${SYM_LONG}_{{{gv}}}$"] if gv is not None else []) + [name]
        lab = labels_that_fit(cands, a, b, fs - 0.5)
        if lab:
            ax.text((a + b) / 2, yc, lab, ha="center", va="center", fontsize=fs - 0.5, color=INK, zorder=4)
        if h1 > t_max:
            arrow_out(t_max, yc)

    # --- MPC: consecutive steps using the same set -> one segment
    def segments(values):
        n_steps = min(len(values), int(np.ceil(t_max / mpc_dt)))
        seg = 0
        for k in range(1, n_steps + 1):
            if k == n_steps or values[k] != values[seg]:
                if values[seg] >= 0:
                    yield values[seg], seg * mpc_dt, min(k * mpc_dt, t_max)
                seg = k

    gap = 0.015 * x_per_in
    for sid, a, b in segments(used):
        bar(a + gap / 2, b - gap / 2, y_mpc, C_MPC, label=fr"${SYM_SHORT}_{{{sid}}}$",
            lsize=fs - 0.5, lcolor=INK)
    # same wording on every segment: the longest form that fits all of them (except slivers)
    ttr_segs = list(segments(used_ttr))
    forms = ["time-to-reach map of {}", "TTR of {}", "{}"]
    form = next((f for f in forms
                 if all(text_w(f.format(fr"${SYM_LONG}_{{{v}}}$"), fs - 0.5) < b - a
                        for v, a, b in ttr_segs if b - a > 1.0)), forms[-1])
    for v, a, b in ttr_segs:
        lab = labels_that_fit([form.format(fr"${SYM_LONG}_{{{v}}}$")], a, b, fs - 0.5)
        bar(a + gap / 2, b - gap / 2, y_ttr, tint(C_MPC, 0.35), edge=C_MPC, label=lab,
            lsize=fs - 0.5)

    # --- group labels on the left
    def group(y0, y1, text):
        ym = (y0 + y1) / 2
        ax.text(-0.012, ym, text, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=fs, color=INK, linespacing=1.15)
        ax.plot([-0.006, -0.006], [y0 - BAR_H / 2, y1 + BAR_H / 2], transform=ax.get_yaxis_transform(),
                color=INK2, lw=0.8, clip_on=False)
    group(min(y_g), max(y_g), "long-horizon\nsets")
    group(min(y_l), max(y_l), "short-horizon\nsets")
    group(y_mpc, y_ttr, "used by\nthe MPC")
    for sep in (y_ttr + ROW / 2 + GROUP_GAP / 2 - ROW / 4, min(y_g) - ROW / 2 - GROUP_GAP / 2 + ROW / 4):
        ax.axhline(sep, color=GRID, lw=0.8, zorder=0)

    # --- axes
    ax.set_xlim(0, t_max)
    ax.set_ylim(y_mpc - ROW / 2 - 0.1, y_top + 0.1)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, t_max + 1e-9, 1.0))
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="x", labelsize=fs, colors=INK, width=0.8, length=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_xlabel("simulation time (s)", fontsize=fs + 0.5)

    # --- legend (marks explained once, above the plot)
    handles = [
        mpatches.Patch(facecolor=INK2, edgecolor="none", label="computation"),
        mpatches.Patch(facecolor=tint(INK2, 0.22), edgecolor=INK2, lw=0.9,
                       label="time interval covered"),
        plt.Line2D([], [], color=INK2, lw=0.9, ls=(0, (1, 1.5)), label="computational budget"),
        mpatches.Patch(facecolor=C_MPC, edgecolor="none", label="used by the MPC"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4 if fs <= 9 else 2, frameon=False,
              fontsize=fs, handlelength=1.6, handleheight=0.9, columnspacing=1.2, borderaxespad=0.2)

    fig.tight_layout(pad=0.3)
    _save(fig, out)


def _save(fig, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {out}.pdf / .png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="results/run_data.pkl")
    p.add_argument("--out", default="results/paper")
    p.add_argument("--tmax", type=float, default=10.0, help="end of the schedule plot (s)")
    p.add_argument("--value-width", type=float, default=7.0, help="width of the value plot (in)")
    p.add_argument("--schedule-fs", type=float, default=11.0,
                   help="font size of the schedule plot (pt, at its width of 7 in)")
    a = p.parse_args()
    with open(a.data, "rb") as f:
        data = pickle.load(f)
    plot_value(data, os.path.join(a.out, "value"), width=a.value_width)
    plot_schedule(data, os.path.join(a.out, "schedule"), t_max=a.tmax, fs=a.schedule_fs)
