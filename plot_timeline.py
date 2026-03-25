import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import sys

# ===========================================================================
# Load data
# ===========================================================================
json_path = sys.argv[1] if len(sys.argv) > 1 else "./hj_set_update.json"
with open(json_path) as f:
    data = json.load(f)

mpc_dt        = data["mpc_step_length"]       # e.g. 0.1
total_steps   = data["total_steps"]            # e.g. 199
horizons      = data["horizon"]                # [[0,10],[5,15],[10,20],[15,20]]
set_compute   = data["set_compute_time"]       # [10.84, 9.43, ...]  actual compute time per set
compute_loop_raw = data["compute_loop_time"]
# compute_loop_time may be a scalar (fixed period) or a list (per-set periods)
if isinstance(compute_loop_raw, list):
    compute_loop = compute_loop_raw
else:
    compute_loop = [compute_loop_raw] * len(horizons)
used_sets     = data["used_sets"]              # set index used at each MPC step

n_sets        = len(horizons)
total_time    = total_steps * mpc_dt

# ===========================================================================
# Color scheme (blue, orange, yellow, green)
# ===========================================================================
COLORS = ["#4472C4", "#ED7D31", "#FFC000", "#70AD47"]

# ===========================================================================
# Compute Verification Loop timeline
# - verify_loop_end_times: absolute time when each loop PERIOD ends (cumsum of compute_loop)
# - verify_compute_end_times: absolute time when each SET is actually ready (cumsum of set_compute)
# ===========================================================================
verify_loop_end_times    = np.cumsum(compute_loop)   # loop period boundaries (for sleep block end)
verify_start_times       = np.concatenate([[0], verify_loop_end_times[:-1]])
verify_compute_end_times = verify_start_times + np.array(set_compute)  # actual compute done time

# ===========================================================================
# Compute MPC Loop time ranges per set
# ===========================================================================
# MPC starts as soon as the first set is computed (not at loop period end)
mpc_start = verify_compute_end_times[0]

# Find the first and last MPC step index for each set
mpc_segments = []
used_arr = np.array(used_sets)
for sid in range(n_sets):
    idxs = np.where(used_arr == sid)[0]
    if len(idxs) == 0:
        continue
    t0 = mpc_start + idxs[0]  * mpc_dt
    t1 = mpc_start + (idxs[-1] + 1) * mpc_dt
    mpc_segments.append((sid, t0, t1))

# ===========================================================================
# Plot
# ===========================================================================
# Layout scales with number of sets:
# - horizon bars stack downward, each needing 0.65 units
# - add fixed padding for verify row (top) and MPC row (bottom)
BAR_H           = 0.55
HORIZON_STRIDE  = 0.65  # vertical spacing between stacked horizon bars
TOP_PAD         = 2.0   # space above horizon bars to verification row
BOT_PAD         = 2.2   # space below horizon bars to MPC row
EXTRA_PAD       = 1.2   # margin above/below everything

horizon_block_h = n_sets * HORIZON_STRIDE
total_h         = TOP_PAD + horizon_block_h + BOT_PAD + 2 * EXTRA_PAD

ROW_MPC         = EXTRA_PAD
ROW_LABEL       = ROW_MPC    + BOT_PAD + (horizon_block_h / 2)
ROW_VERIFY      = ROW_LABEL  + (horizon_block_h / 2) + TOP_PAD

# Recompute horizon y-offsets dynamically
horizon_y_offsets = [i * (-HORIZON_STRIDE) + (horizon_block_h - HORIZON_STRIDE) / 2
                     for i in range(n_sets)]

fig_w = max(18, int(verify_loop_end_times[-1] * 0.5))
fig, ax = plt.subplots(figsize=(fig_w, total_h))
ax.set_xlim(-0.5, max(verify_loop_end_times[-1], total_time + mpc_start) + 1)
ax.set_ylim(EXTRA_PAD - 0.8, ROW_VERIFY + 1.2)
ax.axis("off")

# ---------- Row labels ----------
ax.text(-0.3, ROW_VERIFY, "Verification\nLoop",
        ha="right", va="center", fontsize=10, fontweight="bold")
ax.text(-0.3, ROW_MPC, "MPC Loop",
        ha="right", va="center", fontsize=10, fontweight="bold")
ax.plot([-0.3, -0.05], [ROW_VERIFY, ROW_VERIFY], "k--", lw=1.5)
ax.plot([-0.3, -0.05], [ROW_MPC,    ROW_MPC   ], "k--", lw=1.5)

# ---------- Verification Loop bars ----------
for k in range(n_sets):
    t0  = verify_start_times[k]
    t_c = verify_compute_end_times[k]   # actual compute end
    t1  = verify_loop_end_times[k]      # loop period end (compute + sleep)
    color = COLORS[k % len(COLORS)]

    # Compute block
    rect_compute = mpatches.FancyBboxPatch(
        (t0, ROW_VERIFY - BAR_H/2), set_compute[k], BAR_H,
        boxstyle="round,pad=0.05", linewidth=1.2,
        edgecolor="white", facecolor=color
    )
    ax.add_patch(rect_compute)
    ax.text(t0 + set_compute[k]/2, ROW_VERIFY,
            f"Compute\n{set_compute[k]:.1f}s",
            ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

    # Sleep block (if any)
    sleep_dur = t1 - t_c
    if sleep_dur > 0.05:
        rect_sleep = mpatches.FancyBboxPatch(
            (t_c, ROW_VERIFY - BAR_H/2), sleep_dur, BAR_H,
            boxstyle="round,pad=0.05", linewidth=1.2,
            edgecolor="white", facecolor="#1a1a1a"
        )
        ax.add_patch(rect_sleep)
        ax.text(t_c + sleep_dur/2, ROW_VERIFY, "sleep",
                ha="center", va="center", fontsize=7.5, color="white")

# ---------- Horizon bars (middle layer, stacked with offset) ----------
for k, (h0, h1) in enumerate(horizons):
    color  = COLORS[k % len(COLORS)]
    y_base = ROW_LABEL + horizon_y_offsets[k]
    # Convert horizon time (relative to MPC start) to absolute time
    abs_h0 = mpc_start + h0
    abs_h1 = mpc_start + h1
    rect = mpatches.FancyBboxPatch(
        (abs_h0, y_base - 0.22), abs_h1 - abs_h0, 0.44,
        boxstyle="round,pad=0.03", linewidth=1,
        edgecolor="white", facecolor=color, alpha=0.92
    )
    ax.add_patch(rect)
    label = f"verification set for {h0:.0f}s to {h1:.0f}s" \
            if h1 - h0 > 1 else f"set to {h1:.0f}s"
    ax.text((abs_h0 + abs_h1)/2, y_base,
            label, ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

# ---------- Arrows: verification → horizon ----------
for k in range(n_sets):
    t_pub = verify_compute_end_times[k]
    color = COLORS[k % len(COLORS)]
    y_base = ROW_LABEL + horizon_y_offsets[k]
    ax.annotate("", xy=(t_pub, y_base + 0.22),
                xytext=(t_pub, ROW_VERIFY - BAR_H/2 - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

# ---------- Arrows: horizon → MPC ----------
for k in range(n_sets):
    t_pub = verify_compute_end_times[k]
    color = COLORS[k % len(COLORS)]
    y_base = ROW_LABEL + horizon_y_offsets[k]
    ax.annotate("", xy=(t_pub, ROW_MPC + BAR_H/2 + 0.05),
                xytext=(t_pub, y_base - 0.22),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

# ---------- MPC Loop bars ----------
for sid, t0, t1 in mpc_segments:
    color = COLORS[sid % len(COLORS)]
    rect = mpatches.FancyBboxPatch(
        (t0, ROW_MPC - BAR_H/2), t1 - t0, BAR_H,
        boxstyle="round,pad=0.05", linewidth=1.2,
        edgecolor="white", facecolor=color
    )
    ax.add_patch(rect)
    ax.text((t0 + t1)/2, ROW_MPC, "MPC LOOP",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

# ---------- Time axis ticks ----------
ax_time = ax.inset_axes([0, -0.08, 1, 0.06])
ax_time.set_xlim(ax.get_xlim())
ax_time.set_ylim(0, 1)
ax_time.axis("off")

x_min, x_max = ax.get_xlim()
tick_step = 5.0 if (x_max - x_min) > 30 else 1.0
ticks = np.arange(0, x_max + tick_step, tick_step)
for t in ticks:
    ax_time.axvline(t, color="gray", lw=0.5, ymin=0, ymax=0.4)
    # Convert to MPC-relative time for labels
    rel_t = t - mpc_start
    label = f"{rel_t:.0f}" if rel_t >= 0 else ""
    ax_time.text(t, -0.2, label, ha="center", va="top", fontsize=8, color="gray")

# Dashed grid lines
for t in ticks:
    ax.axvline(t, color="gray", lw=0.4, ls="--", alpha=0.4)

plt.suptitle("Verification Loop & MPC Loop Timeline", fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
out = json_path.replace(".json", "_timeline.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()