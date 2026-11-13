import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
from matplotlib.animation import FuncAnimation, PillowWriter
import time as tm
import os
import shutil

from tqdm import tqdm

from controller import Controller
from obstacles import obstacle_rects
from log import log

class Animation:

    def __init__(self, registry, state_queue, stop_event):
        self.registry = registry
        self.state_queue = state_queue
        self.stop_event = stop_event

        self.config = self.registry['config']

        self.HJ_TIMESTEP = self.config['verifier']['timestep']
        self.HJ_GRID_MIN = [axis['bounds'][0] for axis in self.config['verifier']['axes'] if axis['name'] != 't']
        self.HJ_GRID_MAX = [axis['bounds'][1] for axis in self.config['verifier']['axes'] if axis['name'] != 't']
        self.HJ_GRID_STP = [axis['points'] for axis in self.config['verifier']['axes'] if axis['name'] != 't']

        self.OBSTACLES = self.config['obstacles']
        # Whole map extent (the value functions only cover the global window)
        self.MAP_X = (self.config['X0'], self.config['X0'] + self.config['XW'])
        self.MAP_Y = (self.config['Y0'], self.config['Y0'] + self.config['YW'])
        self.DYN_OBS = self.config['dynamic_obstacles']
        self.VEHICLE = self.config.get('vehicle')
        # Size of the drawn glyphs (arrows, zoom), relative to the 1:10 corridor
        self.SCALE = self.config['wheelbase'] / 0.32
        self.T_PRED = self.config['verifier']['t_prediction']

    @staticmethod
    def _wrap_yaw(yaw):
        # return (yaw + np.pi) % (2 * np.pi) - np.pi
        return yaw

    @staticmethod
    def _draw_obstacles(ax, obstacles=None):
        if obstacles is None:
            obstacles = []
        for ox, oy, ow, oh in obstacles:
            ax.add_patch(
                plt.Rectangle(
                    (ox, oy), ow, oh,
                    edgecolor="black",
                    facecolor="gray",
                    alpha=0.4,
                )
            )


    def _hj_axes(self):
        x_grid = np.linspace(self.HJ_GRID_MIN[0], self.HJ_GRID_MAX[0], self.HJ_GRID_STP[0])
        y_grid = np.linspace(self.HJ_GRID_MIN[1], self.HJ_GRID_MAX[1], self.HJ_GRID_STP[1])
        # periodic yaw on the hj_reachability grid: N nodes at -pi + k*2pi/N, +pi excluded
        yaw_grid = self.HJ_GRID_MIN[2] + np.arange(self.HJ_GRID_STP[2]) * \
            (self.HJ_GRID_MAX[2] - self.HJ_GRID_MIN[2]) / self.HJ_GRID_STP[2]
        vel_grid = np.linspace(self.HJ_GRID_MIN[3], self.HJ_GRID_MAX[3], self.HJ_GRID_STP[3])
        return x_grid, y_grid, yaw_grid, vel_grid

    @staticmethod
    def _nearest_index(grid, value):
        return int(np.argmin(np.abs(grid - value)))


    def animate_heat_map_global(
        self,
        verifier_timeline,
        used_sets,
        real_path,
        pred_path,
        real_cost=None,
        interp_cost=None,
        save_name=None,
        obstacles=None,
        interval_ms=100,
        global_timeline=None,
    ):
        """
        Animate:
        1) real VF slice in (x, y)  +  -∇V arrows on predicted trajectory
        2) gradient slope per stage (True V vs linear surrogate along horizon)
        3) real VF slice in (yaw, vel)
        4) time vs HJ cost
        5) local zoom: current point + predicted trajectory + -∇V arrows
 
        Parameters
        ----------
        latest_vf : ndarray
            Value function array from registry results.
        vf_interp : RegularGridInterpolator
            Interpolator built from latest_vf.
        latest_info : dict
            Must have key "interval": (t0, tf).
        real_path : list
            List of [t, x, y, yaw, vel]
        pred_path : list
            List of predicted trajectories, each of shape (N+1, NX)
        real_cost : list or None
            History of scalar HJ values (raw), one per MPC step
        interp_cost : list or None
            History of scalar HJ values (interpolated), one per MPC step
        save_name : str or None
            If given, save frames + GIF with this name
        obstacles : list or None
            List of rectangles (ox, oy, ow, oh)
        interval_ms : int
            Animation frame interval in ms
        global_timeline : list or None
            One entry per global window ({window, target, sim_time_end, ...}),
            drawn on the map panel from the moment it became available.
        """
        real_path = np.asarray(real_path, dtype=float)
        pred_path = [np.asarray(p, dtype=float) for p in pred_path]
 
        if obstacles is None:
            obstacles = []
 
        if len(pred_path) == 0:
            raise ValueError("pred_path is empty")
 
        # real_path is often one longer than pred_path
        n_frames = min(len(pred_path), len(real_path))
 
        x_grid, y_grid, yaw_grid, vel_grid = self._hj_axes()
        xg, yg = np.meshgrid(x_grid, y_grid, indexing="ij")
 
        # matches the 3-period extension [raw, raw, raw] of the value functions
        yaw_grid_full = np.concatenate([
            yaw_grid - 2 * np.pi,
            yaw_grid,
            yaw_grid + 2 * np.pi,
        ])
 
        def _window_xy(entry):
            """x, y grids of the set in `entry`: it lives on its global window."""
            win = entry.get('window')
            if win is None:
                return x_grid, y_grid
            (wx0, wx1), (wy0, wy1) = win
            return (np.linspace(wx0, wx1, self.HJ_GRID_STP[0]),
                    np.linspace(wy0, wy1, self.HJ_GRID_STP[1]))

        current_set_idx = used_sets[0]
        _entry = verifier_timeline[current_set_idx]
        x_grid, y_grid = _window_xy(_entry)
        vf = np.load(_entry['path'])
        raw = vf
        vf = np.concatenate([raw, raw, raw], axis=3)   # periodic yaw
        t0_vf = _entry['interval'][0]
        vf_time_grid = t0_vf + np.arange(vf.shape[0]) * self.HJ_TIMESTEP
 
        yaw_min, yaw_max = self.HJ_GRID_MIN[2] - 2 * np.pi, self.HJ_GRID_MAX[2] + 2 * np.pi
        vel_min, vel_max = self.HJ_GRID_MIN[3], self.HJ_GRID_MAX[3]
        x_min, x_max = x_grid[0], x_grid[-1]
        y_min, y_max = y_grid[0], y_grid[-1]
 
        # finite-difference step sizes (half a grid cell)
        eps_x = (x_grid[1]   - x_grid[0])   * 0.5
        eps_y = (y_grid[1]   - y_grid[0])   * 0.5
        eps_fd = np.array([
            eps_x,
            eps_y,
            (yaw_grid[1] - yaw_grid[0]) * 0.5,
            (vel_grid[1] - vel_grid[0]) * 0.5,
        ])
 
        # layout: full-width map (top), three panels (middle), full-width cost (bottom)
        fig = plt.figure(figsize=(22, 17))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.0, 0.7])
        ax1 = fig.add_subplot(gs[0, :])   # whole map + global window + XY value function
        ax5 = fig.add_subplot(gs[1, 0])   # local zoom + -∇V arrows
        ax2 = fig.add_subplot(gs[1, 1])   # gradient slope per stage
        ax3 = fig.add_subplot(gs[1, 2])   # yaw-vel
        ax4 = fig.add_subplot(gs[2, :])   # cost over time
 
        # colorbar holder for ax2 — created once, never recreated
        _cbar_ax2 = None
 
        # ------------------------------------------------------------------
        # Helper: legend in a row under the panel, so it never covers the plot
        # ------------------------------------------------------------------
        def _legend_below(ax, ncol=3):
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                return
            # fixed offset in points (below tick labels and x label), whatever the panel height
            below = mtransforms.offset_copy(ax.transAxes, fig=fig, y=-34, units="points")
            ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                      bbox_transform=below, ncol=min(ncol, len(handles)), fontsize=8,
                      frameon=False)

        # ------------------------------------------------------------------
        # Helper: draw obstacle rectangles
        # ------------------------------------------------------------------
        def _draw_obstacles_local(ax):
            for ox, oy, ow, oh in obstacles:
                ax.add_patch(
                    plt.Rectangle(
                        (ox, oy), ow, oh,
                        linewidth=1.5,
                        edgecolor="white",
                        facecolor="gray",
                        alpha=0.6,
                        zorder=5,
                    )
                )

        # ------------------------------------------------------------------
        # Helper: draw the moving obstacle at its position for time t
        # ------------------------------------------------------------------
        def _draw_dynamic_obstacle(ax, t):
            first = True
            for dyn in self.DYN_OBS:
                for ox, oy, ow, oh in obstacle_rects(t, dyn):
                    ax.add_patch(
                        plt.Rectangle(
                            (ox, oy), ow, oh,
                            linewidth=1.5,
                            edgecolor="white",
                            facecolor="crimson",
                            alpha=0.7,
                            zorder=7,
                            label="dynamic obstacle" if first else None,
                        )
                    )
                    first = False

        # ------------------------------------------------------------------
        # Helper: global window active at time t (the latest one that has started,
        # i.e. t >= t0, and is available by t)
        # ------------------------------------------------------------------
        def _draw_global_window(ax, t):
            if not global_timeline:
                return
            active = [g for g in global_timeline if max(g['t0'], g['sim_time_end']) <= t + 1e-9]
            g = active[-1] if active else global_timeline[0]
            (wx0, wx1), (wy0, wy1) = g['window']
            ax.add_patch(plt.Rectangle(
                (wx0, wy0), wx1 - wx0, wy1 - wy0,
                fill=False, edgecolor="yellow", linestyle="--", linewidth=2.0,
                zorder=8, label=f"global window v{g['version']}",
            ))
            tx, ty, tw, th = g['target']
            ax.add_patch(plt.Rectangle(
                (tx, ty), tw, th,
                facecolor="lime", edgecolor="lime", alpha=0.35, linewidth=1.5,
                zorder=8, label="goal" if g['target_is_goal'] else "global target",
            ))

        # ------------------------------------------------------------------
        # Helper: heading arrow
        # ------------------------------------------------------------------
        def _draw_heading_arrow(ax, x, y, yaw, length=0.25, color="white", lw=2.0):
            s = self.SCALE
            dx = length * s * np.cos(yaw)
            dy = length * s * np.sin(yaw)
            ax.arrow(
                x, y, dx, dy,
                head_width=0.08 * s,
                head_length=0.10 * s,
                fc=color,
                ec=color,
                linewidth=lw,
                length_includes_head=True,
                zorder=8,
            )
 
        # ------------------------------------------------------------------
        # Helper: raw VF nearest-neighbour query (scalar)
        # ------------------------------------------------------------------
        def _query_raw(it_, state):
            ix_   = self._nearest_index(x_grid,        np.clip(state[0], x_min, x_max))
            iy_   = self._nearest_index(y_grid,        np.clip(state[1], y_min, y_max))
            iyaw_ = self._nearest_index(yaw_grid_full, self._wrap_yaw(state[2]))
            ivel_ = self._nearest_index(vel_grid,      np.clip(state[3], vel_min, vel_max))
            return float(vf[it_, ix_, iy_, iyaw_, ivel_])
 
        # ------------------------------------------------------------------
        # Helper: compute -∇V xy components at a single predicted point
        #         using raw VF central finite differences
        # ------------------------------------------------------------------
        def _neg_grad_xy(it_, pk, iyaw_k, ivel_k):
            ix_p = self._nearest_index(x_grid, np.clip(pk[0] + eps_x, x_min, x_max))
            ix_m = self._nearest_index(x_grid, np.clip(pk[0] - eps_x, x_min, x_max))
            iy_k = self._nearest_index(y_grid, np.clip(pk[1],         y_min, y_max))
            gx   = (vf[it_, ix_p, iy_k, iyaw_k, ivel_k] -
                    vf[it_, ix_m, iy_k, iyaw_k, ivel_k]) / (2.0 * eps_x)
 
            ix_k = self._nearest_index(x_grid, np.clip(pk[0],         x_min, x_max))
            iy_p = self._nearest_index(y_grid, np.clip(pk[1] + eps_y, y_min, y_max))
            iy_m = self._nearest_index(y_grid, np.clip(pk[1] - eps_y, y_min, y_max))
            gy   = (vf[it_, ix_k, iy_p, iyaw_k, ivel_k] -
                    vf[it_, ix_k, iy_m, iyaw_k, ivel_k]) / (2.0 * eps_y)
 
            return -gx, -gy   # negate → safe direction
 
        # ------------------------------------------------------------------
        # Helper: draw -∇V arrows on predicted trajectory
        # ------------------------------------------------------------------
        def _draw_grad_arrows(ax, it_, pred, scale=0.15):
            if pred is None or len(pred) == 0:
                return
            for k in range(len(pred)):
                pk     = pred[k]
                iyaw_k = self._nearest_index(yaw_grid_full, self._wrap_yaw(pk[2]))
                ivel_k = self._nearest_index(vel_grid, np.clip(pk[3], vel_min, vel_max))
                ngx, ngy = _neg_grad_xy(it_, pk, iyaw_k, ivel_k)
                mag = np.hypot(ngx, ngy)
                if mag < 1e-8:
                    continue
                dx = ngx / mag * scale * self.SCALE
                dy = ngy / mag * scale * self.SCALE
                ax.annotate(
                    "",
                    xy=(pk[0] + dx, pk[1] + dy),
                    xytext=(pk[0], pk[1]),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="cyan",
                        lw=1.5,
                    ),
                    zorder=9,
                )
 
        # ------------------------------------------------------------------
        # Panel 1/5 shared: XY heat map base
        # ------------------------------------------------------------------
        def _plot_xy_base(ax, Z, x_lim, y_lim, t):
            zmin, zmax = np.min(Z), np.max(Z)
            if zmax - zmin > 1e-10:
                ax.contourf(
                    x_grid, y_grid, Z.T,
                    levels=np.linspace(zmin, zmax, 50),
                    cmap="viridis",
                )
                ax.contour(
                    x_grid, y_grid, Z.T,
                    levels=np.linspace(zmin, zmax, 10),
                    colors="black",
                    linewidths=0.5,
                )
            else:
                ax.contourf(x_grid, y_grid, Z.T, levels=1, cmap="viridis")
            if zmin < 0 < zmax:
                ax.contour(x_grid, y_grid, Z.T,
                           levels=[0.0], colors="red", linewidths=2.0)
            _draw_obstacles_local(ax)
            _draw_dynamic_obstacle(ax, t)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
 
        # ------------------------------------------------------------------
        # Panel 1: raw XY + real path + prediction + -∇V arrows
        # ------------------------------------------------------------------
        global_xy_cache = {}

        def _draw_global_zero_level(ax, t):
            """Zero level of the global set the current local set was derived from, at time t
            (min over yaw, vel): the local set is contained in it."""
            version = verifier_timeline[current_set_idx].get('global_version')
            g = next((g for g in (global_timeline or []) if g['version'] == version), None)
            if g is None or 'xy_path' not in g:
                return
            if version not in global_xy_cache:
                global_xy_cache[version] = np.load(g['xy_path'])
            Zg_all = global_xy_cache[version]
            ig = int(np.clip(round((t - g['t0']) / self.HJ_TIMESTEP), 0, Zg_all.shape[0] - 1))
            Zg = Zg_all[ig]
            if not (Zg.min() < 0 < Zg.max()):
                return
            gx, gy = _window_xy(g)
            outline = [pe.withStroke(linewidth=3.0, foreground="black")]
            cs = ax.contour(gx, gy, Zg.T, levels=[0.0], colors="white",
                            linestyles="--", linewidths=1.5, zorder=5)
            cs.set(path_effects=outline)
            ax.plot([], [], color="white", linestyle="--", linewidth=1.5,
                    path_effects=outline, label=f"global V=0 (v{version})")

        def _current_crop():
            """PDE box of the local set in use: [(lo, hi)] for x, y, yaw, vel, or None."""
            entry = verifier_timeline[current_set_idx]
            return entry.get('crop'), entry.get('crop_pct')

        def _draw_crop_box_xy(ax):
            crop, pct = _current_crop()
            if not crop:
                return
            (bx0, bx1), (by0, by1) = crop[0], crop[1]
            ax.add_patch(plt.Rectangle(
                (bx0, by0), bx1 - bx0, by1 - by0,
                fill=False, edgecolor="magenta", linestyle="-.", linewidth=2.0, zorder=8,
                label=f"PDE box ({pct:.0f}% cells)" if pct is not None else "PDE box",
            ))

        def _draw_crop_box_yv(ax):
            crop, _ = _current_crop()
            if not crop or len(crop) < 4:
                return
            (a0, a1), (v0, v1) = crop[2], crop[3]
            # yaw is drawn on [-3pi, 3pi): repeat the box every 2pi (a cut yaw interval
            # may extend past +-pi, the copies show its wrap-around)
            for k, shift in enumerate((-2 * np.pi, 0.0, 2 * np.pi)):
                ax.add_patch(plt.Rectangle(
                    (a0 + shift, v0), a1 - a0, v1 - v0,
                    fill=False, edgecolor="magenta", linestyle="-.", linewidth=2.0, zorder=8,
                    label="PDE box" if k == 1 else None,
                ))

        def _plot_xy_panel(ax, Z, title, t, x, y, yaw, pred, it_):
            # whole map; the value function only covers the window of the current set
            _plot_xy_base(ax, Z, self.MAP_X, self.MAP_Y, t)
            _draw_global_window(ax, t)
            _draw_global_zero_level(ax, t)
            _draw_crop_box_xy(ax)
            ax.plot([], [], color="red", linewidth=2.0, label="local V=0")
 
            ax.plot(
                real_path[:frame_idx + 1, 1],
                real_path[:frame_idx + 1, 2],
                color="white", linewidth=2.0, zorder=6, label="real path",
            )
            ax.plot(x, y, "ro", markersize=7, zorder=7, label="current")
            _draw_heading_arrow(ax, x, y, yaw, color="white")
 
            if pred is not None and len(pred) > 0:
                ax.plot(
                    pred[:, 0], pred[:, 1],
                    color="orange", linestyle="--", linewidth=2.0,
                    zorder=6, label="prediction",
                )
                ax.plot(pred[0, 0], pred[0, 1], "yo", markersize=6, zorder=7)
 
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_title(title)
            _legend_below(ax, ncol=6)
 
        # ------------------------------------------------------------------
        # Panel 2: gradient slope per stage
        # ------------------------------------------------------------------
        def _plot_gradient_slope_panel(ax, t, pred, it_):
            nonlocal _cbar_ax2
 
            ax.set_title("Gradient Slope per Stage")
            if pred is None or len(pred) == 0:
                return
 
            dt_ctrl  = self.config['controller']['timestep']
            N_stages = len(pred)
            stages   = np.arange(N_stages)
            t_stages = t + stages * dt_ctrl
 
            # true V at every predicted state via raw nearest-neighbour
            V_true = np.array([_query_raw(it_, pred[k]) for k in range(N_stages)])
 
            # per-stage gradient (all 4 dims) via raw FD
            grads = np.zeros((N_stages, 4), dtype=float)
            for k in range(N_stages):
                xk = pred[k, :4].copy()
                for dim in range(4):
                    xp = xk.copy(); xp[dim] += eps_fd[dim]
                    xm = xk.copy(); xm[dim] -= eps_fd[dim]
                    grads[k, dim] = (_query_raw(it_, xp) - _query_raw(it_, xm)) / (2.0 * eps_fd[dim])
 
            cmap = plt.cm.plasma
            norm = plt.Normalize(vmin=0, vmax=max(N_stages - 2, 1))
 
            # true V: white line + coloured scatter
            ax.plot(stages, V_true,
                    color="white", linewidth=1.5, linestyle="-",
                    zorder=3, label="True V")
            ax.scatter(stages, V_true,
                       c=stages, cmap=cmap, norm=norm,
                       s=40, zorder=5, edgecolors="white", linewidths=0.5)
 
            # per-stage tangent line
            for k in range(N_stages - 1):
                delta_x     = pred[k + 1, :4] - pred[k, :4]
                V_slope_end = V_true[k] + grads[k] @ delta_x
                color       = cmap(norm(k))
                ax.plot([k, k + 1], [V_true[k], V_slope_end],
                        color=color, linewidth=2.5, zorder=4)
                ax.plot(k + 1, V_slope_end, "o",
                        color=color, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        zorder=6)
 
            ax.axhline(0.0, color="red", linestyle="--",
                       linewidth=1.0, label="V=0")
            ax.set_xlabel("MPC Stage  k")
            ax.set_ylabel("V value")
            ax.set_title(f"Gradient Slope per Stage  (V₀={V_true[0]:.3f})")
            _legend_below(ax)
            ax.grid(True, alpha=0.4)
 
            # colorbar: create only once
            if _cbar_ax2 is None:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                _cbar_ax2 = plt.colorbar(sm, ax=ax, label="Stage k", pad=0.02)
 
        # ------------------------------------------------------------------
        # Panel 3: yaw-vel
        # ------------------------------------------------------------------
        def _plot_yv_panel(ax, Z, t, x, y, yaw, vel, pred):
            zmin, zmax = np.min(Z), np.max(Z)
            if zmax - zmin > 1e-10:
                ax.contourf(
                    yaw_grid_full, vel_grid, Z.T,
                    levels=np.linspace(zmin, zmax, 50),
                    cmap="viridis",
                )
                ax.contour(
                    yaw_grid_full, vel_grid, Z.T,
                    levels=np.linspace(zmin, zmax, 10),
                    colors="black",
                    linewidths=0.5,
                )
            else:
                ax.contourf(yaw_grid_full, vel_grid, Z.T, levels=1, cmap="viridis")
 
            if zmin < 0 < zmax:
                ax.contour(yaw_grid_full, vel_grid, Z.T,
                           levels=[0.0], colors="red", linewidths=2.0)
 
            _draw_crop_box_yv(ax)

            ax.plot(yaw, vel, "ro", markersize=7, label="current")
 
            if pred is not None and len(pred) > 0:
                pred_yaw = self._wrap_yaw(pred[:, 2])
                pred_vel = pred[:, 3]
                ax.plot(
                    pred_yaw, pred_vel,
                    color="orange", linestyle="--", linewidth=2.0,
                    label="prediction",
                )
                ax.plot(pred_yaw[0], pred_vel[0], "yo", markersize=6)
 
            ax.set_xlim(yaw_min, yaw_max)
            ax.set_ylim(vel_min, vel_max)
            ax.set_xlabel("Theta")
            ax.set_ylabel("Velocity")
            ax.set_title("Value Function (Theta vs V)")
            ax.grid(True)
            _legend_below(ax)
 
        # ------------------------------------------------------------------
        # Panel 5: local zoom + -∇V arrows
        # ------------------------------------------------------------------
        def _plot_local_panel(ax, Z, t, x, y, yaw, pred, it_):
            window = 0.3 * self.SCALE
            lx_min, lx_max = x - window, x + window
            ly_min, ly_max = y - window, y + window
 
            _plot_xy_base(ax, Z, (lx_min, lx_max), (ly_min, ly_max), t)
 
            ax.plot(x, y, "ro", markersize=8, zorder=7, label="current")
 
            if pred is not None and len(pred) > 0:
                ax.plot(
                    pred[:, 0], pred[:, 1],
                    color="orange", linestyle="--", linewidth=1.5,
                    zorder=6, label="prediction",
                )
                ax.plot(pred[0, 0], pred[0, 1], "yo", markersize=6, zorder=7)
                _draw_grad_arrows(ax, it_, pred, scale=0.12)
 
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"-∇V safe direction  (x={x:.2f}, y={y:.2f})")
            _legend_below(ax)
            ax.grid(True, alpha=0.3)
 
        # ------------------------------------------------------------------
        # Save directory
        # ------------------------------------------------------------------
        if save_name is not None:
            if os.path.exists(save_name):
                shutil.rmtree(save_name)
            os.mkdir(save_name)
 
        pbar = tqdm(total=n_frames, desc="Animating frames")
 
        # ------------------------------------------------------------------
        # Update function
        # ------------------------------------------------------------------
        def update(frame_idx_local):
            nonlocal frame_idx, current_set_idx, vf, vf_time_grid, t0_vf
            nonlocal x_grid, y_grid, x_min, x_max, y_min, y_max
            frame_idx = frame_idx_local

            sid = used_sets[frame_idx] if frame_idx < len(used_sets) else used_sets[-1]
            if sid != current_set_idx:
                current_set_idx = sid
                _entry = verifier_timeline[sid]
                x_grid, y_grid = _window_xy(_entry)
                x_min, x_max = x_grid[0], x_grid[-1]
                y_min, y_max = y_grid[0], y_grid[-1]
                vf = np.load(_entry['path'])
                raw = vf
                vf = np.concatenate([raw, raw, raw], axis=3)   # periodic yaw
                t0_vf = _entry['interval'][0]
                vf_time_grid = t0_vf + np.arange(vf.shape[0]) * self.HJ_TIMESTEP
 
            for ax in (ax1, ax2, ax3, ax4, ax5):
                ax.clear()
 
            t, x, y, yaw, vel = real_path[frame_idx]
            yaw  = self._wrap_yaw(yaw)
            pred = pred_path[frame_idx] if frame_idx < len(pred_path) else None
 
            # nearest indices for raw VF slices
            it   = self._nearest_index(vf_time_grid, t)
            ix   = self._nearest_index(x_grid,        x)
            iy   = self._nearest_index(y_grid,        y)
            iyaw = self._nearest_index(yaw_grid_full, yaw)
            ivel = self._nearest_index(vel_grid,      vel)
 
            # raw XY slice at the car's current yaw and vel
            Z_xy_real = vf[it, :, :, iyaw, ivel]
            # XY projection at time t: min over yaw and vel, i.e. the union
            # over all headings/speeds (same as pyspect's impl.plot(axes=('x','y')))
            Z_xy_proj = vf[it].min(axis=(2, 3))
 
            # ----------------------------------------------------------------
            # 1) Projected XY + -∇V arrows on predicted trajectory
            # ----------------------------------------------------------------
            _plot_xy_panel(ax1, Z_xy_proj,
                           "V at t, min over (yaw, vel)",
                           t, x, y, yaw, pred, it)
 
            # ----------------------------------------------------------------
            # 2) Gradient slope per stage
            # ----------------------------------------------------------------
            _plot_gradient_slope_panel(ax2, t, pred, it)
 
            # ----------------------------------------------------------------
            # 3) Yaw-vel slice
            # ----------------------------------------------------------------
            Z_yv_real = vf[it, ix, iy, :, :]
            _plot_yv_panel(ax3, Z_yv_real, t, x, y, yaw, vel, pred)
 
            # ----------------------------------------------------------------
            # 4) Time vs cost
            # ----------------------------------------------------------------
            cost_arr = np.asarray([real_cost, interp_cost], dtype=float).T
            if len(cost_arr) > 0:
                time_axis = np.arange(len(cost_arr)) * self.config['controller']['timestep']
                ax4.plot(time_axis, cost_arr[:, 0],
                         color="steelblue", linewidth=1.5, label="Real Cost")
                ax4.plot(time_axis, cost_arr[:, 1],
                         color="coral",     linewidth=1.5, label="Interpolated Cost")
                ax4.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
                if frame_idx < len(cost_arr):
                    ax4.axvline(time_axis[frame_idx],
                                color="red", linestyle="--", linewidth=1.5,
                                label=f"t = {time_axis[frame_idx]:.2f} s")
                    ax4.plot(time_axis[frame_idx], cost_arr[frame_idx, 0],
                             "ro", markersize=7)
                    ax4.plot(time_axis[frame_idx], cost_arr[frame_idx, 1],
                             "ro", markersize=7)
                ax4.set_xlabel("Time (s)")
                ax4.set_ylabel("HJ Value V(x)")
                ax4.set_title("Time vs HJ Cost  (V ≤ 0 = safe)")
                _legend_below(ax4)
                ax4.grid(True, alpha=0.4)
                ax4.set_ylim(-2, 2)
 
            # ----------------------------------------------------------------
            # 5) Local zoom: current point + -∇V arrows
            # ----------------------------------------------------------------
            _plot_local_panel(ax5, Z_xy_proj, t, x, y, yaw, pred, it)
 
            fig.suptitle(f"t = {t:.2f}", fontsize=14)
            fig.tight_layout()
 
            if save_name is not None:
                frame_file = os.path.join(save_name, f"frame_{frame_idx:04d}.png")
                fig.savefig(frame_file, dpi=80)
 
            pbar.update(1)
            return []
 
        frame_idx = 0
        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
 
        if save_name is not None:
            anim.save(f"{save_name}.gif",
                      writer=PillowWriter(fps=max(1, int(1000 / interval_ms))))
 
        pbar.close()
        return fig, (ax1, ax2, ax3, ax4, ax5), anim
    
    def plot_timeline(self, save_path: str = "timeline.png") -> None:
        """
        Timeline of the global loop, the local loop and the MPC, on one simulation-time axis.

        Lanes (top to bottom)
            Global loop   compute   when each global window was computed (labelled vN)
                          horizons  the time interval each global set covers, [t0, t0 + HJ_HORIZON],
                                    with its spatial window and target
            Local loop    compute   when each local set was computed (labelled sN)
                          horizons  the time interval each local set covers
            MPC           TTR map   which global window the progress (TTR) cost refers to
                          set used  which local set the controller was using
        Dashed vertical lines mark the moment each new global window becomes available.

        Registry keys consumed
            verifier.global_timeline : one entry per global window
                {version, t0, window, target_is_goal, compute_start, compute_end, sim_time_end}
            verifier.timeline : one entry per local set
                {set_idx, interval, compute_start, compute_end, sim_start, sim_end, global_version}
            controller.timeline : {mpc_dt, used_sets, used_ttr}
                (used_sets[k]: set_idx, used_ttr[k]: global version of the TTR map at MPC step k)
            mission.done (optional) : {time, ...}
        """
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors

        gtl = self.registry.get('verifier.global_timeline', [])
        vtl = self.registry.get('verifier.timeline', [])
        ctl = self.registry.get('controller.timeline', {})
        done = self.registry.get('mission.done', None)
        if not vtl or not ctl:
            log("Animation", "no verifier/controller timeline in the registry: timeline not plotted")
            return

        # Palette (validated categorical slots 1-3, light mode) and text/grid tokens
        C_GLOBAL, C_LOCAL, C_MPC = "#2a78d6", "#eb6834", "#1baf7a"
        INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"

        def tint(hex_color, a=0.30):
            # opaque light tint (color blended on the surface), so grid lines stay behind it
            c, sfc = np.array(mcolors.to_rgb(hex_color)), np.array(mcolors.to_rgb(SURFACE))
            return tuple(a * c + (1 - a) * sfc)

        mpc_dt = ctl['mpc_dt']
        used_sets = list(ctl['used_sets'])
        t_end = len(used_sets) * mpc_dt
        if done is not None:
            t_end = min(t_end, done['time'])
        t_max = max(t_end, max(e['interval'][1] for e in vtl))

        H_GLOBAL = self.config['verifier']['horizon']

        # Global horizons overlap by more than two at a time when HJ_HORIZON > 2 * the update
        # period: each one goes on the first sub-row that is free at its start.
        g_row, row_end = [], []
        for g in gtl:
            r = next((i for i, e in enumerate(row_end) if e <= g['t0'] + 1e-9), len(row_end))
            if r == len(row_end):
                row_end.append(0.0)
            row_end[r] = g['t0'] + H_GLOBAL
            g_row.append(r)
        n_g_rows = max(len(row_end), 1)

        # Lane layout (y grows upwards); the horizon lanes have staggered sub-rows
        BAR_H, GAP = 0.6, 0.02 * t_max / 30   # GAP: 2px-like surface gap between adjacent fills
        ROW = 0.7                              # spacing of the horizon sub-rows
        g_hor = tuple(5.9 + ROW * (n_g_rows - 1 - i) for i in range(n_g_rows))
        LANES = {
            'g_comp': g_hor[0] + 1.0, 'g_hor': g_hor,
            'l_comp': 4.4, 'l_hor': (3.4, 2.7),
            'mpc_ttr': 1.35, 'mpc': 0.65,
        }
        y_top = LANES['g_comp'] + 0.8
        MIN_W = 0.06   # minimum drawn width (s) so instantaneous computations stay visible

        fig_w = max(14.0, min(40.0, t_max * 0.45))
        fig, ax = plt.subplots(figsize=(fig_w, 6.5 + 0.55 * (y_top - 8.0)))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)

        def bar(t0, t1, y, color, alpha=1.0, h=BAR_H, label=None, edge=None, label_above=False):
            w = max(t1 - t0, MIN_W)
            ax.add_patch(mpatches.FancyBboxPatch(
                (t0 + GAP / 2, y - h / 2), max(w - GAP, MIN_W), h,
                boxstyle="round,pad=0,rounding_size=0.04", linewidth=1.0 if edge else 0,
                edgecolor=edge or "none", facecolor=color, alpha=alpha, zorder=3))
            if label is None:
                return
            # does the text fit inside the bar? (rough width: ~0.6 em per character at 7.5 pt,
            # converted to seconds with the axes scale)
            def text_w(s):
                return (len(s) * 0.6 * 7.5 / 72 + 0.08) * t_max / (0.85 * fig_w)
            if w <= text_w(label):
                label = label.split("  ")[0]   # too long: keep only the name (e.g. "v13")
            if w > text_w(label):
                ax.text(t0 + w / 2, y, label, ha="center", va="center",
                        fontsize=7.5, color=INK, zorder=4, clip_on=True)
            elif label_above:
                # bar too narrow for the text: put the name just above it
                ax.text(t0, y + h / 2 + 0.04, label, ha="left", va="bottom",
                        fontsize=7, color=INK, zorder=4, clip_on=True)

        # --- Global loop -------------------------------------------------
        for k, g in enumerate(gtl):
            # computed in advance (background thread), used from its window start t0
            bar(g.get('sim_time_start', g['t0']), g['sim_time_end'], LANES['g_comp'], C_GLOBAL,
                label=f"v{g['version']}", label_above=True)
            t_on = max(g['t0'], g['sim_time_end'])
            (x0, x1), _ = g['window']
            tgt = "GOAL" if g['target_is_goal'] else "band"
            space = f"x∈[{x0:.1f}, {x1:.1f}] → {tgt}"
            # time interval covered by the global set (may extend past the plot: clipped),
            # with the spatial window it considers and its target
            h0, h1 = g['t0'], g['t0'] + H_GLOBAL
            t_txt = f"[{h0:.1f}, {h1:.1f}] s" if h1 <= t_max else f"→ {h1:.1f} s"
            bar(h0, min(h1, t_max), LANES['g_hor'][g_row[k]], tint(C_GLOBAL), edge=C_GLOBAL, h=0.5,
                label=f"v{g['version']}  {t_txt}   {space}")
            # global window starts being used by the local sets: dashed line across all lanes
            ax.axvline(t_on, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)

        # --- Local loop --------------------------------------------------
        for e in vtl:
            k = e['set_idx']
            s0 = e.get('sim_start', e['interval'][0])
            s1 = e.get('sim_end', s0)
            bar(s0, s1, LANES['l_comp'], C_LOCAL, label=f"s{k}", label_above=True)
            h0, h1 = e['interval']
            bar(h0, h1, LANES['l_hor'][k % 2], tint(C_LOCAL), edge=C_LOCAL, h=0.5, label=f"s{k}")

        # --- MPC: consecutive steps with the same set / map -> one segment
        def segments(values):
            seg_start = 0
            for k in range(1, len(values) + 1):
                if k == len(values) or values[k] != values[seg_start]:
                    yield values[seg_start], seg_start * mpc_dt, min(k * mpc_dt, t_end)
                    seg_start = k

        for sid, a, b in segments(used_sets):
            bar(a, b, LANES['mpc'], C_MPC, h=0.5, label=f"s{sid}" if sid >= 0 else "-")

        # progress map: global version of the TTR cost, and the global of the set in use
        # (they coincide by construction; a mismatch is drawn in red)
        used_ttr = list(ctl.get('used_ttr', []))
        set_gv = {e['set_idx']: e.get('global_version') for e in vtl}
        for k, (v, a, b) in enumerate(segments(used_ttr)):
            if v < 0:
                continue
            steps = range(int(round(a / mpc_dt)), int(round(b / mpc_dt)))
            mismatch = any(set_gv.get(used_sets[i]) not in (None, v) for i in steps if i < len(used_sets))
            bar(a, b, LANES['mpc_ttr'], tint(C_GLOBAL), edge="#d03b3b" if mismatch else C_GLOBAL, h=0.5,
                label=f"TTR v{v}" + ("  ≠ set" if mismatch else ""))

        if done is not None:
            ax.axvline(done['time'], color=INK, lw=1.4, zorder=2)
            ax.text(done['time'], y_top - 0.25, " goal reached", ha="left", va="center", fontsize=8, color=INK)

        # --- Axes, lane labels, summary ----------------------------------
        ax.set_xlim(0, t_max)
        ax.set_ylim(0.2, y_top)
        ax.set_yticks([LANES['g_comp'], float(np.mean(LANES['g_hor'])),
                       LANES['l_comp'], sum(LANES['l_hor']) / 2, LANES['mpc_ttr'], LANES['mpc']])
        ax.set_yticklabels(["Global loop · compute",
                            "Global loop · set horizons",
                            "Local loop · compute", "Local loop · set horizons",
                            "MPC · progress map (TTR)", "MPC · set in use"], fontsize=9, color=INK)
        for sep in (5.2, 2.0):   # separate global / local / MPC groups
            ax.axhline(sep, color=GRID, lw=1.0, zorder=0)
        tick = 1.0 if t_max <= 20 else 2.0 if t_max <= 40 else 5.0
        ax.set_xticks(np.arange(0, t_max + 1e-9, tick))
        ax.grid(axis='x', color=GRID, lw=0.6, zorder=0)
        ax.tick_params(axis='x', colors=INK2, labelsize=8)
        ax.tick_params(axis='y', length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_xlabel("simulation time (s)", color=INK2, fontsize=9)

        def stats(entries, a, b):
            d = [x[b] - x[a] for x in entries]
            return f"mean {np.mean(d):.2f} s, max {np.max(d):.2f} s" if d else "n/a"
        # overlap of consecutive global horizons vs the time span of one local set
        overlaps = [g['t0'] + H_GLOBAL - n['t0'] for g, n in zip(gtl, gtl[1:])]
        local_span = self.config['verifier'].get('t_delay', 0.0) + self.T_PRED
        overlap_txt = (f"min global overlap {min(overlaps):.2f} s (local set needs {local_span:.2f} s)"
                       if overlaps else "")
        summary = (f"{len(gtl)} global windows (compute {stats(gtl, 'compute_start', 'compute_end')})   ·   "
                   f"{len(vtl)} local sets (compute {stats(vtl, 'compute_start', 'compute_end')})"
                   + (f"   ·   {overlap_txt}" if overlap_txt else ""))
        ax.set_title("Global loop, local loop and MPC timeline", loc="left",
                     fontsize=12, fontweight="bold", color=INK, pad=22)
        ax.text(0, 1.02, summary + "   ·   compute times are wall-clock",
                transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        log("Animation", f"timeline saved to {save_path}")
        plt.close(fig)

    def run(self):

        while self.registry.get("results") is None:
            if self.stop_event.is_set():
                return
            # print("Waiting for results from controller...")
            tm.sleep(0.1)


        results = self.registry.get("results")
        verifier_timeline = self.registry.get('verifier.timeline', [])
        ctl = self.registry.get('controller.timeline', {})
        used_sets = ctl['used_sets']

        self.animate_heat_map_global(
            verifier_timeline=verifier_timeline,
            used_sets=used_sets,
            real_path=results["real_path"],
            pred_path=results["pred_path"],
            real_cost=results["real_cost"],
            interp_cost=results["interp_cost"],
            save_name="vf_debug",
            obstacles=self.OBSTACLES,
            interval_ms=100,
            global_timeline=self.registry.get('verifier.global_timeline', []),
        )

        self.plot_timeline(save_path="timeline.png")
    
    def close(self):
        pass