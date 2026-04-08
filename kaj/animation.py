import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time as tm
import os
import shutil

from tqdm import tqdm

from controller import Controller

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
        yaw_grid = np.linspace(self.HJ_GRID_MIN[2], self.HJ_GRID_MAX[2], self.HJ_GRID_STP[2])
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
 
        yaw_grid_full = np.concatenate([
            yaw_grid[:-1] - 2 * np.pi,
            yaw_grid,
            yaw_grid[1:] + 2 * np.pi,
        ])
 
        current_set_idx = used_sets[0]
        _entry = verifier_timeline[current_set_idx]
        vf = np.load(_entry['path'])
        raw = vf
        vf = np.concatenate([raw, raw[:,:,:,1:], raw[:,:,:,1:]], axis=3)
        t0_vf = _entry['interval'][0]
        vf_time_grid = t0_vf + np.arange(vf.shape[0]) * self.HJ_TIMESTEP
 
        yaw_min, yaw_max = self.HJ_GRID_MIN[2] - 2 * np.pi, self.HJ_GRID_MAX[2] + 2 * np.pi
        vel_min, vel_max = self.HJ_GRID_MIN[3], self.HJ_GRID_MAX[3]
        x_min, x_max = self.HJ_GRID_MIN[0], self.HJ_GRID_MAX[0]
        y_min, y_max = self.HJ_GRID_MIN[1], self.HJ_GRID_MAX[1]
 
        # finite-difference step sizes (half a grid cell)
        eps_x = (x_grid[1]   - x_grid[0])   * 0.5
        eps_y = (y_grid[1]   - y_grid[0])   * 0.5
        eps_fd = np.array([
            eps_x,
            eps_y,
            (yaw_grid[1] - yaw_grid[0]) * 0.5,
            (vel_grid[1] - vel_grid[0]) * 0.5,
        ])
 
        # layout: 2 rows × 4 cols (top), full-width cost (bottom)
        fig = plt.figure(figsize=(26, 12))
        ax1 = fig.add_subplot(2, 4, 1)   # raw XY
        ax5 = fig.add_subplot(2, 4, 2)   # local zoom + -∇V arrows
        ax2 = fig.add_subplot(2, 4, 3)   # gradient slope per stage
        ax3 = fig.add_subplot(2, 4, 4)   # yaw-vel
        ax4 = fig.add_subplot(2, 1, 2)   # cost over time
 
        # colorbar holder for ax2 — created once, never recreated
        _cbar_ax2 = None
 
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
        # Helper: heading arrow
        # ------------------------------------------------------------------
        def _draw_heading_arrow(ax, x, y, yaw, length=0.25, color="white", lw=2.0):
            dx = length * np.cos(yaw)
            dy = length * np.sin(yaw)
            ax.arrow(
                x, y, dx, dy,
                head_width=0.08,
                head_length=0.10,
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
                dx = ngx / mag * scale
                dy = ngy / mag * scale
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
        def _plot_xy_base(ax, Z, x_lim, y_lim):
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
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
 
        # ------------------------------------------------------------------
        # Panel 1: raw XY + real path + prediction + -∇V arrows
        # ------------------------------------------------------------------
        def _plot_xy_panel(ax, Z, title, t, x, y, yaw, pred, it_):
            _plot_xy_base(ax, Z, (x_min, x_max), (y_min, y_max))
 
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
            ax.legend(loc="upper right", fontsize=8)
 
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
            ax.legend(loc="upper right", fontsize=8)
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
            ax.legend(loc="upper right", fontsize=8)
 
        # ------------------------------------------------------------------
        # Panel 5: local zoom + -∇V arrows
        # ------------------------------------------------------------------
        def _plot_local_panel(ax, Z, t, x, y, yaw, pred, it_):
            window = 0.3
            lx_min, lx_max = x - window, x + window
            ly_min, ly_max = y - window, y + window
 
            _plot_xy_base(ax, Z, (lx_min, lx_max), (ly_min, ly_max))
 
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
            ax.legend(loc="upper right", fontsize=8)
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
            nonlocal frame_idx, current_set_idx, vf, vf_time_grid
            frame_idx = frame_idx_local

            sid = used_sets[frame_idx] if frame_idx < len(used_sets) else used_sets[-1]
            if sid != current_set_idx:
                current_set_idx = sid
                _entry = verifier_timeline[sid]
                vf = np.load(_entry['path'])
                raw = vf
                vf = np.concatenate([raw, raw[:,:,:,1:], raw[:,:,:,1:]], axis=3)
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
 
            # shared raw XY slice
            Z_xy_real = vf[it, :, :, iyaw, ivel]
 
            # ----------------------------------------------------------------
            # 1) Raw XY + -∇V arrows on predicted trajectory
            # ----------------------------------------------------------------
            _plot_xy_panel(ax1, Z_xy_real,
                           "Raw Value Function (X vs Y)",
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
                time_axis = np.arange(len(cost_arr)) * self.HJ_TIMESTEP
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
                ax4.legend(loc="upper right", fontsize=8)
                ax4.grid(True, alpha=0.4)
                ax4.set_ylim(-2, 2)
 
            # ----------------------------------------------------------------
            # 5) Local zoom: current point + -∇V arrows
            # ----------------------------------------------------------------
            _plot_local_panel(ax5, Z_xy_real, t, x, y, yaw, pred, it)
 
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
        Plot the Verification Loop & MPC Loop Timeline using data from the registry.

        Registry keys consumed
        ----------------------
        verifier.timeline : list of dicts, one per published set
            {set_idx, interval, compute_start, compute_end, loop_end}
            All times are absolute wall-clock (tm.time()).

        controller.timeline : dict
            {mpc_dt: float, used_sets: list[int]}
            used_sets[k] is the set_idx active at MPC step k.
        """
        import matplotlib.patches as mpatches

        vtl = self.registry.get('verifier.timeline', [])
        ctl = self.registry.get('controller.timeline', {})

        if not vtl:
            print("[plot_timeline] No verifier.timeline in registry – nothing to plot.")
            return
        if not ctl:
            print("[plot_timeline] No controller.timeline in registry – nothing to plot.")
            return

        # ------------------------------------------------------------------
        # Extract verifier data, normalise to t=0 at first compute_start
        # ------------------------------------------------------------------
        t_origin = vtl[0]['compute_start']

        n_sets             = len(vtl)
        horizons           = [entry['interval']                         for entry in vtl]
        set_compute        = [entry['compute_end'] - entry['compute_start'] for entry in vtl]
        verify_start_times = [entry['compute_start'] - t_origin         for entry in vtl]
        verify_compute_end = [entry['compute_end']   - t_origin         for entry in vtl]
        verify_loop_end    = [entry['loop_end']      - t_origin         for entry in vtl]

        # ------------------------------------------------------------------
        # Extract controller data
        # ------------------------------------------------------------------
        mpc_dt    = ctl['mpc_dt']
        used_sets = np.asarray(ctl['used_sets'], dtype=int)
        total_steps = len(used_sets)
        total_time  = total_steps * mpc_dt

        # MPC starts as soon as the first set is ready
        mpc_start = verify_compute_end[0]

        # MPC time ranges per set
        mpc_segments = []
        for sid in range(n_sets):
            idxs = np.where(used_sets == sid)[0]
            if len(idxs) == 0:
                continue
            t0_seg = mpc_start + idxs[0]  * mpc_dt
            t1_seg = mpc_start + (idxs[-1] + 1) * mpc_dt
            mpc_segments.append((sid, t0_seg, t1_seg))

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        COLORS = ["#4472C4", "#ED7D31", "#FFC000", "#70AD47"]
        BAR_H          = 0.55
        HORIZON_STRIDE = 0.65
        TOP_PAD        = 2.0
        BOT_PAD        = 2.2
        EXTRA_PAD      = 1.2

        horizon_block_h = n_sets * HORIZON_STRIDE
        total_h         = TOP_PAD + horizon_block_h + BOT_PAD + 2 * EXTRA_PAD

        ROW_MPC    = EXTRA_PAD
        ROW_LABEL  = ROW_MPC   + BOT_PAD + (horizon_block_h / 2)
        ROW_VERIFY = ROW_LABEL + (horizon_block_h / 2) + TOP_PAD

        horizon_y_offsets = [
            i * (-HORIZON_STRIDE) + (horizon_block_h - HORIZON_STRIDE) / 2
            for i in range(n_sets)
        ]

        x_max_plot = max(verify_loop_end[-1], total_time + mpc_start) + 1
        fig_w = max(18, int(x_max_plot * 0.5))
        fig, ax = plt.subplots(figsize=(fig_w, total_h))
        ax.set_xlim(-0.5, x_max_plot)
        ax.set_ylim(EXTRA_PAD - 0.8, ROW_VERIFY + 1.2)
        ax.axis("off")

        # Row labels
        ax.text(-0.3, ROW_VERIFY, "Verification\nLoop",
                ha="right", va="center", fontsize=10, fontweight="bold")
        ax.text(-0.3, ROW_MPC, "MPC Loop",
                ha="right", va="center", fontsize=10, fontweight="bold")
        ax.plot([-0.3, -0.05], [ROW_VERIFY, ROW_VERIFY], "k--", lw=1.5)
        ax.plot([-0.3, -0.05], [ROW_MPC,    ROW_MPC   ], "k--", lw=1.5)

        # Verification Loop bars
        for k in range(n_sets):
            t0  = verify_start_times[k]
            t_c = verify_compute_end[k]
            t1  = verify_loop_end[k]
            color = COLORS[k % len(COLORS)]

            ax.add_patch(mpatches.FancyBboxPatch(
                (t0, ROW_VERIFY - BAR_H / 2), set_compute[k], BAR_H,
                boxstyle="round,pad=0.05", linewidth=1.2,
                edgecolor="white", facecolor=color,
            ))
            ax.text(t0 + set_compute[k] / 2, ROW_VERIFY,
                    f"Compute\n{set_compute[k]:.1f}s",
                    ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")

            sleep_dur = t1 - t_c
            if sleep_dur > 0.05:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (t_c, ROW_VERIFY - BAR_H / 2), sleep_dur, BAR_H,
                    boxstyle="round,pad=0.05", linewidth=1.2,
                    edgecolor="white", facecolor="#1a1a1a",
                ))
                ax.text(t_c + sleep_dur / 2, ROW_VERIFY, "sleep",
                        ha="center", va="center", fontsize=7.5, color="white")

        # Horizon bars (middle)
        for k, (h0, h1) in enumerate(horizons):
            color  = COLORS[k % len(COLORS)]
            y_base = ROW_LABEL + horizon_y_offsets[k]
            abs_h0 = mpc_start + h0
            abs_h1 = mpc_start + h1
            ax.add_patch(mpatches.FancyBboxPatch(
                (abs_h0, y_base - 0.22), abs_h1 - abs_h0, 0.44,
                boxstyle="round,pad=0.03", linewidth=1,
                edgecolor="white", facecolor=color, alpha=0.92,
            ))
            label = (f"verification set for {h0:.0f}s to {h1:.0f}s"
                     if h1 - h0 > 1 else f"set to {h1:.0f}s")
            ax.text((abs_h0 + abs_h1) / 2, y_base, label,
                    ha="center", va="center", fontsize=8, color="white", fontweight="bold")

        # Arrows: verification → horizon
        for k in range(n_sets):
            t_pub  = verify_compute_end[k]
            color  = COLORS[k % len(COLORS)]
            y_base = ROW_LABEL + horizon_y_offsets[k]
            ax.annotate("", xy=(t_pub, y_base + 0.22),
                        xytext=(t_pub, ROW_VERIFY - BAR_H / 2 - 0.05),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

        # Arrows: horizon → MPC
        for k in range(n_sets):
            t_pub  = verify_compute_end[k]
            y_base = ROW_LABEL + horizon_y_offsets[k]
            ax.annotate("", xy=(t_pub, ROW_MPC + BAR_H / 2 + 0.05),
                        xytext=(t_pub, y_base - 0.22),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

        # MPC Loop bars
        for sid, t0_seg, t1_seg in mpc_segments:
            color = COLORS[sid % len(COLORS)]
            ax.add_patch(mpatches.FancyBboxPatch(
                (t0_seg, ROW_MPC - BAR_H / 2), t1_seg - t0_seg, BAR_H,
                boxstyle="round,pad=0.05", linewidth=1.2,
                edgecolor="white", facecolor=color,
            ))
            ax.text((t0_seg + t1_seg) / 2, ROW_MPC, "MPC LOOP",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")

        # Time axis
        ax_time = ax.inset_axes([0, -0.08, 1, 0.06])
        ax_time.set_xlim(ax.get_xlim())
        ax_time.set_ylim(0, 1)
        ax_time.axis("off")

        x_min_ax, x_max_ax = ax.get_xlim()
        tick_step = 5.0 if (x_max_ax - x_min_ax) > 30 else 1.0
        ticks = np.arange(0, x_max_ax + tick_step, tick_step)
        for t in ticks:
            ax_time.axvline(t, color="gray", lw=0.5, ymin=0, ymax=0.4)
            rel_t = t - mpc_start
            if rel_t >= 0:
                ax_time.text(t, -0.2, f"{rel_t:.0f}",
                             ha="center", va="top", fontsize=8, color="gray")
            ax.axvline(t, color="gray", lw=0.4, ls="--", alpha=0.4)

        plt.suptitle("Verification Loop & MPC Loop Timeline",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_timeline] Saved to {save_path}")
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
        )

        self.plot_timeline(save_path="timeline.png")
    
    def close(self):
        pass