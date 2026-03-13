#! /usr/bin/env python3
"""
4x4 Corridor Scenario (5D Bicycle model)
 
Implements:
1. A 4x4 map with two rectangular obstacles.
2. A Temporal Logic Tree task: avoid all obstacles until the goal region is reached.
3. Backward reachable tube (BRT) computation and export to a NumPy file.
 
State: [x, y, yaw, delta, vel] — position, heading, steering angle, speed.
"""
import sys
import os
import jax

jax.config.update("jax_platform_name", "gpu")
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "./pyspect_test"))
src_path = os.path.join(parent_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# PYSPECT IMPORTS
from pyspect import *
from pyspect.primitives import *
from pyspect.tlt import *
from pyspect.set_builder import *
from pyspect.systems.hj_reachability.bicycle5d import Bicycle5D
from pyspect.impls.hj_reachability import TVHJImpl
from pyspect.plotting.levelsets import plot3D_levelset

from functools import reduce
from math import pi
import numpy as np
import time as tm

class CorridorScenario:
    """4x4 corridor BRT scenario for a 5D bicycle model."""
 
    # ── Map bounds ──
    X0, XN = 0.0, 4.0
    Y0, YN = 0.0, 4.0
    dt = 0.2
 
    # ── Static obstacles ──
    LOWER_OBS = dict(x=0.0, y=0.0, w=2.0, h=2.0)
    UPPER_OBS  = dict(x=1.2, y=2.8, w=2.8, h=0.4)
 
    # ── Goal region ──
    GOAL = dict(x=3.58, y=3.46, w=0.30, h=0.36)
 
    # ── Car geometry & initial state ──
    CAR_W, CAR_H = 0.25, 0.40
    STATE_INIT = np.array([3.00, 0.32, pi / 2, 0.0, 0.5])
 
    # ── Moving obstacle parameters ──
    MOV_SPEED    = 0.3
    MOV_WIDTH    = 0.4
    MOV_CENTER_X = 2.0
    MOV_TIMES    = [0.1, 5.1, 10.1, 15.1, 19.1]
 
    # ── Dynamics limits ──
    MAX_STEER      = 0.7
    MAX_STEER_RATE = np.pi
    MAX_ACCEL      = 0.4
    MAX_VEL        = 0.8
    WHEELBASE      = 0.32
    T_MAX          = 20
 
    def __init__(self):
        self._build_sets()
        self._build_task()
        self._build_impl()
 
    # ──────────────────────────────────────────────
    # 1) Build map sets
    # ──────────────────────────────────────────────
 
    def _build_sets(self):
        lo = self.LOWER_OBS
        uo = self.UPPER_OBS
        g  = self.GOAL
 
        self.target_set = BoundedSet(
            x=(g["x"], g["x"] + g["w"]),
            y=(g["y"], g["y"] + g["h"]),
        )
 
        # Map boundary (slightly inset)
        self.map_box = BoundedSet(
            x=(self.X0 + 0.01, self.X0 + self.XN - 0.01),
            y=(self.Y0 + 0.01, self.Y0 + self.YN - 0.01),
        )
 
        lower_obstacle = AND(
            AND(self.map_box, BoundedSet(x=(..., lo["x"] + lo["w"]))),
            BoundedSet(y=(..., lo["y"] + lo["h"])),
        )
        upper_obstacle = AND(
            AND(self.map_box, BoundedSet(x=(uo["x"], ...))),
            BoundedSet(y=(uo["y"], uo["y"] + uo["h"])),
        )
        self.obstacles       = OR(lower_obstacle, upper_obstacle)
        self.moving_obstacle = self._build_moving_obstacle()
        self.safe_region     = AND(self.map_box, NOT(OR(self.obstacles, self.moving_obstacle)))
 
        self._print_map_info()
 
    def _build_moving_obstacle(self):
        _v = self.MOV_SPEED
        _w = self.MOV_WIDTH
        _x = self.MOV_CENTER_X
 
        boundary        = BoundedSet(x=(_x - _w / 2, _x + _w / 2), y=(3.0, ...))
        moving_obstacle = EMPTY
 
        for i, t in enumerate(self.MOV_TIMES):
            __v = +_v if i % 2 == 0 else -_v
            __t = t + 0.5 if i % 2 == 0 else t - 0.5
            infront = HalfSpaceSet(normal=[+1, -__v], offset=[3.5,  t ], axes=["y", "t"])
            behind  = HalfSpaceSet(normal=[-1, +__v], offset=[3.0, __t], axes=["y", "t"])
            moving_obstacle = Union(
                moving_obstacle,
                reduce(Inter, [Compl(infront), Compl(behind), boundary]),
            )
 
        return moving_obstacle
 
    def _print_map_info(self):
        lo = self.LOWER_OBS
        uo = self.UPPER_OBS
        g  = self.GOAL
        print(f"Lower obstacle: x=({lo['x']}, {lo['x']+lo['w']}), y=({lo['y']}, {lo['y']+lo['h']})")
        print(f"Upper obstacle: x=({uo['x']}, {uo['x']+uo['w']}), y=({uo['y']}, {uo['y']+uo['h']})")
        print(f"Goal region:    x=({g['x']},  {g['x']+g['w']}),  y=({g['y']},  {g['y']+g['h']})")
        print("STATE_INIT:", self.STATE_INIT)
 
    # ──────────────────────────────────────────────
    # 2) Build TLT task
    # ──────────────────────────────────────────────
 
    @staticmethod
    def _follow_to(constraint, goal_or_constraint, *more):
        if more:  # 2nd arg is a constraint
            return CorridorScenario._follow_to(And(constraint, goal_or_constraint), *more)
        else:     # 2nd arg is the exit goal
            return Until(constraint, goal_or_constraint)
 
    def _build_task(self):
        self.vehicle_task = UNTIL(self.safe_region, self.target_set)
 
    # ──────────────────────────────────────────────
    # 3) Build TVHJImpl
    # ──────────────────────────────────────────────
 
    def _build_impl(self):
        self.impl = TVHJImpl(
            dynamics=dict(
                cls=Bicycle5D,
                min_steer=-self.MAX_STEER_RATE,
                max_steer= self.MAX_STEER_RATE,
                min_accel=-self.MAX_ACCEL,
                max_accel= self.MAX_ACCEL,
                wheelbase= self.WHEELBASE,
            ),
            axis_specs=[
                dict(name="t",     bounds=[0, self.T_MAX],                    points=101),
                dict(name="x",     bounds=[self.X0, self.X0 + self.XN],       points=25),
                dict(name="y",     bounds=[self.Y0, self.Y0 + self.YN],       points=25),
                dict(name="*yaw",  bounds=[-pi, pi],                          points=15),
                dict(name="delta", bounds=[-self.MAX_STEER, self.MAX_STEER],  points=11),
                dict(name="vel",   bounds=[0.0, self.MAX_VEL],                points=11),
            ],
        )
 
    # ──────────────────────────────────────────────
    # Public methods
    # ──────────────────────────────────────────────
 
    def plot_map(self):
        """Visualize the 4x4 corridor map."""
        lo = self.LOWER_OBS
        uo = self.UPPER_OBS
        g  = self.GOAL
 
        fig, ax = plt.subplots(figsize=(7, 7))
 
        # Workspace boundary
        ax.add_patch(Rectangle((self.X0, self.Y0), self.XN, self.YN,
                                fill=False, lw=2, edgecolor="black"))
 
        # Static obstacles
        ax.add_patch(Rectangle((lo["x"], lo["y"]), lo["w"], lo["h"],
                                facecolor="white", edgecolor="tab:blue",
                                hatch="++", linewidth=0.0, label="Obstacle"))
        ax.add_patch(Rectangle((uo["x"], uo["y"]), uo["w"], uo["h"],
                                facecolor="white", edgecolor="tab:blue",
                                hatch="////", linewidth=0.0))
 
        # Goal set
        ax.add_patch(Rectangle((g["x"], g["y"]), g["w"], g["h"],
                                facecolor="#34698d", edgecolor="#0c2d48",
                                linewidth=1.8, label="Goal set"))
 
        # Car pose
        heading_deg = np.degrees(self.STATE_INIT[2])
        car_rect    = Rectangle((-self.CAR_H / 2, -self.CAR_W / 2), self.CAR_H, self.CAR_W,
                                facecolor="0.8", edgecolor="0.5",
                                alpha=0.95, linewidth=1.0, label="Car")
        t_transform = (Affine2D()
                       .rotate_deg(heading_deg)
                       .translate(self.STATE_INIT[0], self.STATE_INIT[1])
                       + ax.transData)
        car_rect.set_transform(t_transform)
        ax.add_patch(car_rect)
        ax.plot(self.STATE_INIT[0], self.STATE_INIT[1], "ko", markersize=2.5)
 
        # Annotations
        ax.hlines(uo["y"], self.X0, uo["x"],
                  colors="k", linestyles=(0, (2, 2)), linewidth=0.8, alpha=0.6)
        ax.text(uo["x"] + uo["w"] / 2, uo["y"] - 0.06,
                "2.8 length", ha="center", va="top", fontsize=14)
        ax.text(uo["x"] + uo["w"] + 0.05, uo["y"] + uo["h"] / 2,
                "0.2 width", ha="left", va="center", fontsize=14)
        ax.text(g["x"] + g["w"] + 0.14, g["y"] + g["h"] / 2,
                "Goal set", ha="left", va="center", fontsize=15)
 
        ax.set_xlim(self.X0, self.X0 + self.XN)
        ax.set_ylim(self.Y0, self.Y0 + self.YN)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("4x4 corridor map")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_xticks([0, 2, 4])
        ax.set_yticks([2, 3, 4])
        ax.grid(False)
        plt.show()
 
    def compute_brt(self, save_path="out_bicycle5d.npy"):
        """Compute and save the backward reachable tube."""
        jax.config.update("jax_platform_name", "cpu")
        print(jax.default_backend())
        print(jax.devices())
 
        out = TLT.construct(self.vehicle_task).realize(self.impl)
        print("BRT shape:", out.shape)
 
        np.save(save_path, np.asarray(out))
        print(f"Saved {save_path}")
 
        self.brt = out
        return out
 
    def plot_brt(self):
        """Visualize the BRT (must call compute_brt first)."""
        assert hasattr(self, "brt"), "Call compute_brt() first."
 
        # XY contour (summed over time)
        self.impl.plot(
            self.brt.sum(axis=0, keepdims=True),
            method="contour",
            axes=("x", "y"),
        )
 
        # 3D x-y-t tube
        self.impl.plot(self.brt, axes=("x", "y", "t"))
 
        # Moving obstacle reachable set
        obs = self.moving_obstacle(self.impl)
        self.impl.plot(obs, axes=("x", "y", "t"), colorscale="reds").show()
        print(f"Moving obstacle value range: [{obs.min():.3f}, {obs.max():.3f}]")

if __name__ == "__main__":
    TLT.select(ContLTL)
    jax.config.update("jax_platform_name", "gpu")
    print(jax.default_backend())
    print(jax.devices())
    times = []
 
    scenario = CorridorScenario()
    for i in range(10):
        tm_start = tm.time()
        scenario.compute_brt()
        tm_end = tm.time()
        times.append(tm_end - tm_start)
        # print(f"Iteration {i+1}: BRT computed in {tm_end - tm_start:.2f} seconds")

    print(f"Time for each iteration: {times}")