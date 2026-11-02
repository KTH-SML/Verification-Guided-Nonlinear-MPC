import numpy as np


def crossing_events(dyn_obs):
    """
    Yield every crossing of the moving obstacle across the band between_y.

    Event i alternates direction (+speed on even index, -speed on odd). The
    obstacle enters the band at the crossing time t_enter (its front edge on
    the near side of the band), travels through it at constant speed and
    leaves it completely at t_exit (its back edge past the far side). Outside
    [t_enter, t_exit] it lies entirely outside the band.

    Both edges are straight lines in (y, t), in global time:
        y_lo(t) = y_lo0 + v * (t - t_enter)
        y_hi(t) = y_hi0 + v * (t - t_enter)

    Shared by Verifier.build_moving_object_local (half-spaces) and the
    animation (rectangles) so both use the exact same obstacle model.

    Yields
    ------
    (i, v, y_lo0, y_hi0, t_enter, t_exit) : (int, float, float, float, float, float)
    """
    y_bot, y_top = dyn_obs['between_y']
    length = dyn_obs['length']
    speed = dyn_obs['speed']
    t_cross = (y_top - y_bot + length) / speed
    for i, t_enter in enumerate(dyn_obs['crossings']):
        if i % 2 == 0:
            # upward: front (upper) edge enters at the bottom of the band
            v, y_hi0 = +speed, y_bot
        else:
            # downward: front (lower) edge enters at the top of the band
            v, y_hi0 = -speed, y_top + length
        yield i, v, y_hi0 - length, y_hi0, t_enter, t_enter + t_cross


def obstacle_rects(t, dyn_obs):
    """
    Rectangles (x_min, y_min, width, height) occupied by the moving obstacle
    at global time t, clipped to the band between_y.
    """
    y_bot, y_top = dyn_obs['between_y']
    rects = []
    for i, v, y_lo0, y_hi0, t_enter, t_exit in crossing_events(dyn_obs):
        if not t_enter <= t <= t_exit:
            continue
        y_lo = max(y_lo0 + v * (t - t_enter), y_bot)
        y_hi = min(y_hi0 + v * (t - t_enter), y_top)
        if y_hi <= y_lo:
            continue
        ox = dyn_obs['center_x'] - dyn_obs['width'] / 2
        rects.append((ox, y_lo, dyn_obs['width'], y_hi - y_lo))
    return rects
