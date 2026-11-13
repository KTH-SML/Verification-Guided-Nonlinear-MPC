import numpy as np


# ------------------------------------------------------------------
# Dynamic (moving) obstacles
# ------------------------------------------------------------------

def crossing_events(dyn_obs):
    """
    Yield every crossing of one moving obstacle across its band between_y.

    `dyn_obs['crossings']` lists the crossings. Each entry is either
        t_enter          : direction alternates (+speed on even index, -speed on odd)
        (t_enter, dir)   : dir = +1 (towards +y) or -1 (towards -y)
    The obstacle enters the band at the crossing time t_enter (its front edge on
    the near side of the band), travels through it at constant speed and
    leaves it completely at t_exit (its back edge past the far side). Outside
    [t_enter, t_exit] it lies entirely outside the band.

    Both edges are straight lines in (y, t), in global time:
        y_lo(t) = y_lo0 + v * (t - t_enter)
        y_hi(t) = y_hi0 + v * (t - t_enter)

    Shared by Verifier.build_moving_object_local (implicit sets) and the
    animation (rectangles) so both use the exact same obstacle model.

    Yields
    ------
    (i, v, y_lo0, y_hi0, t_enter, t_exit) : (int, float, float, float, float, float)
    """
    y_bot, y_top = dyn_obs['between_y']
    length = dyn_obs['length']
    speed = dyn_obs['speed']
    t_cross = (y_top - y_bot + length) / speed
    for i, entry in enumerate(dyn_obs['crossings']):
        if isinstance(entry, (tuple, list)):
            t_enter, direction = entry
        else:
            t_enter, direction = entry, (+1 if i % 2 == 0 else -1)
        if direction > 0:
            # upward: front (upper) edge enters at the bottom of the band
            v, y_hi0 = +speed, y_bot
        else:
            # downward: front (lower) edge enters at the top of the band
            v, y_hi0 = -speed, y_top + length
        yield i, v, y_hi0 - length, y_hi0, t_enter, t_enter + t_cross


def obstacle_rects(t, dyn_obs):
    """
    Rectangles (x_min, y_min, width, height) occupied by one moving obstacle
    at global time t, clipped to its band between_y.
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


# ------------------------------------------------------------------
# Vehicle footprint
# The state (x, y) of the bicycle model is the centre of the rear axle. The body is
# the rectangle [-rear_overhang, length - rear_overhang] x [-width/2, width/2] in the
# vehicle frame. The verifier treats the vehicle as a point, so every obstacle O is
# replaced by its configuration-space version O (+) (-F(yaw)), F(yaw) being the body
# rotated by yaw. For each yaw grid node the bounding box of F over the yaw interval
# covered by that node is used (conservative, exact at yaw = 0 for a straight road).
# ------------------------------------------------------------------

def footprint_corners(vehicle):
    """Body corners in the vehicle frame (rear-axle centre at the origin), shape (4, 2),
    or None for a point vehicle."""
    if not vehicle or vehicle.get('length') is None:
        return None
    x_r = -vehicle['rear_overhang']
    x_f = vehicle['length'] - vehicle['rear_overhang']
    w = vehicle['width'] / 2
    return np.array([[x_r, -w], [x_f, -w], [x_f, +w], [x_r, +w]])


def _contains_angle(c, a, b):
    """True where some c + 2*pi*k lies in [a, b] (elementwise)."""
    two_pi = 2 * np.pi
    return np.floor((b - c) / two_pi) >= np.ceil((a - c) / two_pi)


def _cos_range(r, phase, a, b):
    """(min, max) of r * cos(yaw + phase) over yaw in [a, b] (elementwise, exact)."""
    va, vb = r * np.cos(a + phase), r * np.cos(b + phase)
    lo, hi = np.minimum(va, vb), np.maximum(va, vb)
    hi = np.where(_contains_angle(-phase, a, b), r, hi)
    lo = np.where(_contains_angle(np.pi - phase, a, b), -r, lo)
    return lo, hi


def footprint_extents(vehicle, yaw_lo, yaw_hi):
    """
    Axis-aligned bounding box of the body (relative to the rear-axle centre, world
    frame) over all headings in [yaw_lo, yaw_hi] (arrays of the same shape).

    Returns fx_min, fx_max, fy_min, fy_max (arrays). A point vehicle gives zeros.
    """
    yaw_lo, yaw_hi = np.asarray(yaw_lo, float), np.asarray(yaw_hi, float)
    corners = footprint_corners(vehicle)
    if corners is None:
        z = np.zeros_like(yaw_lo)
        return z, z, z, z
    fx_min = fy_min = np.full_like(yaw_lo, np.inf)
    fx_max = fy_max = np.full_like(yaw_lo, -np.inf)
    for xb, yb in corners:
        r, phase = np.hypot(xb, yb), np.arctan2(yb, xb)
        # x = r cos(yaw + phase), y = r sin(yaw + phase) = r cos(yaw + phase - pi/2)
        x_lo, x_hi = _cos_range(r, phase, yaw_lo, yaw_hi)
        y_lo, y_hi = _cos_range(r, phase - np.pi / 2, yaw_lo, yaw_hi)
        fx_min, fx_max = np.minimum(fx_min, x_lo), np.maximum(fx_max, x_hi)
        fy_min, fy_max = np.minimum(fy_min, y_lo), np.maximum(fy_max, y_hi)
    return fx_min, fx_max, fy_min, fy_max


def footprint_polygon(x, y, yaw, vehicle):
    """Body corners in the world frame for the pose (x, y, yaw), shape (4, 2), or None."""
    corners = footprint_corners(vehicle)
    if corners is None:
        return None
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([x, y])


def inflate_rect(rect, vehicle, yaw=0.0):
    """Configuration-space version of rect = (x_min, y_min, width, height) for the
    rear-axle centre at heading `yaw` (only for drawing)."""
    fx_min, fx_max, fy_min, fy_max = (float(v) for v in footprint_extents(vehicle, yaw, yaw))
    ox, oy, ow, oh = rect
    return (ox - fx_max, oy - fy_max, ow + fx_max - fx_min, oh + fy_max - fy_min)
