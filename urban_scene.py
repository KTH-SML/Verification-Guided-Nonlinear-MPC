"""
Top-view drawing of the urban road scenario (SCENARIO = 'urban_road' in config.py),
to scale, from the same values the simulation uses. Extracted from TTR.ipynb
("Urban road scenario") so that scripts (plot_snapshots.py) can reuse it.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Ellipse, Polygon
from matplotlib.lines import Line2D

import config
from obstacles import crossing_events

# Configuration of the urban scenario (same values the simulation uses)
_reg = {}
config.run(_reg)
ucfg = _reg['config']
assert ucfg['scenario'] == 'urban_road', "set SCENARIO = 'urban_road' in config.py"

X0, XW, YW = ucfg['X0'], ucfg['X0'] + ucfg['XW'], ucfg['Y0'] + ucfg['YW']
Y_RIGHT, Y_CENTRE, LANE = config.Y_RIGHT, config.Y_CENTRE, config.LANE_WIDTH
VEH = ucfg['vehicle']
STATIC = ucfg['obstacles'][len(config.ROAD_CURBS):]          # blockages (curbs excluded)
DYN = {d['name']: d for d in ucfg['dynamic_obstacles']}
SIDEWALK = 3.0                                               # m, drawn only (not part of the model)

# Where the road users cross: crosswalks for pedestrians, a cycle crossing for
# cyclists, intersections with a side street for crossing cars, a tram line for trams.
CROSSWALKS = [d['center_x'] for d in DYN.values() if d['name'].startswith('pedestrian')]
BIKE_X = [d['center_x'] for d in DYN.values() if d['name'].startswith('cyclist')]
SIDE_STREETS = [d['center_x'] for d in DYN.values() if d['name'].startswith('crossing car')]
TRAM_X = [d['center_x'] for d in DYN.values() if d['name'].startswith('tram')]
CROSSWALK_W, BIKE_W, SIDE_W, TRAM_W = 4.0, 2.0, 8.0, 5.0      # m, extent along x
GAUGE = 1.435                                                 # m, standard track gauge

# colours
C_ASPHALT, C_SHOULDER = '#8c8f94', '#95989d'
C_SIDEWALK, C_CURB = '#cfc9bf', '#8f8a82'
C_MARK = '#f4f4f4'
C_BIKE = '#c0392b'
C_EGO = '#1f77b4'      # ego car: blue
C_OTHER = '#f28e2b'    # other (crossing) cars: orange
C_GLASS = '#bfe3f5'
C_HAZARD = '#ffb000'
C_TREE, C_TREE_HI = '#3f7f2f', '#6aa84f'
C_TRAM, C_TRACKBED, C_RAIL = '#c8102e', '#8a8680', '#d9d9d9'


def _body(ax, x, y, yaw, x_b, y_b, length, width, fc, rounding=0.35, z=6, **kw):
    """Rounded rectangle [x_b, x_b + length] x [y_b, y_b + width] in a body frame placed at (x, y, yaw).
    Returns the transform of the body frame."""
    tr = mtransforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
    ax.add_patch(FancyBboxPatch((x_b, y_b), length, width, boxstyle=f'round,pad=0,rounding_size={rounding}',
                                facecolor=fc, edgecolor='k', lw=1.0, zorder=z, transform=tr, **kw))
    return tr


def draw_car(ax, x, y, yaw, length, width, rear, fc, z=6, hazards=False):
    """Top view of a car; (x, y) = centre of the rear axle, body [-rear, length - rear]."""
    tr = _body(ax, x, y, yaw, -rear, -width / 2, length, width, fc, z=z)
    xf = length - rear
    # windscreen, roof, rear window
    ax.add_patch(Polygon([[xf - 1.55, -0.42 * width], [xf - 1.05, -0.36 * width],
                          [xf - 1.05, 0.36 * width], [xf - 1.55, 0.42 * width]],
                         facecolor=C_GLASS, edgecolor='k', lw=0.6, zorder=z + 0.1, transform=tr))
    ax.add_patch(Rectangle((-rear + 0.35, -0.38 * width), 0.45, 0.76 * width, facecolor=C_GLASS,
                           edgecolor='k', lw=0.6, zorder=z + 0.1, transform=tr))
    # headlights
    for s in (-1, 1):
        ax.add_patch(Ellipse((xf - 0.12, s * 0.33 * width), 0.18, 0.32, facecolor='#fff7c2',
                             edgecolor='k', lw=0.4, zorder=z + 0.1, transform=tr))
    if hazards:
        for cx_ in (-rear + 0.1, xf - 0.1):
            for s in (-1, 1):
                ax.add_patch(Circle((cx_, s * (width / 2 - 0.1)), 0.22, facecolor=C_HAZARD, alpha=0.9,
                                    edgecolor='none', zorder=z + 0.2, transform=tr))


def draw_van(ax, ox, oy, ow, oh):
    """Delivery van filling the rectangle, facing +x, hazard lights on."""
    y_c = oy + oh / 2
    tr = _body(ax, ox, y_c, 0.0, 0.0, -oh / 2, ow, oh, '#f2f2f2', rounding=0.3)
    ax.add_patch(Rectangle((ow - 1.6, -oh / 2), 1.6, oh, facecolor='#dcdcdc', edgecolor='k', lw=0.8,
                           zorder=6.05, transform=tr))                               # cab
    ax.add_patch(Polygon([[ow - 0.9, -0.42 * oh], [ow - 0.5, -0.37 * oh], [ow - 0.5, 0.37 * oh],
                          [ow - 0.9, 0.42 * oh]], facecolor=C_GLASS, edgecolor='k', lw=0.6,
                         zorder=6.1, transform=tr))                                  # windscreen
    for yy in (-0.25 * oh, 0.25 * oh):                                               # roof rails
        ax.plot([0.3, ow - 1.9], [yy, yy], color='0.6', lw=1.0, zorder=6.1, transform=tr)
    for cx_ in (0.1, ow - 0.1):
        for s in (-1, 1):
            ax.add_patch(Circle((cx_, s * (oh / 2 - 0.1)), 0.22, facecolor=C_HAZARD, alpha=0.9,
                                edgecolor='none', zorder=6.2, transform=tr))


def draw_roadworks(ax, ox, oy, ow, oh):
    """Work zone: excavated area, red/white barriers on the traffic sides, cones, sign."""
    ax.add_patch(Rectangle((ox, oy), ow, oh, facecolor='#c9a46b', edgecolor='none', zorder=5))
    ax.add_patch(Rectangle((ox + 1.2, oy + 0.9), ow - 2.4, oh - 1.6, facecolor='#8d6e48',
                           edgecolor='none', zorder=5.05))                            # trench
    # barriers along the open sides (bottom = towards the traffic, left/right ends)
    seg = 0.5
    for k, xx in enumerate(np.arange(ox, ox + ow, seg)):
        ax.add_patch(Rectangle((xx, oy), seg, 0.25, facecolor='#d62728' if k % 2 == 0 else 'white',
                               edgecolor='none', zorder=5.2))
    for x_side in (ox, ox + ow - 0.25):
        for k, yy in enumerate(np.arange(oy, oy + oh, seg)):
            ax.add_patch(Rectangle((x_side, yy), 0.25, min(seg, oy + oh - yy),
                                   facecolor='#d62728' if k % 2 == 0 else 'white', edgecolor='none', zorder=5.2))
    for xx in np.arange(ox + 0.6, ox + ow, 1.6):                                      # cones
        ax.add_patch(Circle((xx, oy + 0.55), 0.22, facecolor='#ff7f0e', edgecolor='k', lw=0.5, zorder=5.3))
        ax.add_patch(Circle((xx, oy + 0.55), 0.08, facecolor='white', edgecolor='none', zorder=5.4))
    # triangular "men at work" sign
    sx, sy = ox + ow / 2, oy + oh / 2 + 0.3
    ax.add_patch(Polygon([[sx - 0.8, sy - 0.6], [sx + 0.8, sy - 0.6], [sx, sy + 0.8]],
                         facecolor='#ffd23f', edgecolor='#d62728', lw=2.0, zorder=5.5))
    ax.text(sx, sy - 0.1, '!', ha='center', va='center', fontsize=9, fontweight='bold', zorder=5.6)


def draw_pedestrian(ax, x, y, v):
    """Pedestrian seen from above (shoulders + head), walking along +-y."""
    ax.add_patch(Ellipse((x, y), 0.6, 0.35, facecolor='#2a6fdb', edgecolor='k', lw=0.6, zorder=8))
    ax.add_patch(Circle((x, y + 0.04 * np.sign(v)), 0.13, facecolor='#5a3e2b', edgecolor='k', lw=0.5, zorder=8.1))


def draw_cyclist(ax, x, y, v, length):
    """Bicycle with rider, travelling along +-y."""
    s = np.sign(v)
    ax.plot([x, x], [y - length / 2, y + length / 2], color='k', lw=2.5, solid_capstyle='round', zorder=8)
    ax.plot([x - 0.3, x + 0.3], [y + s * 0.45 * length / 2] * 2, color='k', lw=2, zorder=8)   # handlebar
    ax.add_patch(Ellipse((x, y), 0.55, 0.35, facecolor='#2ca02c', edgecolor='k', lw=0.6, zorder=8.1))
    ax.add_patch(Circle((x, y + s * 0.15), 0.13, facecolor='#ffd23f', edgecolor='k', lw=0.5, zorder=8.2))


def draw_tram(ax, x, y_c, v, length, width, n_mod=5, z=8):
    """Articulated tram seen from above, centred at (x, y_c), travelling along +-y."""
    yaw = np.pi / 2 if v > 0 else -np.pi / 2          # body frame: +x = direction of travel
    gap = 0.5
    seg = (length - (n_mod - 1) * gap) / n_mod
    for k in range(n_mod):
        xb = -length / 2 + k * (seg + gap)
        tr = _body(ax, x, y_c, yaw, xb, -width / 2, seg, width, C_TRAM,
                   rounding=0.8 if k in (0, n_mod - 1) else 0.15, z=z)
        # roof: light band with the equipment boxes
        ax.add_patch(Rectangle((xb + 0.6, -0.3 * width), seg - 1.2, 0.6 * width, facecolor='#e8e8e8',
                               edgecolor='k', lw=0.5, zorder=z + 0.1, transform=tr))
        if k == n_mod // 2:                           # pantograph on the middle module
            xp = xb + seg / 2
            ax.plot([xp - 1.0, xp, xp + 1.0, xp], [0, 0.35 * width, 0, -0.35 * width],
                    color='k', lw=1.2, zorder=z + 0.2, transform=tr)
        if k < n_mod - 1:                             # bellows between the modules
            ax.add_patch(Rectangle((xb + seg, -0.4 * width), gap, 0.8 * width, facecolor='#333333',
                                   edgecolor='none', zorder=z, transform=tr))
    # windscreens at both ends (bidirectional tram)
    for x_end, s in ((length / 2, -1), (-length / 2, 1)):
        ax.add_patch(Polygon([[x_end + s * 0.15, -0.38 * width], [x_end + s * 0.6, -0.42 * width],
                              [x_end + s * 0.6, 0.42 * width], [x_end + s * 0.15, 0.38 * width]],
                             facecolor=C_GLASS, edgecolor='k', lw=0.6, zorder=z + 0.1, transform=tr))


def draw_tree(ax, x, y, r=1.1):
    ax.add_patch(Circle((x + 0.15, y - 0.15), r, facecolor='k', alpha=0.15, lw=0, zorder=2.2))
    ax.add_patch(Circle((x, y), r, facecolor=C_TREE, edgecolor='#2c5a20', lw=0.6, zorder=2.3))
    ax.add_patch(Circle((x - 0.3 * r, y + 0.3 * r), 0.45 * r, facecolor=C_TREE_HI, lw=0, zorder=2.4))


def _in_any(x, centres, half):
    return any(abs(x - c) < half for c in centres)


def draw_road(ax):
    """Asphalt, shoulders, sidewalks with curbs and trees, lane markings, crossings."""
    ax.set_facecolor('#e9e6df')
    # sidewalks and curbs
    for y_sw, y_curb in ((-SIDEWALK, 0.0), (YW, YW)):
        ax.add_patch(Rectangle((X0, y_sw), XW - X0, SIDEWALK, facecolor=C_SIDEWALK, edgecolor='none', zorder=1))
        ax.plot([X0, XW], [y_curb, y_curb], color=C_CURB, lw=3, zorder=3)
    # side streets (intersection): asphalt across the sidewalks, stop lines
    for cx in SIDE_STREETS:
        for y_sw in (-SIDEWALK, YW):
            ax.add_patch(Rectangle((cx - SIDE_W / 2, y_sw), SIDE_W, SIDEWALK, facecolor=C_ASPHALT,
                                   edgecolor='none', zorder=3.1))
        ax.add_patch(Rectangle((cx - 0.15, -SIDEWALK), 0.3, SIDEWALK, facecolor=C_MARK, zorder=3.2))
        ax.add_patch(Rectangle((cx - 0.15, YW), 0.3, SIDEWALK, facecolor=C_MARK, zorder=3.2))
        for y_stop in (-0.6, YW + 0.3):                                               # stop lines
            half = (cx - SIDE_W / 2 + 0.3, cx - 0.4) if y_stop < 0 else (cx + 0.4, cx + SIDE_W / 2 - 0.3)
            ax.add_patch(Rectangle((half[0], y_stop), half[1] - half[0], 0.3, facecolor=C_MARK, zorder=3.2))
    # tram line: track bed across the sidewalks
    for cx in TRAM_X:
        for y_sw in (-SIDEWALK, YW):
            ax.add_patch(Rectangle((cx - TRAM_W / 2, y_sw), TRAM_W, SIDEWALK, facecolor=C_TRACKBED,
                                   edgecolor='none', zorder=3.1))
    # carriageway and shoulders
    ax.add_patch(Rectangle((X0, 0), XW - X0, YW, facecolor=C_ASPHALT, edgecolor='none', zorder=1.5))
    for y_sh in (0.0, YW - (Y_RIGHT - 0.0)):
        ax.add_patch(Rectangle((X0, y_sh), XW - X0, Y_RIGHT, facecolor=C_SHOULDER, edgecolor='none', zorder=1.6))
    # lane markings: solid edge lines, dashed centre line (3 m dash, 4.5 m gap),
    # interrupted at crossings and at the intersection
    gaps = [(c, CROSSWALK_W / 2 + 0.5) for c in CROSSWALKS] + [(c, BIKE_W / 2 + 0.5) for c in BIKE_X] + \
           [(c, SIDE_W / 2) for c in SIDE_STREETS] + [(c, TRAM_W / 2 + 0.5) for c in TRAM_X]
    def free(x):
        return not any(abs(x - c) < h for c, h in gaps)
    xs = np.arange(X0, XW, 0.25)
    for y_line in (Y_RIGHT, Y_CENTRE + LANE):
        for x in xs:
            if free(x) and free(x + 0.25) and not _in_any(x, SIDE_STREETS, SIDE_W / 2):
                ax.add_patch(Rectangle((x, y_line - 0.075), 0.25, 0.15, facecolor=C_MARK, edgecolor='none', zorder=2))
    for x in np.arange(X0 + 1.0, XW, 7.5):
        if free(x) and free(x + 3.0):
            ax.add_patch(Rectangle((x, Y_CENTRE - 0.075), 3.0, 0.15, facecolor=C_MARK, edgecolor='none', zorder=2))
    # zebra crossings: stripes parallel to the traffic, 0.5 m wide every 1 m
    for cx in CROSSWALKS:
        for y in np.arange(0.25, YW - 0.4, 1.0):
            ax.add_patch(Rectangle((cx - CROSSWALK_W / 2, y), CROSSWALK_W, 0.5, facecolor=C_MARK,
                                   edgecolor='none', zorder=2.1))
        for y_sw in (-SIDEWALK, YW):                                                   # kerb ramps
            ax.add_patch(Rectangle((cx - CROSSWALK_W / 2, y_sw), CROSSWALK_W, SIDEWALK,
                                   facecolor='#bdb6aa', edgecolor='none', zorder=1.1))
    # cycle crossing: red surface and square "elephant feet" markings, cycle path on the sidewalks
    for cx in BIKE_X:
        ax.add_patch(Rectangle((cx - BIKE_W / 2, 0), BIKE_W, YW, facecolor=C_BIKE, alpha=0.55,
                               edgecolor='none', zorder=2.05))
        for y in np.arange(0.1, YW - 0.3, 0.9):
            for xe in (cx - BIKE_W / 2 - 0.25, cx + BIKE_W / 2 - 0.25):
                ax.add_patch(Rectangle((xe, y), 0.5, 0.5, facecolor=C_MARK, edgecolor='none', zorder=2.1))
        for y_sw in (-SIDEWALK, YW):
            ax.add_patch(Rectangle((cx - BIKE_W / 2, y_sw), BIKE_W, SIDEWALK, facecolor=C_BIKE,
                                   alpha=0.55, edgecolor='none', zorder=1.2))
    # tram line: rails across the whole road and the sidewalks, warning lines before it
    for cx in TRAM_X:
        for xr in (cx - GAUGE / 2, cx + GAUGE / 2):
            ax.plot([xr, xr], [-SIDEWALK, YW + SIDEWALK], color=C_RAIL, lw=1.8, zorder=3.3)
        for xw in (cx - TRAM_W / 2 - 0.3, cx + TRAM_W / 2):
            for y in np.arange(0.0, YW, 0.6):
                ax.add_patch(Rectangle((xw, y), 0.3, 0.3, facecolor='#ffd23f', edgecolor='none', zorder=2.1))
    # trees on the sidewalks, away from crossings, side streets and the tram line
    def tree_ok(x):
        return not (_in_any(x, CROSSWALKS + BIKE_X, 4.5) or _in_any(x, SIDE_STREETS, SIDE_W / 2 + 2.0)
                    or _in_any(x, TRAM_X, TRAM_W / 2 + 2.0))
    for x in np.arange(X0 + 4.0, XW - 2.0, 9.0):
        if tree_ok(x):
            draw_tree(ax, x, -SIDEWALK + 1.3)
        if tree_ok(x + 4.5) and x + 4.5 < XW - 2.0:
            draw_tree(ax, x + 4.5, YW + SIDEWALK - 1.3)


def draw_goal(ax, gx, gy, gw, gh):
    ax.add_patch(Rectangle((gx, gy), gw, gh, facecolor='#2ca02c', alpha=0.35, edgecolor='none', zorder=4))
    ax.add_patch(Rectangle((gx, gy), gw, gh, fill=False, edgecolor='#2ca02c', lw=2, ls='--', zorder=4.1))
    ax.text(gx + gw / 2, gy + gh / 2, 'GOAL', ha='center', va='center', color='white', fontsize=8,
            fontweight='bold', zorder=4.2)


def draw_scenario(ax, road_users=True):
    draw_road(ax)
    van, works, broken = STATIC
    draw_van(ax, *van)
    draw_roadworks(ax, *works)
    bx, by, bw, bh = broken
    draw_car(ax, bx + 0.9, by + bh / 2, 0.0, bw, bh, 0.9, '#9aa3ad', z=6, hazards=True)
    draw_goal(ax, *ucfg['goal'])
    xs, ys, yaw = ucfg['x_init'][:3]
    draw_car(ax, xs, ys, yaw, VEH['length'], VEH['width'], VEH['rear_overhang'], C_EGO, z=7)
    ax.annotate('', xy=(xs + 6.0, ys), xytext=(xs + VEH['length'] - VEH['rear_overhang'] + 0.3, ys),
                arrowprops=dict(arrowstyle='-|>', color=C_EGO, lw=2), zorder=7)
    if road_users:
        for d in ucfg['dynamic_obstacles']:
            for i, v, y_lo0, y_hi0, t_in, t_out in crossing_events(d):
                # drawn halfway through its crossing
                t_mid = 0.5 * (t_in + t_out)
                y_mid = 0.5 * (y_lo0 + y_hi0) + v * (t_mid - t_in)
                x = d['center_x']
                if d['name'].startswith('pedestrian'):
                    draw_pedestrian(ax, x, y_mid, v)
                elif d['name'].startswith('cyclist'):
                    draw_cyclist(ax, x, y_mid, v, d['length'])
                elif d['name'].startswith('tram'):
                    draw_tram(ax, x, y_mid, v, d['length'], d['width'])
                else:   # crossing car: body along +-y, (x, y) of draw_car is the rear axle
                    yaw_c = np.pi / 2 if v > 0 else -np.pi / 2
                    draw_car(ax, x, y_mid - np.sign(v) * (d['length'] / 2 - 0.9), yaw_c,
                             d['length'], d['width'], 0.9, C_OTHER, z=8)
                y0, y1 = (1.0, YW - 1.0) if v > 0 else (YW - 1.0, 1.0)
                # arrow next to the road user (outside the cycle crossing / the tram body)
                if d['name'].startswith('cyclist'):
                    xa = x + BIKE_W / 2 + 0.7
                elif d['name'].startswith('tram'):
                    xa = x + d['width'] / 2 + 0.8
                else:
                    xa = x + 1.4
                ax.annotate('', xy=(xa, y1), xytext=(xa, y0),
                            arrowprops=dict(arrowstyle='-|>', color='#d62728', lw=1.8), zorder=9)
                ax.text(xa + 0.5, YW / 2, f"{d['name']}",
                        ha='left', va='center', fontsize=10, color='#d62728', fontweight='bold', zorder=9,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85))
    ax.set_xlim(X0, XW)
    ax.set_ylim(-SIDEWALK, YW + SIDEWALK)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(X0, XW + 1e-9, 10.0))
    ax.set_yticks([0, Y_RIGHT, Y_CENTRE, Y_CENTRE + LANE, YW])
    ax.set_xlabel('x [m]', fontsize=18); ax.set_ylabel('y [m]', fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
