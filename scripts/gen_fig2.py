"""Generate Figure 2: Dual-Route Cross-Embodiment Retargeting & Data Contract.

Vertically aligned 4-stage abstraction across Route A (articulated dexterous hands)
and Route B (two-finger parallel grippers), reusing real 3D URDF / tactile / UV assets.

User Design Notes:
- Hand icon: use assets/fig2/hand.svg rendered in warm skin tone (#E8D4C5/#A6826B),
  mirrored thumb-left to align with URDF/skeleton orientation and 1-to-1 fingertip mapping.
- Gripper icon (Route B Col 2 & Col 4): use downward-rendered real 3D WSG-50 + GelSight
  Mini from Depth Align (same downward pose, both inner pads colored for mapping/remap).
- Right UV preview: match Figure 1's vertical Mask -> UV flow (remove the 3-Part Data
  Contract card), with MANO UV Mask on top, downward mapping arrow, and split
  UV preview below (center white dashed divider, muted left border, blue right
  highlight border).
- UV Color Remap fingertips: avoid flat/uniform 5-color "nail polish" fills AND
  avoid dark/creepy eyeball-like pad bases; render organic, irregular multi-blob
  contact smudges directly on the skin-toned fingertip (alpha fading to 0 at edges,
  using warm Oranges/YlOrRd on raw input -> soft Greys on uint8 output).

Usage:
    uv run --with matplotlib --with numpy --with pillow python scripts/gen_fig2.py
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path
from PIL import Image

SKILL_SCRIPTS = os.path.expanduser("~/.gemini/config/skills/cs-paper-figure/scripts")
if not os.path.isdir(SKILL_SCRIPTS):
    SKILL_SCRIPTS = os.path.expanduser("~/.agents/skills/cs-paper-figure/scripts")
sys.path.insert(0, SKILL_SCRIPTS)
from paperkit import Canvas  # noqa: E402

# ---------------------------------------------------------------------------
# Visual system constants (matching Figure 1, ACL full-width 156 mm)
# ---------------------------------------------------------------------------
W, H = 156.0, 81.0
T = lambda top_mm: H - top_mm  # noqa: E731

C_SLATE = "#475569"
C_SLATE_FILL = "#EDF1F5"
C_SLATE_SOFT = "#CBD5E1"
C_INK = "#0F172A"
C_MUTED = "#64748B"

C_HERO = "#1D4ED8"
C_HERO_FILL = "#EFF6FF"
C_HERO_SOFT = "#BFDBFE"

C_EVAL = "#0F766E"
C_EVAL_FILL = "#F0FDFA"
C_EVAL_SOFT = "#99F6E4"

C_ACCENT = "#D97706"
C_ACCENT_SOFT = "#FEF3C7"

LW_BASE = 0.6
LW_HERO = 0.85
SZ_HEAD = 7.4
SZ_BODY = 6.4
SZ_SUB = 5.8


def place_rgba_image(cv, path, cx, cy, w, z=8, alpha=1.0):
    """Place a transparent RGBA PNG directly onto the canvas."""
    im = Image.open(path).convert("RGBA")
    h = w * (im.height / im.width)
    arr = np.asarray(im).copy()
    if alpha < 1.0:
        arr[..., 3] = (arr[..., 3].astype(np.float32) * alpha).astype(np.uint8)
    cv.ax.imshow(
        arr,
        extent=[cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0],
        origin="upper",
        interpolation="lanczos",
        zorder=z,
    )
    return h


def _draw_mini_skeleton(cv, kx, cy, scale, color, dashed=False, rot=0.0):
    """Draw a compact 5-finger 21-keypoint hand skeleton."""
    wrist = (kx, cy - 3.4 * scale)
    fingers = [
        [(-2.3, -1.4), (-3.3, -0.3), (-4.0, 0.7)],
        [(-1.4, -0.5), (-1.7, 1.2), (-1.9, 2.7), (-2.0, 3.8)],
        [(-0.3, -0.4), (-0.3, 1.5), (-0.3, 3.1), (-0.3, 4.3)],
        [(0.8, -0.5), (1.0, 1.2), (1.1, 2.7), (1.2, 3.8)],
        [(1.9, -1.0), (2.3, 0.5), (2.6, 1.8), (2.8, 2.8)],
    ]
    ca, sa = np.cos(rot), np.sin(rot)
    cv.ax.add_patch(Circle(wrist, 0.38, fc=color, ec="white", lw=0.25, zorder=9))
    for f_pts in fingers:
        pts = [wrist]
        for dx, dy in f_pts:
            rx = (dx * ca - dy * sa) * scale
            ry = (dx * sa + dy * ca) * scale
            pts.append((kx + rx, cy + ry))
        for i in range(len(pts) - 1):
            cv.line([pts[i], pts[i + 1]], color=color, lw=0.6, dashed=dashed)
        for j, pt in enumerate(pts[1:]):
            is_tip = (j == len(pts) - 2)
            cv.ax.add_patch(
                Circle(pt, 0.32 if is_tip else 0.24, fc=color, ec="white", lw=0.2, zorder=9)
            )


# ---------------------------------------------------------------------------
# Route A (Articulated Dexterous Hands) — Aligned 4 Macro-Stages
# ---------------------------------------------------------------------------

def draw_route_a_col1_prep(cv, cx, cy):
    """Col 1: Real URDF hand -> 21-kpt FK skeleton + real tactile luma (skip missing)."""
    place_rgba_image(cv, "assets/fig1/_urdf_sharpa.png", cx=cx - 7.5, cy=cy, w=5.8, z=7, alpha=0.96)
    cv.arrow([(cx - 4.3, cy), (cx - 2.9, cy)], color=C_SLATE, lw=0.55, head=0.65)
    _draw_mini_skeleton(cv, cx + 0.7, cy - 0.4, scale=0.88, color=C_SLATE)
    h_dot = place_rgba_image(cv, "assets/fig1/_tactile_dot.png", cx=cx + 8.0, cy=cy + 1.1, w=4.6, z=7)
    cv.ax.add_patch(
        Rectangle((cx + 5.7, cy + 1.1 - h_dot / 2.0), 4.6, h_dot, fc="none", ec=C_SLATE, lw=0.45, zorder=8)
    )
    for i in range(5):
        dx = cx + 8.0 + (i - 2) * 1.05
        if i < 4:
            cv.ax.add_patch(Circle((dx, cy - 2.5), 0.36, fc=C_ACCENT, ec="none", zorder=7))
        else:
            cv.ax.add_patch(Circle((dx, cy - 2.5), 0.36, fc="white", ec=C_MUTED, lw=0.4, ls="--", zorder=7))


def draw_route_a_col2_align(cv, cx, cy):
    """Col 2: Inverse bone-vector fit + 11-pt Umeyama: Misaligned -> Overlapped."""
    # Left: Misaligned robot (slate) + tilted MANO (dashed blue)
    _draw_mini_skeleton(cv, cx - 6.15, cy - 0.45, scale=0.94, color=C_SLATE)
    _draw_mini_skeleton(cv, cx - 4.95, cy - 0.25, scale=0.84, color=C_HERO, dashed=True, rot=-0.26)
    # Middle arrow: optimization + similarity transform
    cv.arrow([(cx - 1.1, cy), (cx + 1.7, cy)], color=C_INK, lw=0.6, head=0.75)
    cv.text(cx + 0.4, cy + 4.4, "z, s, R, t", size=SZ_SUB, color=C_HERO)
    # Right: Perfectly overlapped aligned skeletons
    _draw_mini_skeleton(cv, cx + 7.25, cy - 0.45, scale=0.94, color=C_SLATE)
    _draw_mini_skeleton(cv, cx + 7.25, cy - 0.45, scale=0.93, color=C_HERO, dashed=True, rot=0.0)


_SVG_HAND_CACHE = {}


def _load_svg_hand_geometry(svg_path="assets/fig2/hand.svg", flip_x=True):
    """Parse assets/fig2/hand.svg into a normalized matplotlib Path centered at (0, 0) with height=1.0."""
    key = (svg_path, flip_x)
    if key in _SVG_HAND_CACHE:
        return _SVG_HAND_CACHE[key]

    root = ET.parse(svg_path).getroot()
    path_el = root.find(".//{http://www.w3.org/2000/svg}path")
    if path_el is None:
        path_el = root.find(".//path")
    d = path_el.attrib["d"]
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", d)

    i = 0
    cur = np.array([0.0, 0.0])
    start = cur.copy()
    verts, codes = [], []
    cmd = None
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
        if cmd in ("M", "m"):
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            cur = np.array([x, y]) if cmd == "M" else cur + np.array([x, y])
            start = cur.copy()
            verts.append(cur.copy())
            codes.append(Path.MOVETO)
            cmd = "L" if cmd == "M" else "l"
        elif cmd in ("L", "l"):
            x, y = float(tokens[i]), float(tokens[i + 1])
            i += 2
            cur = np.array([x, y]) if cmd == "L" else cur + np.array([x, y])
            verts.append(cur.copy())
            codes.append(Path.LINETO)
        elif cmd in ("C", "c"):
            vals = [float(tokens[i + k]) for k in range(6)]
            i += 6
            base = np.array([0.0, 0.0]) if cmd == "C" else cur
            p1 = base + np.array([vals[0], vals[1]])
            p2 = base + np.array([vals[2], vals[3]])
            p3 = base + np.array([vals[4], vals[5]])
            verts.extend([p1, p2, p3])
            codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
            cur = p3
        elif cmd in ("Z", "z"):
            verts.append(start.copy())
            codes.append(Path.CLOSEPOLY)
            cur = start.copy()
        else:
            raise ValueError(f"Unsupported SVG path command: {cmd}")

    v = np.asarray(verts, dtype=float)
    # Convert SVG y-down (0..1024) to matplotlib y-up; flip_x puts thumb on the left (right-hand palmar view).
    vx = (1024.0 - v[:, 0]) if flip_x else v[:, 0]
    vy = 1024.0 - v[:, 1]
    v_up = np.column_stack([vx, vy])

    # Normalize using the evaluated curve polygon bounds so height == 1.0 and bbox center == (0, 0).
    poly = Path(v_up, codes).to_polygons()[0]
    cx_raw = 0.5 * (poly[:, 0].min() + poly[:, 0].max())
    cy_raw = 0.5 * (poly[:, 1].min() + poly[:, 1].max())
    h_raw = poly[:, 1].max() - poly[:, 1].min()

    v_norm = (v_up - np.array([cx_raw, cy_raw])) / h_raw
    _SVG_HAND_CACHE[key] = (v_norm, codes, cx_raw, cy_raw, h_raw)
    return _SVG_HAND_CACHE[key]


def _draw_svg_hand(
    cv,
    cx,
    cy,
    height=7.6,
    hand_fc="#E8D4C5",
    hand_ec="#A6826B",
    finger_colors=None,
    pad_ec="none",
    svg_path="assets/fig2/hand.svg",
):
    """Draw the hand from assets/fig2/hand.svg in warm skin tone with aligned fingertip pads.

    Returns a dict mapping each finger name ('thumb', 'index', 'middle', 'ring', 'pinky')
    to its world (x, y) fingertip pad center so connectors can attach accurately.
    """
    v_norm, codes, cx_raw, cy_raw, h_raw = _load_svg_hand_geometry(svg_path, flip_x=True)
    v_world = v_norm * height + np.array([cx, cy])
    hand_patch = PathPatch(
        Path(v_world, codes),
        facecolor=hand_fc,
        edgecolor=hand_ec,
        linewidth=0.48,
        joinstyle="round",
        zorder=6,
    )
    cv.ax.add_patch(hand_patch)

    # Exact fingertip pad centers (in flipped y-up 0..1024 coordinates), width, height, and tilt angle (deg).
    raw_tips = {
        "thumb":  (166.0, 428.0, 82.0, 128.0,  44.0),
        "index":  (387.0, 796.0, 82.0, 130.0,   7.0),
        "middle": (569.0, 882.0, 82.0, 134.0,   0.0),
        "ring":   (736.0, 796.0, 82.0, 130.0, -10.0),
        "pinky":  (858.0, 622.0, 78.0, 124.0, -20.0),
    }

    import matplotlib.pyplot as plt

    tip_centers = {}
    for name, (rx, ry, rw, rh, ang) in raw_tips.items():
        wx = cx + ((rx - cx_raw) / h_raw) * height
        wy = cy + ((ry - cy_raw) / h_raw) * height
        tip_centers[name] = (wx, wy)
        if finger_colors and name in finger_colors:
            spec = finger_colors[name]
            ew = (rw / h_raw) * height
            eh = (rh / h_raw) * height
            if isinstance(spec, dict):
                # Optional faint pad outline only (no dark solid base that looks like eyeballs)
                if pad_ec != "none":
                    pad = Ellipse(
                        (wx, wy),
                        width=ew,
                        height=eh,
                        angle=ang,
                        facecolor="none",
                        edgecolor=pad_ec,
                        linewidth=0.25,
                        alpha=0.45,
                        zorder=8,
                    )
                    pad.set_clip_path(hand_patch)
                    cv.ax.add_patch(pad)

                # Organic, irregular multi-blob contact smudge fading smoothly into skin (alpha -> 0)
                res = 36
                yy, xx = np.mgrid[0:res, 0:res]
                nx = (xx + 0.5) / res * 2.0 - 1.0
                ny = (yy + 0.5) / res * 2.0 - 1.0
                rad = np.deg2rad(ang)
                ca, sa = np.cos(rad), np.sin(rad)
                span = max(ew, eh) * 0.58
                lx = (nx * span * ca + ny * span * sa) / (ew * 0.5)
                ly = (-nx * span * sa + ny * span * ca) / (eh * 0.5)
                inside = (lx ** 2 + ly ** 2) <= 0.92 ** 2

                # Combine 2-3 irregular Gaussian lobes per finger for natural contact shape
                blobs = spec.get("blobs", [(0.0, 0.0, 0.65, 0.60, spec.get("peak", 0.9))])
                field = np.zeros_like(lx)
                for ox, oy, sx, sy, amp in blobs:
                    d2 = ((lx - ox) / sx) ** 2 + ((ly - oy) / sy) ** 2
                    field += amp * np.exp(-d2 * 1.1)
                field = np.clip(field, 0.0, 1.0)

                cmap_name = spec.get("cmap", "inferno")
                cmap = plt.get_cmap(cmap_name)
                if cmap_name == "Greys":
                    # Darker, more prominent grey-dominated contact smudge
                    val = np.clip(0.46 + 0.54 * field, 0.0, 1.0)
                else:
                    # Real CONTACT_LUT (inferno): mostly dark purple-red with warm localized peak
                    val = np.clip(0.18 + 0.76 * field, 0.0, 1.0)
                rgba = cmap(val)
                # Smooth edge falloff near the pad boundary so it blends naturally with skin
                edge_fade = np.clip((0.94 ** 2 - (lx ** 2 + ly ** 2)) / 0.14, 0.0, 1.0)
                rgba[..., 3] = np.where(inside, np.clip(field * 1.65 * edge_fade, 0.0, 0.98), 0.0)
                cv.ax.imshow(
                    rgba,
                    extent=[wx - span, wx + span, wy - span, wy + span],
                    origin="lower",
                    interpolation="bilinear",
                    zorder=9,
                )
            else:
                pad = Ellipse(
                    (wx, wy),
                    width=ew,
                    height=eh,
                    angle=ang,
                    facecolor=spec,
                    edgecolor=pad_ec,
                    linewidth=0.32,
                    zorder=8,
                )
                pad.set_clip_path(hand_patch)
                cv.ax.add_patch(pad)

    return tip_centers


def _draw_color_remap_center(cv, cx, cy):
    """Draw the center color-remap cue: inferno strip / arrow / uint8 grayscale strip."""
    grad = np.linspace(0, 1, 32).reshape(1, -1)
    bw, bh = 3.2, 0.85
    cv.ax.imshow(grad, cmap="inferno", extent=[cx - bw / 2.0, cx + bw / 2.0, cy + 1.05, cy + 1.05 + bh],
                 origin="lower", zorder=7)
    cv.ax.add_patch(Rectangle((cx - bw / 2.0, cy + 1.05), bw, bh, fc="none", ec=C_SLATE, lw=0.35, zorder=8))

    cv.arrow([(cx - 1.9, cy), (cx + 1.9, cy)], color=C_HERO, lw=0.6, head=0.7)

    cv.ax.imshow(grad, cmap="gray", extent=[cx - bw / 2.0, cx + bw / 2.0, cy - 1.05 - bh, cy - 1.05],
                 origin="lower", zorder=7)
    cv.ax.add_patch(Rectangle((cx - bw / 2.0, cy - 1.05 - bh), bw, bh, fc="none", ec=C_HERO, lw=0.35, zorder=8))


def draw_route_a_col3_proj(cv, cx, cy):
    """Col 3: 5x5 luma grid -> distal-phalanx local frame cKDTree (d <= 6.5 mm) + v/180."""
    # Shift right by +0.75 mm to balance visual mass against the lighter numeric labels on the right.
    dx_shift = 0.75
    gx0, gy0 = cx - 10.6 + dx_shift, cy - 2.625
    cell = 1.08
    for r in range(5):
        for c in range(5):
            d = abs(r - 2) + abs(c - 2)
            col = C_ACCENT if d <= 1 else ("#FDBA74" if d == 2 else "#334155")
            cv.ax.add_patch(Rectangle((gx0 + c * cell, gy0 + r * cell), cell * 0.86, cell * 0.86,
                                      fc=col, ec="none", zorder=7))
    cv.arrow([(cx - 4.9 + dx_shift, cy), (cx - 3.4 + dx_shift, cy)], color=C_SLATE, lw=0.55, head=0.65)

    kx = cx + 0.4 + dx_shift
    cv.ax.add_patch(
        FancyBboxPatch((kx - 3.3, cy - 3.3), 6.6, 6.6, boxstyle="round,pad=0,rounding_size=2.2",
                       fc=C_SLATE_FILL, ec=C_SLATE, lw=0.5, zorder=6)
    )
    cv.ax.add_patch(
        Circle((kx, cy), 2.35, fc=C_ACCENT_SOFT, ec=C_ACCENT, lw=0.5, ls="--", zorder=7)
    )
    cv.arrow([(kx, cy), (kx + 1.8, cy)], color=C_INK, lw=0.5, head=0.5)
    cv.arrow([(kx, cy), (kx, cy - 1.8)], color=C_EVAL, lw=0.5, head=0.5)
    cv.ax.add_patch(Circle((kx, cy), 0.38, fc=C_HERO, ec="white", lw=0.25, zorder=9))
    for dx, dy in [(-1.0, 0.8), (0.8, 1.1), (-0.9, -0.7)]:
        cv.ax.add_patch(Circle((kx + dx, cy + dy), 0.30, fc=C_ACCENT, ec="none", zorder=8))

    bx0, by0 = cx + 5.2 + dx_shift, cy - 3.1
    grad = np.linspace(0, 1, 32).reshape(-1, 1)
    cv.ax.imshow(grad, cmap="inferno", extent=[bx0, bx0 + 1.5, by0, by0 + 6.2], origin="lower", zorder=7)
    cv.ax.add_patch(Rectangle((bx0, by0), 1.5, 6.2, fc="none", ec=C_SLATE, lw=0.45, zorder=8))
    cv.text(bx0 + 1.9, by0 + 5.5, "180", size=SZ_SUB, color=C_INK, ha="left")
    cv.text(bx0 + 1.9, by0 + 0.6, "0", size=SZ_SUB, color=C_INK, ha="left")


def draw_route_a_col4_export(cv, cx, cy):
    """Col 4: Route A 5-finger hand organic contact smudges -> uint8 grayscale."""
    finger_blobs = {
        "thumb":  [(-0.20, 0.14, 0.65, 0.55, 0.98), (0.12, -0.10, 0.45, 0.40, 0.58)],
        "index":  [(0.16, 0.18, 0.60, 0.65, 0.92), (-0.16, -0.06, 0.42, 0.42, 0.52)],
        "middle": [(-0.12, 0.22, 0.58, 0.50, 0.80)],
        "ring":   [(0.18, 0.10, 0.48, 0.40, 0.50)],
        "pinky":  [(-0.16, -0.10, 0.52, 0.58, 0.84), (0.12, 0.20, 0.38, 0.36, 0.48)],
    }
    raw_5 = {k: {"cmap": "inferno", "blobs": v} for k, v in finger_blobs.items()}
    u8_5 = {k: {"cmap": "Greys", "blobs": v} for k, v in finger_blobs.items()}
    _draw_svg_hand(
        cv, cx - 6.6, cy, height=7.8,
        hand_fc="#E8D4C5", hand_ec="#A6826B",
        finger_colors=raw_5, pad_ec="#A6826B",
    )
    _draw_color_remap_center(cv, cx, cy)
    # Keep the hand's own skin color unchanged; only the sensor pads remap to uint8 grayscale.
    _draw_svg_hand(
        cv, cx + 6.6, cy, height=7.8,
        hand_fc="#E8D4C5", hand_ec="#A6826B",
        finger_colors=u8_5, pad_ec="#475569",
    )


# ---------------------------------------------------------------------------
# Route B (Two-Finger Parallel Grippers) — Aligned 4 Macro-Stages
# ---------------------------------------------------------------------------

def draw_route_b_col1_prep(cv, cx, cy):
    """Col 1: Real 3D parallel gripper + real visuo-tactile depth + flip-only align."""
    place_rgba_image(cv, "assets/fig1/_urdf_gelsight.png", cx=cx - 7.3, cy=cy, w=6.8, z=7, alpha=0.96)
    cv.arrow([(cx - 3.6, cy), (cx - 2.2, cy)], color=C_EVAL, lw=0.55, head=0.65)
    h_f = place_rgba_image(cv, "assets/fig1/_tactile_force.png", cx=cx + 0.3, cy=cy + 1.9, w=3.4, z=7)
    cv.ax.add_patch(
        Rectangle((cx - 1.4, cy + 1.9 - h_f / 2.0), 3.4, h_f, fc="none", ec=C_ACCENT, lw=0.55, zorder=8)
    )
    h_g = place_rgba_image(cv, "assets/fig1/_tactile_grating.png", cx=cx + 0.3, cy=cy - 1.9, w=3.4, z=7)
    cv.ax.add_patch(
        Rectangle((cx - 1.4, cy - 1.9 - h_g / 2.0), 3.4, h_g, fc="none", ec=C_HERO, lw=0.55, zorder=8)
    )
    cv.ax.add_patch(
        FancyBboxPatch((cx + 3.4, cy - 2.9), 7.2, 5.8, boxstyle="round,pad=0,rounding_size=0.5",
                       fc=C_EVAL_FILL, ec=C_EVAL, lw=0.5, zorder=6)
    )
    cv.text(cx + 7.0, cy, "Flip\nonly", size=SZ_SUB, color=C_EVAL, weight="bold")


def draw_route_b_col2_align(cv, cx, cy):
    """Col 2: Static 778-vertex MANO container (real downward 3D gripper left->thumb, right->index)."""
    gx = cx - 6.7
    gw = 7.0
    gh = place_rgba_image(cv, "assets/fig2/_urdf_gelsight_down_static.png", cx=gx, cy=cy - 0.1, w=gw, z=7, alpha=0.98)

    hx = cx + 6.6
    tips = _draw_svg_hand(
        cv, hx, cy - 0.1, height=7.8,
        hand_fc="#E8D4C5", hand_ec="#A6826B",
        finger_colors={"thumb": C_ACCENT, "index": C_HERO},
    )

    # Exact pad centroids on the rendered downward 3D gripper (norm_x = +-0.05, norm_y_up = -0.376)
    p_left = (gx - 0.05 * gw, (cy - 0.1) - 0.375 * gh)
    p_right = (gx + 0.05 * gw, (cy - 0.1) - 0.377 * gh)
    p_thumb = tips["thumb"]
    p_index = tips["index"]

    ts = np.linspace(0, 1, 28)
    # Left (orange) pad -> Thumb: quadratic Bezier dipping cleanly below the right jaw (zero overlap with blue pad)
    c_orange = (0.38 * p_left[0] + 0.62 * p_thumb[0], min(p_left[1], p_thumb[1]) - 2.35)
    ox = (1 - ts) ** 2 * p_left[0] + 2 * (1 - ts) * ts * c_orange[0] + ts ** 2 * p_thumb[0]
    oy = (1 - ts) ** 2 * p_left[1] + 2 * (1 - ts) * ts * c_orange[1] + ts ** 2 * p_thumb[1]
    cv.ax.plot(ox, oy, color=C_ACCENT, lw=0.75, ls="--", zorder=9)
    # Right (blue) pad -> Index: quadratic Bezier arcing cleanly through the center gap
    c_blue = (0.52 * p_right[0] + 0.48 * p_index[0], p_index[1] + 0.35)
    bx = (1 - ts) ** 2 * p_right[0] + 2 * (1 - ts) * ts * c_blue[0] + ts ** 2 * p_index[0]
    by = (1 - ts) ** 2 * p_right[1] + 2 * (1 - ts) * ts * c_blue[1] + ts ** 2 * p_index[1]
    cv.ax.plot(bx, by, color=C_HERO, lw=0.75, ls="--", zorder=9)


def draw_route_b_col3_proj(cv, cx, cy):
    """Col 3: 5-triangle 3D barycentric lookup -> per-texel avg & NaN-preserving merge."""
    lx = cx - 5.35
    cy_m = cy - 0.6
    pts = [
        (lx - 4.7, cy_m + 2.4),
        (lx - 0.3, cy_m + 3.6),
        (lx + 4.3, cy_m + 1.8),
        (lx - 2.9, cy_m - 2.2),
        (lx + 2.3, cy_m - 2.4),
        (lx - 0.3, cy_m + 0.4),
    ]
    tris = [(0, 1, 5), (1, 2, 5), (0, 3, 5), (3, 4, 5), (2, 4, 5)]
    for i, (a, b, c) in enumerate(tris):
        poly = Polygon(
            [pts[a], pts[b], pts[c]],
            fc=C_EVAL_SOFT if i == 1 else "#F8FAFC",
            ec="#B48A6E",
            lw=0.5,
            zorder=6,
        )
        cv.ax.add_patch(poly)
    p_proj = (lx + 1.4, cy_m + 1.8)
    for v_idx in (1, 2, 5):
        cv.line([p_proj, pts[v_idx]], color=C_EVAL, lw=0.45, dashed=True)
    cv.ax.add_patch(Circle(p_proj, 0.48, fc=C_ACCENT, ec="white", lw=0.3, zorder=9))

    cv.arrow([(cx - 0.65, cy), (cx + 2.95, cy)], color=C_EVAL, lw=0.55, head=0.7)
    cv.text(cx + 1.15, cy + 3.6, r"$b_0,b_1,b_2$", size=SZ_SUB, color=C_EVAL)

    mx0, my0 = cx + 3.45, cy - 3.3
    cell = 1.65
    grid = [
        [0, 0, 1, 0],
        [0, 1, 2, 1],
        [1, 2, 1, 0],
        [0, 0, 0, 0],
    ]
    for r in range(4):
        for c in range(4):
            v = grid[r][c]
            if v == 2:
                fc, hatch = C_ACCENT, None
            elif v == 1:
                fc, hatch = "#1E293B", None
            else:
                fc, hatch = "#F1F5F9", "///"
            cv.ax.add_patch(
                Rectangle((mx0 + c * cell, my0 + r * cell), cell, cell,
                          fc=fc, ec=C_SLATE_SOFT, lw=0.3, hatch=hatch, zorder=6)
            )


def draw_route_b_col4_export(cv, cx, cy):
    """Col 4: Route B real downward 3D gripper pad colors -> SVG hand thumb & index uint8 grayscale."""
    place_rgba_image(cv, "assets/fig2/_urdf_gelsight_down_uv.png", cx=cx - 6.7, cy=cy - 0.1, w=7.0, z=7, alpha=0.98)
    _draw_color_remap_center(cv, cx, cy)
    # Keep the hand's own skin color unchanged; only the mapped thumb/index sensor pads change to uint8 grayscale.
    _draw_svg_hand(
        cv, cx + 6.6, cy - 0.1, height=7.8,
        hand_fc="#E8D4C5", hand_ec="#A6826B",
        finger_colors={
            "thumb": {"cmap": "Greys", "blobs": [(-0.16, 0.16, 0.64, 0.56, 0.96), (0.14, -0.08, 0.44, 0.38, 0.56)]},
            "index": {"cmap": "Greys", "blobs": [(0.14, 0.12, 0.58, 0.60, 0.82)]},
        },
        pad_ec="#475569",
    )


def draw():
    uv_path = "assets/fig1/_mano_uv_atlas.png"
    mask_path = "assets/fig1/_mano_mask_sample.png"

    cv = Canvas(W, H, family="serif", size=SZ_BODY)

    stage_xs = [4.0, 30.8, 57.6, 84.4]
    stage_w = 24.4

    # -----------------------------------------------------------------------
    # Shared 4-Stage Column Header Bar (top = 1.6..6.4)
    # -----------------------------------------------------------------------
    col_headers = [
        "1. Input Prep",
        "2. MANO Align",
        "3. Surface Project",
        "4. UV Export",
    ]
    for i, (sx, hdr) in enumerate(zip(stage_xs, col_headers)):
        cv.box(
            sx, T(6.4), stage_w, 4.8,
            fill=C_HERO_FILL, stroke=C_HERO_SOFT, lw=0.5, radius=1.0, z=4,
        )
        cv.text(sx + stage_w / 2.0, T(4.0), hdr, size=SZ_BODY, weight="bold", color=C_HERO)
        if i < 3:
            cv.arrow([(sx + stage_w + 0.2, T(4.0)), (stage_xs[i + 1] - 0.2, T(4.0))],
                     color=C_HERO, lw=0.55, head=0.9)

    # -----------------------------------------------------------------------
    # Top Lane: Route A - Articulated Dexterous Hands (x = 2..112, top = 8.0..40.6)
    # -----------------------------------------------------------------------
    cv.box(
        2.0, T(40.6), 110.0, 32.6,
        fill=C_SLATE_FILL, stroke=C_SLATE, lw=LW_BASE, radius=2.0, z=2,
    )
    cv.text(57.0, T(10.8), "Route A: Articulated Dexterous Hands", size=SZ_HEAD, weight="bold", color=C_INK)
    # Shared skeleton legend in Route A header (above Col 1 & Col 2) to unify URDF FK and Pose Alignment encodings.
    ly_leg = T(10.8)
    cv.line([(4.6, ly_leg), (7.6, ly_leg)], color=C_SLATE, lw=0.75)
    cv.ax.add_patch(Circle((6.1, ly_leg), 0.34, fc=C_SLATE, ec="white", lw=0.2, zorder=9))
    cv.text(8.3, ly_leg, "Robot", size=SZ_SUB, color=C_SLATE, weight="bold", ha="left")
    cv.line([(16.8, ly_leg), (19.8, ly_leg)], color=C_HERO, lw=0.75, dashed=True)
    cv.ax.add_patch(Circle((18.3, ly_leg), 0.34, fc=C_HERO, ec="white", lw=0.2, zorder=9))
    cv.text(20.5, ly_leg, "MANO", size=SZ_SUB, color=C_HERO, weight="bold", ha="left")

    # Unified Col 4 title & caption across Route A/B to highlight shared color-remapping semantics.
    a_titles = ["URDF FK", "Pose Alignment", "Local cKDTree", "UV Color Remap"]
    a_captions = [
        ("21 kpts, 16 vecs, luma", C_INK),
        ("Bone-vec LSQ + Umeyama", C_HERO),
        ("Local frame, d ≤ 6.5 mm", C_INK),
        ("Color remap → 128² uint8", C_INK),
    ]
    a_drawers = [draw_route_a_col1_prep, draw_route_a_col2_align, draw_route_a_col3_proj, draw_route_a_col4_export]

    for i, (sx, title, (cap, cap_col), fn) in enumerate(zip(stage_xs, a_titles, a_captions, a_drawers)):
        cv.box(sx, T(38.8), stage_w, 25.0, fill="white", stroke=C_SLATE_SOFT, lw=0.5, radius=1.4, z=4)
        cv.text(sx + stage_w / 2.0, T(16.2), title, size=SZ_BODY, weight="bold", color=C_INK)
        fn(cv, sx + stage_w / 2.0, T(26.3))
        cv.text(sx + stage_w / 2.0, T(36.4), cap, size=SZ_SUB, color=cap_col)
        if i < 3:
            cv.arrow([(sx + stage_w, T(26.3)), (stage_xs[i + 1], T(26.3))], color=C_SLATE, lw=LW_BASE, head=1.1)

    # -----------------------------------------------------------------------
    # Bottom Lane: Route B - Two-Finger Parallel Grippers (x = 2..112, top = 43.0..79.2)
    # -----------------------------------------------------------------------
    cv.box(
        2.0, T(79.2), 110.0, 36.2,
        fill=C_EVAL_FILL, stroke=C_EVAL, lw=LW_BASE, radius=2.0, z=2,
    )
    cv.text(57.0, T(45.8), "Route B: Two-Finger Parallel Grippers", size=SZ_HEAD, weight="bold", color=C_EVAL)

    # Unified Col 4 title & caption with Route A (same color/intensity remap to canonical uint8 UV).
    b_titles = ["Depth Align", "Static Container", "Barycentric Map", "UV Color Remap"]
    b_captions = [
        ("float32 depth, flip-only", C_INK),
        ("Static 778-vert container", C_EVAL),
        ("Barycentric, NaN ≠ 0", C_INK),
        ("Color remap → 128² uint8", C_INK),
    ]
    b_drawers = [draw_route_b_col1_prep, draw_route_b_col2_align, draw_route_b_col3_proj, draw_route_b_col4_export]

    for i, (sx, title, (cap, cap_col), fn) in enumerate(zip(stage_xs, b_titles, b_captions, b_drawers)):
        cv.box(sx, T(74.0), stage_w, 25.0, fill="white", stroke=C_EVAL_SOFT, lw=0.5, radius=1.4, z=4)
        cv.text(sx + stage_w / 2.0, T(51.4), title, size=SZ_BODY, weight="bold", color=C_EVAL)
        fn(cv, sx + stage_w / 2.0, T(61.5))
        cv.text(sx + stage_w / 2.0, T(71.6), cap, size=SZ_SUB, color=cap_col)
        if i < 3:
            cv.arrow([(sx + stage_w, T(61.5)), (stage_xs[i + 1], T(61.5))], color=C_EVAL, lw=LW_BASE, head=1.1)

    cv.text(
        57.0, T(76.7),
        "Fallback for uncalibrated grippers (StandardPad): train-frozen median / p99 trend projection onto thumb & index footprints.",
        size=SZ_SUB, style="italic", color=C_EVAL,
    )

    # -----------------------------------------------------------------------
    # Merge Bus -> Right Hero Column
    # -----------------------------------------------------------------------
    cv.line([(112.0, T(24.3)), (114.6, T(24.3))], color=C_SLATE, lw=LW_BASE)
    cv.line([(112.0, T(61.1)), (114.6, T(61.1))], color=C_EVAL, lw=LW_BASE)
    cv.line([(114.6, T(24.3)), (114.6, T(61.1))], color=C_HERO, lw=LW_BASE)
    cv.arrow([(114.6, T(42.7)), (117.5, T(42.7))], color=C_HERO, lw=LW_HERO, head=1.4)

    # -----------------------------------------------------------------------
    # Right Column (HERO): Unified 128x256 UV Atlas (x = 117.5..154.0)
    # -----------------------------------------------------------------------
    cv.box(
        117.5, T(79.2), 36.5, 71.2,
        fill=C_HERO_FILL, stroke=C_HERO, lw=LW_HERO, radius=2.2, z=3,
    )
    cv.text(135.75, T(13.0), "Unified 128 × 256", size=SZ_HEAD, weight="bold", color=C_HERO)
    cv.text(135.75, T(16.8), "Canonical MANO UV Atlas", size=SZ_BODY, weight="bold", color=C_INK)

    # Step 1: MANO UV Validity Mask (top, matching Figure 1)
    cv.image(mask_path, cx=135.75, cy=T(27.5), w=25.0, z=8, knockout_white=False, crop=False)
    cv.ax.add_patch(
        Rectangle((135.75 - 12.5, T(27.5) - 6.25), 25.0, 12.5, fc="none", ec=C_SLATE, lw=0.45, zorder=9)
    )

    # Explicit Mask -> UV mapping connector
    cv.arrow([(135.75, T(34.5)), (135.75, T(42.0))], color=C_HERO, lw=LW_HERO, head=1.3)
    cv.text(133.8, T(38.25), "MANO UV Mask", size=SZ_SUB, weight="bold", color=C_INK, ha="right")
    cv.text(137.7, T(38.25), "Maps valid surface", size=SZ_SUB, color=C_HERO, style="italic", ha="left")

    # Step 2: Bimanual UV Maps (split Left = Schematic vs. Right = Stored UV Map)
    uv_w, uv_h = 32.0, 16.0
    uv_cy = T(55.5)
    uv_x0 = 135.75 - uv_w / 2.0
    uv_y0 = uv_cy - uv_h / 2.0

    cv.text(135.75 - uv_w / 4.0, T(45.2), "Hand Layout", size=SZ_SUB, color=C_MUTED)
    cv.text(135.75 + uv_w / 4.0, T(45.2), "Stored UV Map", size=SZ_SUB, weight="bold", color=C_HERO)

    cv.image(uv_path, cx=135.75, cy=uv_cy, w=uv_w, z=8, knockout_white=False, crop=False)
    cv.ax.add_patch(
        Rectangle((uv_x0, uv_y0), uv_w / 2.0, uv_h, fc="none", ec=C_SLATE, lw=0.45, zorder=9)
    )
    cv.ax.add_patch(
        Rectangle((135.75, uv_y0), uv_w / 2.0, uv_h, fc="none", ec=C_HERO, lw=0.75, zorder=10)
    )
    cv.ax.plot([135.75, 135.75], [uv_y0, uv_y0 + uv_h], color="white", lw=0.5, ls="--", alpha=0.7, zorder=11)
    cv.text(135.75, T(67.2), "Left: schematic · Right: stored UV map", size=SZ_SUB, color=C_MUTED)
    cv.text(135.75, T(72.6), "128 × 128 per hand · 30 FPS", size=SZ_SUB, color=C_INK)

    # -----------------------------------------------------------------------
    # Geometric audits & export (SVG + PNG, no PDF per AGENTS.md)
    # -----------------------------------------------------------------------
    overlaps = cv.overlaps()
    text_ov = cv.text_overlaps()
    crossings = cv.wire_crossings()
    print(f"Audits -> overlaps: {len(overlaps)}, text_overlaps: {len(text_ov)}, wire_crossings: {len(crossings)}")
    if overlaps:
        print("  overlaps:", overlaps)
    if text_ov:
        print("  text_overlaps:", text_ov)
    if crossings:
        print("  wire_crossings:", crossings)

    assert not overlaps and not text_ov and not crossings

    os.makedirs("img", exist_ok=True)
    os.makedirs("assets/fig2", exist_ok=True)
    out_stem = "img/fig2_retarget_pipeline_v2"
    cv.save(out_stem, formats=("svg", "png"))
    print(f"Saved: {out_stem}.svg, {out_stem}.png")


if __name__ == "__main__":
    draw()
