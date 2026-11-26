"""Generate Figure 2: Dual-Route Cross-Embodiment Retargeting & Data Contract.

Usage:
    uv run --with matplotlib --with numpy --with pillow python scripts/gen_fig2.py
"""
import os
import sys

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

SKILL_SCRIPTS = os.path.expanduser("~/.agents/skills/cs-paper-figure/scripts")
sys.path.insert(0, SKILL_SCRIPTS)
from paperkit import Canvas  # noqa: E402

# ---------------------------------------------------------------------------
# Visual system constants (matching Figure 1, ACL full-width 156 mm)
# ---------------------------------------------------------------------------
W, H = 156.0, 74.0
T = lambda top_mm: H - top_mm  # noqa: E731

C_SLATE = "#475569"
C_SLATE_FILL = "#EDF1F5"
C_SLATE_SOFT = "#E2E8F0"
C_INK = "#0F172A"
C_MUTED = "#64748B"

C_HERO = "#1D4ED8"
C_HERO_FILL = "#EFF6FF"
C_HERO_SOFT = "#BFDBFE"

C_EVAL = "#0F766E"
C_EVAL_FILL = "#F0FDFA"
C_EVAL_SOFT = "#CCFBF1"

C_ACCENT = "#D97706"

LW_BASE = 0.6
LW_HERO = 0.85
SZ_HEAD = 7.8
SZ_BODY = 6.5
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
    wrist = (kx, cy - 3.6 * scale)
    fingers = [
        [(-2.4, -1.5), (-3.5, -0.3), (-4.2, 0.8)],
        [(-1.5, -0.6), (-1.8, 1.3), (-2.0, 2.8), (-2.1, 3.9)],
        [(-0.3, -0.4), (-0.3, 1.6), (-0.3, 3.3), (-0.3, 4.5)],
        [(0.9, -0.6), (1.1, 1.3), (1.2, 2.9), (1.3, 4.0)],
        [(2.0, -1.1), (2.4, 0.6), (2.7, 1.9), (2.9, 2.9)],
    ]
    ca, sa = np.cos(rot), np.sin(rot)
    cv.ax.add_patch(Circle(wrist, 0.44, fc=color, ec="white", lw=0.3, zorder=9))
    for f_pts in fingers:
        pts = [wrist]
        for dx, dy in f_pts:
            rx = (dx * ca - dy * sa) * scale
            ry = (dx * sa + dy * ca) * scale
            pts.append((kx + rx, cy + ry))
        for i in range(len(pts) - 1):
            cv.line([pts[i], pts[i + 1]], color=color, lw=0.65, dashed=dashed)
        for j, pt in enumerate(pts[1:]):
            is_tip = (j == len(pts) - 2)
            cv.ax.add_patch(
                Circle(pt, 0.38 if is_tip else 0.28, fc=color, ec="white", lw=0.25, zorder=9)
            )


def draw_route_a_stage1_fk(cv, cx, cy):
    """Real Sharpa hand + 21-keypoint Sharpa skeleton -> MANO 21-keypoint skeleton."""
    place_rgba_image(cv, "assets/fig1/_urdf_sharpa.png", cx=cx - 8.2, cy=cy + 1.0, w=6.4, z=7, alpha=0.92)
    _draw_mini_skeleton(cv, cx - 1.5, cy + 1.0, scale=0.92, color=C_SLATE)
    cv.arrow([(cx + 1.6, cy + 1.0), (cx + 3.3, cy + 1.0)], color=C_SLATE, lw=0.6, head=0.7)
    _draw_mini_skeleton(cv, cx + 7.5, cy + 1.0, scale=0.92, color=C_HERO)
    cv.text(cx, cy - 6.8, "21 keypoints (FK)", size=SZ_SUB, color=C_INK)


def draw_route_a_stage2_retarget(cv, cx, cy):
    """Inverse vector retargeting: Sharpa FK vs. MANO(z) -> Vector Loss -> LSQ loop."""
    lw_s = 0.55
    # Left box: Sharpa
    cv.ax.add_patch(
        FancyBboxPatch((cx - 10.4, cy + 2.4), 8.6, 3.2, boxstyle="round,pad=0,rounding_size=0.45",
                       fc=C_SLATE_FILL, ec=C_SLATE, lw=lw_s, zorder=6)
    )
    cv.text(cx - 6.1, cy + 4.0, "Sharpa", size=SZ_SUB, color=C_INK)
    # Right box: MANO(z)
    cv.ax.add_patch(
        FancyBboxPatch((cx - 0.2, cy + 2.4), 9.2, 3.2, boxstyle="round,pad=0,rounding_size=0.45",
                       fc=C_HERO_FILL, ec=C_HERO, lw=lw_s, zorder=6)
    )
    cv.text(cx + 4.4, cy + 4.0, "MANO(z)", size=SZ_SUB, color=C_HERO)
    # Bottom: Loss + LSQ
    cv.ax.add_patch(
        FancyBboxPatch((cx - 8.4, cy - 3.4), 14.8, 3.2, boxstyle="round,pad=0,rounding_size=0.45",
                       fc="#FFF7ED", ec=C_ACCENT, lw=lw_s, zorder=6)
    )
    cv.text(cx - 1.0, cy - 1.8, "Loss + LSQ", size=SZ_SUB, color=C_INK)
    # Down arrows into Loss
    cv.arrow([(cx - 6.1, cy + 2.4), (cx - 4.0, cy - 0.2)], color=C_SLATE, lw=0.5, head=0.65)
    cv.arrow([(cx + 4.4, cy + 2.4), (cx + 2.2, cy - 0.2)], color=C_HERO, lw=0.5, head=0.65)
    # Dashed optimization feedback loop from LSQ back to MANO(z)
    cv.line([(cx + 6.4, cy - 1.8), (cx + 10.4, cy - 1.8), (cx + 10.4, cy + 4.0)], color=C_HERO, lw=0.55, dashed=True)
    cv.arrow([(cx + 10.4, cy + 4.0), (cx + 9.0, cy + 4.0)], color=C_HERO, lw=0.55, head=0.7)
    cv.text(cx, cy - 6.8, "min ||u_M - u_S||^2 + reg", size=SZ_SUB, color=C_HERO)


def draw_route_a_stage3_proj(cv, cx, cy):
    """11-point Umeyama similarity alignment (s, R, t): Before -> Aligned."""
    _draw_mini_skeleton(cv, cx - 6.8, cy + 0.6, scale=0.96, color=C_SLATE)
    _draw_mini_skeleton(cv, cx - 5.6, cy + 0.8, scale=0.86, color=C_HERO, dashed=True, rot=-0.26)
    cv.arrow([(cx - 1.3, cy + 0.6), (cx + 1.5, cy + 0.6)], color=C_INK, lw=0.6, head=0.75)
    cv.text(cx + 0.1, cy + 4.4, "s, R, t", size=SZ_SUB, color=C_INK)
    _draw_mini_skeleton(cv, cx + 6.6, cy + 0.6, scale=0.96, color=C_SLATE)
    _draw_mini_skeleton(cv, cx + 6.6, cy + 0.6, scale=0.95, color=C_HERO, dashed=True, rot=0.0)
    cv.text(cx, cy - 6.8, "11-pt Umeyama", size=SZ_SUB, color=C_INK)


def draw_route_a_stage4_raster(cv, cx, cy):
    """Tactile retarget: 240x240 sensor grid -> 3D NN pad -> MANO fingertip UV."""
    # Left: 5x5 tactile grid with warm orange center
    gx0, gy0 = cx - 10.6, cy - 1.8
    cell = 1.12
    for r in range(5):
        for c in range(5):
            d = abs(r - 2) + abs(c - 2)
            col = C_ACCENT if d <= 1 else ("#FDBA74" if d == 2 else "#334155")
            cv.ax.add_patch(Rectangle((gx0 + c * cell, gy0 + r * cell), cell * 0.86, cell * 0.86,
                                      fc=col, ec="none", zorder=7))
    cv.arrow([(cx - 4.7, cy + 1.0), (cx - 3.1, cy + 1.0)], color=C_SLATE, lw=0.55, head=0.7)
    # Middle: 3D fingertip pad with NN points
    cv.ax.add_patch(
        FancyBboxPatch((cx - 2.8, cy - 2.4), 5.2, 6.6, boxstyle="round,pad=0,rounding_size=1.8",
                       fc=C_SLATE_FILL, ec=C_SLATE, lw=0.5, zorder=6)
    )
    for r in range(3):
        for c in range(3):
            is_c = (r >= 1 and c == 1) or (r == 1)
            cv.ax.add_patch(
                Circle((cx - 1.5 + c * 1.3, cy - 0.6 + r * 1.45), 0.36,
                       fc=C_ACCENT if is_c else C_SLATE, ec="none", zorder=8)
            )
    cv.arrow([(cx + 2.6, cy + 1.0), (cx + 4.2, cy + 1.0)], color=C_SLATE, lw=0.55, head=0.7)
    # Right: MANO fingertip UV texture with 1-px dilated contact
    cv.ax.add_patch(
        FancyBboxPatch((cx + 4.5, cy - 2.4), 5.2, 6.6, boxstyle="round,pad=0,rounding_size=1.8",
                       fc=C_SLATE_FILL, ec=C_HERO, lw=0.55, zorder=6)
    )
    cv.ax.add_patch(Circle((cx + 7.1, cy + 1.4), 1.85, fc=C_ACCENT, alpha=0.35, ec="none", zorder=7))
    cv.ax.add_patch(Circle((cx + 7.1, cy + 1.4), 1.15, fc=C_ACCENT, ec="none", zorder=8))
    cv.text(cx, cy - 6.8, "3D NN + UV rasterize", size=SZ_SUB, color=C_INK)


def draw_route_b_stage1_ref(cv, cx, cy):
    """Step 1: Two-finger parallel gripper pads mapped to fixed MANO thumb & index."""
    # Left: 2-jaw parallel gripper
    gx = cx - 6.0
    cv.ax.add_patch(
        FancyBboxPatch((gx - 4.0, cy + 3.2), 8.0, 1.7, boxstyle="round,pad=0,rounding_size=0.4",
                       fc=C_SLATE_SOFT, ec=C_SLATE, lw=0.5, zorder=6)
    )
    cv.ax.add_patch(
        FancyBboxPatch((gx - 3.6, cy - 3.2), 1.5, 6.4, boxstyle="round,pad=0,rounding_size=0.35",
                       fc=C_SLATE_SOFT, ec=C_SLATE, lw=0.5, zorder=6)
    )
    cv.ax.add_patch(
        FancyBboxPatch((gx + 2.1, cy - 3.2), 1.5, 6.4, boxstyle="round,pad=0,rounding_size=0.35",
                       fc=C_SLATE_SOFT, ec=C_SLATE, lw=0.5, zorder=6)
    )
    # Orange (left) and Blue (right) inner sensor pads
    cv.ax.add_patch(
        FancyBboxPatch((gx - 2.1, cy - 2.6), 0.9, 3.2, boxstyle="round,pad=0,rounding_size=0.25",
                       fc=C_ACCENT, ec="none", zorder=7)
    )
    cv.ax.add_patch(
        FancyBboxPatch((gx + 1.2, cy - 2.6), 0.9, 3.2, boxstyle="round,pad=0,rounding_size=0.25",
                       fc=C_HERO, ec="none", zorder=7)
    )

    # Right: Fixed MANO hand silhouette with thumb (orange) & index (blue) regions
    hx = cx + 6.2
    cv.ax.add_patch(
        FancyBboxPatch((hx - 2.8, cy - 3.2), 5.4, 4.4, boxstyle="round,pad=0,rounding_size=1.0",
                       fc="#E8D4C5", ec="#A6826B", lw=0.5, zorder=6)
    )
    for fx, fh in [(-2.0, 3.9), (-0.6, 4.6), (0.8, 4.0), (2.1, 3.1)]:
        cv.ax.add_patch(
            FancyBboxPatch((hx + fx - 0.55, cy + 0.7), 1.1, fh, boxstyle="round,pad=0,rounding_size=0.45",
                           fc="#E8D4C5", ec="#A6826B", lw=0.45, zorder=6)
        )
    thumb = Polygon(
        [(hx - 2.6, cy - 2.0), (hx - 4.3, cy - 0.3), (hx - 3.5, cy + 0.7), (hx - 2.0, cy - 0.7)],
        closed=True, fc="#E8D4C5", ec="#A6826B", lw=0.45, zorder=6,
    )
    cv.ax.add_patch(thumb)
    cv.ax.add_patch(Circle((hx - 3.6, cy + 0.05), 0.68, fc=C_ACCENT, ec="none", zorder=8))
    cv.ax.add_patch(
        FancyBboxPatch((hx - 2.42, cy + 2.7), 0.85, 1.5, boxstyle="round,pad=0,rounding_size=0.3",
                       fc=C_HERO, ec="none", zorder=8)
    )

    # Dashed correspondence arcs
    ts = np.linspace(0, 1, 18)
    ox = (1 - ts) * (gx - 1.2) + ts * (hx - 3.8)
    oy = (1 - ts) * (cy - 1.0) + ts * (cy - 0.05) - 1.6 * np.sin(np.pi * ts)
    cv.ax.plot(ox, oy, color=C_ACCENT, lw=0.75, ls="--", zorder=7)
    bx = (1 - ts) * (gx + 1.2) + ts * (hx - 2.4)
    by = (1 - ts) * (cy - 0.8) + ts * (cy + 3.4) + 1.0 * np.sin(np.pi * ts)
    cv.ax.plot(bx, by, color=C_HERO, lw=0.75, ls="--", zorder=7)

    cv.text(cx, cy - 6.8, "Thumb / Index pads", size=SZ_SUB, color=C_INK)


def draw_route_b_stage2_depth(cv, cx, cy):
    """Step 2: Align flat sensor surface to curved MANO finger pad."""
    lx = cx - 5.6
    ts = np.linspace(-1.0, 1.0, 21)
    arch_x_l = lx + ts * 4.4
    arch_y_l = cy - 2.4 + 5.6 * (1.0 - ts * ts)
    cv.ax.plot(arch_x_l, arch_y_l, color="#B48A6E", lw=1.8, solid_capstyle="round", zorder=6)
    cv.line([(lx - 4.6, cy + 0.4), (lx + 4.6, cy + 0.4)], color=C_HERO, lw=1.2)

    cv.arrow([(cx - 1.0, cy + 0.6), (cx + 1.2, cy + 0.6)], color=C_EVAL, lw=0.6, head=0.75)

    rx = cx + 5.6
    arch_x_r = rx + ts * 4.4
    arch_y_r = cy - 2.4 + 5.6 * (1.0 - ts * ts)
    cv.ax.plot(arch_x_r, arch_y_r, color="#B48A6E", lw=1.8, solid_capstyle="round", zorder=6)
    for k in range(0, len(ts), 2):
        cv.ax.add_patch(Circle((arch_x_r[k], arch_y_r[k] + 0.55), 0.30, fc=C_HERO, ec="none", zorder=8))
    cv.text(cx, cy - 6.8, "Local surface align", size=SZ_SUB, color=C_INK)


def draw_route_b_stage3_encode(cv, cx, cy):
    """Step 3: Project each aligned point onto nearest MANO mesh triangle."""
    pts = [
        (cx - 9.2, cy + 2.8),   # 0: top-left
        (cx - 0.6, cy + 4.2),   # 1: top-mid
        (cx + 9.0, cy + 2.2),   # 2: top-right
        (cx - 5.6, cy - 2.6),   # 3: bot-left
        (cx + 4.4, cy - 2.8),   # 4: bot-right
        (cx - 0.5, cy + 0.4),   # 5: center vertex
    ]
    tris = [(0, 1, 5), (1, 2, 5), (0, 3, 5), (3, 4, 5), (2, 4, 5)]
    for i, (a, b, c) in enumerate(tris):
        poly = Polygon(
            [pts[a], pts[b], pts[c]],
            fc=C_EVAL_SOFT if i == 1 else "#F8FAFC",
            ec="#B48A6E",
            lw=0.55,
            zorder=6,
        )
        cv.ax.add_patch(poly)
    p_proj = (cx + 2.8, cy + 2.2)
    for v_idx in (1, 2, 5):
        cv.line([p_proj, pts[v_idx]], color=C_EVAL, lw=0.45, dashed=True)
    cv.ax.add_patch(Circle(p_proj, 0.62, fc=C_ACCENT, ec="white", lw=0.35, zorder=9))
    cv.text(cx, cy - 6.8, "Barycentric (<1 mm)", size=SZ_SUB, color=C_INK)


def draw_route_b_stage4_pad(cv, cx, cy):
    """Step 4: Fixed-range UV encoding & StandardPad projection."""
    bx0, by0 = cx - 9.4, cy - 2.8
    grad = np.linspace(0, 1, 32).reshape(-1, 1)
    cv.ax.imshow(grad, cmap="turbo", extent=[bx0, bx0 + 1.8, by0, by0 + 7.4], origin="lower", zorder=7)
    cv.ax.add_patch(Rectangle((bx0, by0), 1.8, 7.4, fc="none", ec=C_SLATE, lw=0.45, zorder=8))
    cv.arrow([(bx0 + 2.5, cy + 0.9), (cx - 3.2, cy + 0.9)], color=C_EVAL, lw=0.55, head=0.75)
    gx0, gy0 = cx - 2.8, cy - 2.8
    cv.ax.add_patch(Rectangle((gx0, gy0), 11.6, 7.4, fc="#1E293B", ec=C_EVAL, lw=0.55, zorder=6))
    cv.ax.add_patch(Circle((gx0 + 3.5, gy0 + 4.2), 1.45, fc=C_ACCENT, ec="none", zorder=7))
    cv.ax.add_patch(Circle((gx0 + 8.1, gy0 + 2.9), 1.45, fc="#38BDF8", ec="none", zorder=7))
    cv.text(cx, cy - 6.8, "Fixed range + mask", size=SZ_SUB, color=C_INK)


def draw():
    uv_path = "assets/fig1/_mano_uv_atlas.png"
    mask_path = "assets/fig1/_mano_mask_sample.png"

    cv = Canvas(W, H, family="serif", size=SZ_BODY)

    # -----------------------------------------------------------------------
    # Top Lane: Route A - 22-DoF Dexterous Hand (x = 2..112, top = 2.5..36.0)
    # -----------------------------------------------------------------------
    cv.box(
        2.0, T(36.0), 110.0, 33.5,
        fill=C_SLATE_FILL, stroke=C_SLATE, lw=LW_BASE, radius=2.0, z=2,
    )
    cv.text(57.0, T(5.4), "Route A: 22-DoF Dexterous Hand (Sharpa)", size=SZ_HEAD, weight="bold", color=C_INK)

    stage_xs = [4.0, 30.8, 57.6, 84.4]
    stage_w = 24.4
    a_titles = ["1. URDF FK", "2. Vector Optim.", "3. Umeyama Align", "4. Tactile & UV"]
    a_drawers = [draw_route_a_stage1_fk, draw_route_a_stage2_retarget, draw_route_a_stage3_proj, draw_route_a_stage4_raster]

    for i, (sx, title, fn) in enumerate(zip(stage_xs, a_titles, a_drawers)):
        cv.box(sx, T(34.2), stage_w, 25.8, fill="white", stroke=C_SLATE_SOFT, lw=0.5, radius=1.4, z=4)
        cv.text(sx + stage_w / 2.0, T(10.8), title, size=SZ_BODY, weight="bold", color=C_INK)
        fn(cv, sx + stage_w / 2.0, T(24.6))
        if i < 3:
            cv.arrow([(sx + stage_w, T(21.3)), (stage_xs[i + 1], T(21.3))], color=C_SLATE, lw=LW_BASE, head=1.1)

    # -----------------------------------------------------------------------
    # Bottom Lane: Route B - Parallel Gripper (x = 2..112, top = 38.5..72.0)
    # -----------------------------------------------------------------------
    cv.box(
        2.0, T(72.0), 110.0, 33.5,
        fill=C_EVAL_FILL, stroke=C_EVAL, lw=LW_BASE, radius=2.0, z=2,
    )
    cv.text(57.0, T(41.4), "Route B: Parallel Gripper (TacVerse / StandardPad)", size=SZ_HEAD, weight="bold", color=C_EVAL)

    b_titles = ["1. Pad-to-MANO", "2. Surface Align", "3. Mesh Project", "4. UV & StandardPad"]
    b_drawers = [draw_route_b_stage1_ref, draw_route_b_stage2_depth, draw_route_b_stage3_encode, draw_route_b_stage4_pad]

    for i, (sx, title, fn) in enumerate(zip(stage_xs, b_titles, b_drawers)):
        cv.box(sx, T(70.2), stage_w, 25.8, fill="white", stroke=C_EVAL_SOFT, lw=0.5, radius=1.4, z=4)
        cv.text(sx + stage_w / 2.0, T(46.8), title, size=SZ_BODY, weight="bold", color=C_EVAL)
        fn(cv, sx + stage_w / 2.0, T(60.6))
        if i < 3:
            cv.arrow([(sx + stage_w, T(57.3)), (stage_xs[i + 1], T(57.3))], color=C_EVAL, lw=LW_BASE, head=1.1)

    # -----------------------------------------------------------------------
    # Merge Bus -> Right Hero Column
    # -----------------------------------------------------------------------
    cv.line([(112.0, T(19.2)), (114.6, T(19.2))], color=C_SLATE, lw=LW_BASE)
    cv.line([(112.0, T(55.2)), (114.6, T(55.2))], color=C_EVAL, lw=LW_BASE)
    cv.line([(114.6, T(19.2)), (114.6, T(55.2))], color=C_HERO, lw=LW_BASE)
    cv.arrow([(114.6, T(37.2)), (117.5, T(37.2))], color=C_HERO, lw=LW_HERO, head=1.4)

    # -----------------------------------------------------------------------
    # Right Column (HERO): Unified 128x256 UV & Data Contract (x = 117.5..154.0)
    # -----------------------------------------------------------------------
    cv.box(
        117.5, T(72.0), 36.5, 69.5,
        fill=C_HERO_FILL, stroke=C_HERO, lw=LW_HERO, radius=2.2, z=3,
    )
    cv.text(135.75, T(5.8), "Unified 128 × 256", size=SZ_HEAD, weight="bold", color=C_HERO)
    cv.text(135.75, T(9.4), "Bimanual UV & Contract", size=SZ_BODY, weight="bold", color=C_INK)

    # Bimanual UV Maps
    cv.image(uv_path, cx=135.75, cy=T(20.6), w=31.0, z=8, knockout_white=False, crop=False)
    cv.ax.add_patch(
        Rectangle((135.75 - 15.5, T(20.6) - 7.75), 31.0, 15.5, fc="none", ec=C_HERO, lw=0.5, zorder=9)
    )
    cv.text(135.75, T(30.4), "Left + Right Hand UV Contact Map", size=SZ_SUB, color=C_MUTED)

    # Official MANO UV Mask
    cv.image(mask_path, cx=135.75, cy=T(39.6), w=23.0, z=8, knockout_white=False, crop=False)
    cv.ax.add_patch(
        Rectangle((135.75 - 11.5, T(39.6) - 5.75), 23.0, 11.5, fc="none", ec=C_SLATE, lw=0.45, zorder=9)
    )
    cv.text(135.75, T(47.4), "MANO UV Mask (Obs-0 vs. Unk)", size=SZ_SUB, weight="bold", color=C_INK)

    # 3-Part Data Contract card
    cv.box(
        119.8, T(69.8), 31.9, 19.2,
        fill="white", stroke=C_HERO, lw=0.55, radius=1.4, z=6,
    )
    cv.text(135.75, T(53.2), "3-Part Data Contract", size=SZ_BODY, weight="bold", color=C_HERO)
    cv.rule(121.8, 149.7, T(55.4), lw=0.45, color=C_HERO_SOFT)
    cv.text(121.8, T(58.4), "1. Proxy vs. Force (N, N·m)", size=SZ_SUB, color=C_INK, ha="left")
    cv.text(121.8, T(62.2), "2. Observed-0 vs. Unknown", size=SZ_SUB, color=C_INK, ha="left")
    cv.text(121.8, T(66.0), "3. Fixed Global Scale", size=SZ_SUB, color=C_INK, ha="left")

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
