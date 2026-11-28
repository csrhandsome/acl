"""Generate Algorithm 1: Single-Frame 22-DoF Dexterous-Hand Tactile Retargeting (sharpa_retarget.py).

Usage:
    uv run --with matplotlib --with numpy --with pillow python scripts/gen_alg_sharpa.py
"""
import os
import sys

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

SKILL_SCRIPTS = os.path.expanduser("~/.gemini/config/skills/cs-paper-figure/scripts")
if not os.path.exists(SKILL_SCRIPTS):
    SKILL_SCRIPTS = os.path.expanduser("~/.agents/skills/cs-paper-figure/scripts")
sys.path.insert(0, SKILL_SCRIPTS)
from paperkit import Canvas  # noqa: E402

# ---------------------------------------------------------------------------
# Visual system constants (matching Fig. 1, Fig. 2 & Alg. 2, ACL full-width 156 mm)
# ---------------------------------------------------------------------------
W, H = 156.0, 108.0
T = lambda top_mm: H - top_mm  # noqa: E731

C_INK = "#0F172A"
C_SLATE = "#475569"
C_MUTED = "#64748B"
C_SLATE_FILL = "#F1F5F9"
C_SLATE_SOFT = "#CBD5E1"

C_HERO = "#1D4ED8"
C_HERO_FILL = "#EFF6FF"
C_HERO_SOFT = "#BFDBFE"

C_EVAL = "#0F766E"
C_EVAL_FILL = "#F0FDFA"
C_EVAL_SOFT = "#CCFBF1"

C_ACCENT = "#D97706"
C_ACCENT_FILL = "#FFFBEB"
C_ACCENT_SOFT = "#FDE68A"

LW_BASE = 0.6
LW_HERO = 0.85

SZ_HEAD = 7.6
SZ_BODY = 6.2
SZ_CODE = 5.8


def draw_code_block(cv, x_num, x_code, top_start, dy, rows, hi_rows=None, hi_color=None, code_w=89.5):
    """Draw numbered pseudocode rows with syntax-colored rich segments."""
    hi_rows = hi_rows or set()
    for idx, (lineno, segs) in enumerate(rows):
        cy = T(top_start + (idx + 0.5) * dy)
        if lineno in hi_rows and hi_color:
            cv.ax.add_patch(
                Rectangle(
                    (x_code - 0.8, cy - dy / 2.0),
                    code_w,
                    dy,
                    fc=hi_color,
                    ec="none",
                    zorder=3,
                )
            )
        cv.text(
            x_num,
            cy,
            f"{lineno:2d}",
            size=SZ_CODE,
            color=C_MUTED,
            ha="right",
            family="monospace",
            z=10,
        )
        cv.rich(x_code, cy, segs, size=SZ_CODE, ha="left", va="center", family="monospace", z=10)


def draw_stage_rail(cv, x, top_y, bot_y, color, title, fill=None):
    """Left vertical stage accent bar + stage eyebrow title."""
    if fill:
        cv.ax.add_patch(
            FancyBboxPatch(
                (x, T(bot_y)),
                96.5,
                bot_y - top_y,
                boxstyle="round,pad=0,rounding_size=1.0",
                fc=fill,
                ec="none",
                zorder=1,
            )
        )
    cv.ax.add_patch(
        Rectangle((x, T(bot_y)), 1.2, bot_y - top_y, fc=color, ec="none", zorder=5)
    )
    cv.text(
        x + 2.6,
        T(top_y + 2.1),
        title,
        size=SZ_BODY,
        weight="bold",
        color=color,
        ha="left",
        z=10,
    )


def _draw_mini_skeleton(cv, kx, cy, scale, color, dashed=False, rot=0.0):
    """Compact 5-finger 21-keypoint hand skeleton (matches Fig. 2 grammar)."""
    wrist = (kx, cy - 3.4 * scale)
    fingers = [
        [(-2.3, -1.4), (-3.4, -0.3), (-4.0, 0.8)],
        [(-1.4, -0.5), (-1.7, 1.3), (-1.9, 2.7), (-2.0, 3.7)],
        [(-0.3, -0.4), (-0.3, 1.5), (-0.3, 3.1), (-0.3, 4.2)],
        [(0.8, -0.5), (1.0, 1.3), (1.1, 2.8), (1.2, 3.8)],
        [(1.9, -1.0), (2.3, 0.6), (2.6, 1.8), (2.8, 2.7)],
    ]
    ca, sa = np.cos(rot), np.sin(rot)
    cv.ax.add_patch(Circle(wrist, 0.42, fc=color, ec="white", lw=0.25, zorder=9))
    for f_pts in fingers:
        pts = [wrist]
        for dx, dy in f_pts:
            rx = (dx * ca - dy * sa) * scale
            ry = (dx * sa + dy * ca) * scale
            pts.append((kx + rx, cy + ry))
        for i in range(len(pts) - 1):
            cv.line([pts[i], pts[i + 1]], color=color, lw=LW_BASE, dashed=dashed)
        for j, pt in enumerate(pts[1:]):
            is_tip = (j == len(pts) - 2)
            cv.ax.add_patch(
                Circle(pt, 0.36 if is_tip else 0.25, fc=color, ec="white", lw=0.2, zorder=9)
            )


def draw_diag_stage1(cv, x0, top_y, w, h):
    """Band 1: 22-DoF q -> URDF FK (21 kpts + 16 unit vecs) + 240x240 luma (skip missing)."""
    cy = T(top_y + h / 2.0)
    cv.box(x0, T(top_y + h), w, h, fill=C_SLATE_FILL, stroke=C_SLATE_SOFT, lw=LW_BASE, radius=1.2, z=2)

    # Left: q in R^22 -> 21-kpt skeleton
    cv.box(x0 + 2.0, cy - 2.6, 10.2, 5.2, fill="white", stroke=C_SLATE, lw=LW_BASE, radius=0.8, z=6)
    cv.text(x0 + 7.1, cy, "q (22-DoF)", size=SZ_CODE, color=C_INK, weight="bold", z=10)
    cv.arrow([(x0 + 12.6, cy), (x0 + 16.0, cy)], color=C_SLATE, lw=LW_BASE, head=0.8)
    _draw_mini_skeleton(cv, x0 + 21.8, cy + 0.6, scale=0.95, color=C_SLATE)
    cv.text(x0 + 21.8, cy - 5.4, "P_S (21), u_S (16)", size=SZ_CODE, color=C_INK, z=10)

    # Divider
    cv.line([(x0 + 31.2, cy - 5.8), (x0 + 31.2, cy + 5.8)], color=C_SLATE_SOFT, lw=LW_BASE, dashed=True)

    # Right: 5 finger tactile patches (4 active + 1 ghosted/missing)
    gx, gy = x0 + 33.0, cy - 1.2
    for i in range(5):
        missing = (i == 4)
        bx = gx + i * 3.5
        fc = "#E2E8F0" if missing else "#1E293B"
        ec = C_MUTED if missing else C_ACCENT
        cv.ax.add_patch(
            Rectangle((bx, gy), 2.9, 5.2, fc=fc, ec=ec, lw=0.45, ls="--" if missing else "-", zorder=6)
        )
        if not missing:
            cv.ax.add_patch(Circle((bx + 1.45, gy + 2.6), 0.85, fc=C_ACCENT, alpha=0.85, ec="none", zorder=7))
        else:
            cv.text(bx + 1.45, gy + 2.6, "x", size=SZ_CODE, color=C_MUTED, z=8)
    cv.text(x0 + 41.6, cy - 5.4, "240^2 luma (skip miss)", size=SZ_CODE, color=C_MUTED, z=10)


def draw_diag_stage2(cv, x0, top_y, w, h):
    """Band 2 (HERO): Warm-started TRF LSQ loop + 11-pt Umeyama alignment."""
    cy = T(top_y + h / 2.0)
    cv.box(x0, T(top_y + h), w, h, fill=C_HERO_FILL, stroke=C_HERO, lw=LW_HERO, radius=1.2, z=2)

    # Left half: Warm-start + TRF optimization loop
    cv.box(x0 + 2.0, cy + 3.6, 13.4, 4.4, fill="white", stroke=C_HERO, lw=LW_BASE, radius=0.8, z=6)
    cv.text(x0 + 8.7, cy + 5.8, "MANO(r, z)", size=SZ_CODE, color=C_HERO, weight="bold", z=10)

    cv.box(x0 + 2.0, cy - 4.2, 18.6, 4.6, fill=C_ACCENT_FILL, stroke=C_ACCENT, lw=LW_BASE, radius=0.8, z=6)
    cv.text(x0 + 11.3, cy - 1.9, "TRF LSQ (<=40)", size=SZ_CODE, color=C_INK, weight="bold", z=10)

    # Down arrow MANO -> LSQ
    cv.arrow([(x0 + 8.7, cy + 3.6), (x0 + 8.7, cy + 0.4)], color=C_HERO, lw=LW_BASE, head=0.8)
    # Dashed warm-start / iteration loop back
    cv.line([(x0 + 20.6, cy - 1.9), (x0 + 23.4, cy - 1.9), (x0 + 23.4, cy + 5.8)], color=C_HERO, lw=LW_BASE, dashed=True)
    cv.arrow([(x0 + 23.4, cy + 5.8), (x0 + 15.4, cy + 5.8)], color=C_HERO, lw=LW_BASE, head=0.8)
    cv.text(x0 + 12.2, cy - 7.8, "warm-start x_{t-1} -> x'_t", size=SZ_CODE, color=C_HERO, z=10)

    # Arrow -> Umeyama seating
    cv.arrow([(x0 + 24.6, cy + 1.2), (x0 + 28.6, cy + 1.2)], color=C_INK, lw=LW_BASE, head=0.9)
    cv.text(x0 + 26.6, cy + 3.8, "s,R,t", size=SZ_CODE, color=C_INK, weight="bold", z=10)

    # Right half: Before/After aligned skeletons (11 anchor pts)
    _draw_mini_skeleton(cv, x0 + 34.4, cy + 1.4, scale=0.95, color=C_SLATE)
    _draw_mini_skeleton(cv, x0 + 35.4, cy + 1.6, scale=0.85, color=C_HERO, dashed=True, rot=-0.24)
    cv.arrow([(x0 + 39.2, cy + 1.2), (x0 + 41.6, cy + 1.2)], color=C_HERO, lw=LW_BASE, head=0.75)
    _draw_mini_skeleton(cv, x0 + 46.2, cy + 1.4, scale=0.95, color=C_SLATE)
    _draw_mini_skeleton(cv, x0 + 46.2, cy + 1.4, scale=0.94, color=C_HERO, dashed=True, rot=0.0)
    cv.text(x0 + 40.4, cy - 7.8, "11-pt Umeyama seat", size=SZ_CODE, color=C_INK, z=10)


def draw_diag_stage3(cv, x0, top_y, w, h):
    """Band 3: Distal local frame (w, -t, p) + cKDTree query + 6.5 mm gate."""
    cy = T(top_y + h / 2.0)
    cv.box(x0, T(top_y + h), w, h, fill=C_ACCENT_FILL, stroke=C_ACCENT, lw=LW_HERO, radius=1.2, z=2)

    # Left: Fingertip pad with local orthonormal axes (w, -t, p)
    fx, fy = x0 + 9.2, cy + 0.6
    cv.ax.add_patch(
        FancyBboxPatch((fx - 5.8, fy - 6.6), 11.6, 13.4, boxstyle="round,pad=0,rounding_size=4.2",
                       fc="white", ec=C_SLATE, lw=LW_BASE, zorder=6)
    )
    # Local axes from fingertip origin
    tip = (fx, fy + 4.4)
    cv.arrow([tip, (fx + 3.3, fy + 4.4)], color=C_HERO, lw=LW_BASE, head=0.65)
    cv.arrow([tip, (fx, fy - 0.6)], color=C_EVAL, lw=LW_BASE, head=0.65)
    cv.ax.add_patch(Circle(tip, 0.55, fc=C_INK, ec="white", lw=0.25, zorder=9))
    cv.text(fx + 3.6, fy + 4.4, "w", size=SZ_CODE, color=C_HERO, ha="left", weight="bold", z=10)
    cv.text(fx + 1.1, fy - 0.4, "-t", size=SZ_CODE, color=C_EVAL, ha="left", weight="bold", z=10)
    # Distal vertices (w > 0.2)
    for dx, dy in [(-2.4, -2.0), (0.0, -3.2), (2.4, -2.0), (-1.8, 1.4), (1.8, 1.4)]:
        cv.ax.add_patch(Circle((fx + dx, fy + dy), 0.52, fc=C_ACCENT, ec="none", zorder=8))
    cv.text(fx, cy - 8.4, "distal frame", size=SZ_CODE, color=C_INK, z=10)

    # Arrow -> cKDTree & 6.5 mm radius gate
    cv.arrow([(x0 + 16.0, cy + 0.6), (x0 + 20.0, cy + 0.6)], color=C_ACCENT, lw=LW_BASE, head=0.85)

    # Middle: cKDTree radius circle (6.5 mm)
    kx, ky = x0 + 28.4, cy + 0.6
    cv.ax.add_patch(Circle((kx, ky), 6.2, fc=C_ACCENT_SOFT, ec=C_ACCENT, lw=LW_BASE, ls="--", zorder=6))
    cv.ax.add_patch(Circle((kx, ky), 0.75, fc=C_HERO, ec="white", lw=0.3, zorder=9))
    # Nearby sensor points (inside vs outside 6.5 mm)
    for px, py, inside in [(-2.2, 1.6, True), (1.8, -1.8, True), (0.4, 2.8, True), (6.8, 4.4, False), (-6.6, -4.2, False)]:
        cv.ax.add_patch(
            Circle((kx + px, ky + py), 0.55, fc=C_ACCENT if inside else C_MUTED, ec="none", zorder=8)
        )
    cv.line([(kx, ky), (kx - 2.2, ky + 1.6)], color=C_INK, lw=LW_BASE)
    cv.text(kx, cy - 8.4, "cKDTree (d <= 6.5mm)", size=SZ_CODE, color=C_ACCENT, weight="bold", z=10)

    # Arrow -> inferno blend
    cv.arrow([(x0 + 35.8, cy + 0.6), (x0 + 39.2, cy + 0.6)], color=C_ACCENT, lw=LW_BASE, head=0.85)

    # Right: inferno + SKIN ramp
    rx, ry = x0 + 40.4, cy - 4.4
    grad = np.linspace(0, 1, 32).reshape(-1, 1)
    cv.ax.imshow(grad, cmap="inferno", extent=[rx, rx + 3.0, ry, ry + 10.0], origin="lower", zorder=7)
    cv.ax.add_patch(Rectangle((rx, ry), 3.0, 10.0, fc="none", ec=C_SLATE, lw=0.45, zorder=8))
    cv.text(rx + 3.8, ry + 9.2, "180", size=SZ_CODE, color=C_INK, ha="left", family="monospace", z=10)
    cv.text(rx + 3.8, ry + 0.8, "0", size=SZ_CODE, color=C_INK, ha="left", family="monospace", z=10)
    cv.text(rx + 4.2, cy - 8.4, "v / 180", size=SZ_CODE, color=C_INK, z=10)


def draw_diag_stage4(cv, x0, top_y, w, h):
    """Band 4: 1024^2 pose-invariant UV face-ID map -> NN 128^2 -> LUT inverse -> uint8."""
    cy = T(top_y + h / 2.0)
    cv.box(x0, T(top_y + h), w, h, fill=C_HERO_FILL, stroke=C_HERO, lw=LW_HERO, radius=1.2, z=2)

    # Left: 1024^2 UV atlas with 1-px seam dilation
    ux, uy = x0 + 2.2, cy - 3.6
    cv.ax.add_patch(Rectangle((ux, uy), 10.6, 9.0, fc="#18181C", ec=C_SLATE, lw=LW_BASE, zorder=6))
    cv.ax.add_patch(Circle((ux + 3.8, uy + 5.4), 2.0, fc="#D2A88C", ec="none", zorder=7))
    cv.ax.add_patch(Circle((ux + 7.1, uy + 3.8), 1.6, fc=C_ACCENT, ec="none", zorder=8))
    cv.text(ux + 5.3, uy - 2.3, "1024^2 + 1px", size=SZ_CODE, color=C_INK, z=10)

    cv.arrow([(x0 + 13.4, cy + 0.9), (x0 + 17.6, cy + 0.9)], color=C_HERO, lw=LW_BASE, head=0.8)
    cv.text(x0 + 15.5, cy + 3.2, "NN", size=SZ_CODE, color=C_HERO, weight="bold", z=10)

    # Middle: CONTACT_LUT inverse box
    cv.box(x0 + 18.2, cy - 3.2, 15.6, 8.2, fill="white", stroke=C_HERO, lw=LW_BASE, radius=0.9, z=6)
    cv.text(x0 + 26.0, cy + 2.4, "Contact LUT", size=SZ_CODE, color=C_HERO, weight="bold", z=10)
    cv.text(x0 + 26.0, cy - 0.8, "argmin L2^2", size=SZ_CODE, color=C_INK, z=10)
    cv.text(x0 + 26.0, uy - 2.3, "bg / res>35^2->0", size=SZ_CODE, color=C_MUTED, z=10)

    cv.arrow([(x0 + 34.4, cy + 0.9), (x0 + 38.0, cy + 0.9)], color=C_HERO, lw=LW_BASE, head=0.8)

    # Right: Final 128x128 uint8 PNG
    ox, oy = x0 + 38.6, cy - 3.6
    cv.ax.add_patch(Rectangle((ox, oy), 11.2, 9.0, fc="#18181C", ec=C_HERO, lw=LW_HERO, zorder=6))
    cv.ax.add_patch(Circle((ox + 4.1, oy + 5.6), 1.55, fc="#E2E8F0", ec="none", zorder=7))
    cv.ax.add_patch(Circle((ox + 7.3, oy + 3.8), 1.85, fc="white", ec="none", zorder=7))
    cv.text(ox + 5.6, oy - 2.3, "128^2 (u8)", size=SZ_CODE, color=C_HERO, weight="bold", z=10)


def draw():
    cv = Canvas(W, H, family="serif", size=SZ_BODY)

    # -----------------------------------------------------------------------
    # Booktabs Algorithm Header
    # -----------------------------------------------------------------------
    cv.rule(2.0, W - 2.0, T(1.8), lw=LW_HERO, color=C_INK)
    cv.rich(
        2.4,
        T(4.6),
        [
            ("Algorithm 1  ", C_HERO, "bold"),
            ("Single-Frame 22-DoF Dexterous-Hand Tactile Retargeting  ", C_INK, "bold"),
            ("(sharpa_retarget.py)", C_MUTED, "normal"),
        ],
        size=SZ_HEAD,
        ha="left",
    )
    cv.rule(2.0, W - 2.0, T(7.2), lw=LW_BASE, color=C_SLATE)

    cv.rich(
        2.4,
        T(9.8),
        [
            ("Input:  ", C_INK, "bold"),
            ("MANO model H; factory 3D sensor maps S (TH, 4F, 240^2x3); 22-DoF q_L, q_R; partial tactile frames; warm-start x_L, x_R", C_INK),
        ],
        size=SZ_CODE,
        ha="left",
    )
    cv.rich(
        2.4,
        T(12.8),
        [
            ("Output: ", C_INK, "bold"),
            ("Bimanual 128 x 128 uint8 contact-proxy PNGs f{t}_{L,R}.png; updated warm-start x'_L, x'_R; posed fits & vertex maps", C_INK),
        ],
        size=SZ_CODE,
        ha="left",
    )
    cv.rule(2.0, W - 2.0, T(15.0), lw=LW_BASE, color=C_SLATE_SOFT)

    # Column headers
    cv.text(50.0, T(17.1), "Pseudocode  (retarget_frame)", size=SZ_BODY, weight="bold", color=C_SLATE)
    cv.text(128.2, T(17.1), "Aligned Geometric & Optimization Flow", size=SZ_BODY, weight="bold", color=C_HERO)

    # -----------------------------------------------------------------------
    # Stage Bands (exact vertical alignment between Left Code and Right Diagrams)
    # -----------------------------------------------------------------------
    x_left = 2.0
    x_num = 7.4
    x_code = 8.6
    x_diag = 102.0
    w_diag = 52.0
    dy = 2.95

    # Stage 1: Tactile Preproc & URDF FK (top 19.2 .. 34.8, h = 15.6)
    s1_t, s1_b = 19.2, 34.8
    draw_stage_rail(cv, x_left, s1_t, s1_b, C_SLATE, "Stage 1: Tactile Luminance Preproc & Pinocchio URDF FK", fill=None)
    rows1 = [
        (1, [("for ", C_HERO, "bold"), ("h in {L, R}, f in 5_fingers ", C_INK), ("if ", C_HERO, "bold"), ("frames[h_f] present:", C_INK)]),
        (2, [("  G[h_f] <- resize_area(luma(frames[h_f]), 240x240)", C_INK), (" # no fab", C_MUTED)]),
        (3, [("for ", C_HERO, "bold"), ("h in {L, R} ", C_INK), ("do  ", C_HERO, "bold"), ("FK(URDF_h, qpos_by_name(q_h))", C_SLATE, "bold")]),
        (4, [("  P_S <- mp21_wrist_centered(FK)*1000; u_S <- unit(task-orig)", C_INK)]),
    ]
    draw_code_block(cv, x_num, x_code, s1_t + 3.4, dy, rows1, code_w=89.5)
    draw_diag_stage1(cv, x_diag, s1_t, w_diag, s1_b - s1_t)

    # Stage 2: Inverse Bone-Vector Fit & Umeyama (top 36.4 .. 57.6, h = 21.2)
    s2_t, s2_b = 36.4, 57.6
    cv.rule(x_left, 98.5, T(35.6), lw=0.4, color=C_SLATE_SOFT)
    draw_stage_rail(cv, x_left, s2_t, s2_b, C_HERO, "Stage 2: Warm-Started Inverse Bone-Vector Fit & 11-Pt Umeyama", fill=None)
    rows2 = [
        (5, [("  x_0 <- x_h if valid else 0_18", C_INK), ("   # x = [r in R^3, z in R^15]", C_MUTED)]),
        (6, [("  x'_h <- ", C_INK), ("TRF_least_squares", C_HERO, "bold"), ("(x_0, max_nfev=40, ftol=1e-5):", C_INK)]),
        (7, [("    min ||unit(bone_16(MANO_flat(r,z,mir_L))) - u_S||^2 + ||0.08z||^2", C_HERO, "bold")]),
        (8, [("  V, J <- rodrigues(r) @ MANO_flat(z, mirror_x=(h==L))", C_INK)]),
        (9, [("  s, R, t <- ", C_INK), ("Umeyama", C_HERO, "bold"), ("(mp21(V, J)[A_11], P_S[A_11])", C_INK), ("  # 11 anchors", C_MUTED)]),
        (10, [("  V <- s*(R @ V) + t;  J <- s*(R @ J) + t", C_INK), ("  # 778 verts, 16 jts", C_MUTED)]),
    ]
    draw_code_block(cv, x_num, x_code, s2_t + 3.4, dy, rows2, hi_rows={6, 7}, hi_color=C_HERO_SOFT, code_w=89.5)
    draw_diag_stage2(cv, x_diag, s2_t, w_diag, s2_b - s2_t)

    # Stage 3: Distal-Phalanx Local Frame & cKDTree Projection (top 59.2 .. 83.4, h = 24.2)
    s3_t, s3_b = 59.2, 83.4
    cv.rule(x_left, 98.5, T(58.4), lw=0.4, color=C_SLATE_SOFT)
    draw_stage_rail(cv, x_left, s3_t, s3_b, C_ACCENT, "Stage 3: Distal-Phalanx Local Frame & cKDTree Projection", fill=None)
    rows3 = [
        (11, [("  dist <- +inf;  val <- 0;  hit <- False", C_INK), ("  # over 778 verts", C_MUTED)]),
        (12, [("  for ", C_HERO, "bold"), ("each observed f in 5_fingers ", C_INK), ("do", C_HERO, "bold")]),
        (13, [("    F_f <- ", C_INK), ("distal_frame", C_ACCENT, "bold"), ("(V, J, weights[:, j_f] > 0.2)", C_INK), ("  # [w,-t,p]", C_MUTED)]),
        (14, [("    q_loc <- (F_f.rot^T @ (V[ids] - tip)) / scale", C_INK)]),
        (15, [("    d_loc, nn <- ", C_INK), ("cKDTree", C_ACCENT, "bold"), ("(S_f).query(q_loc);  d <- d_loc * scale", C_INK)]),
        (16, [("    if d < dist: dist <- d; val <- G[h_f][nn] if d<=6.5mm else 0", C_ACCENT, "bold")]),
        (17, [("  rgb <- blend(SKIN, inferno(clip(val/180,0,1)), w=clip(2.2*lvl,0,1))", C_INK)]),
    ]
    draw_code_block(cv, x_num, x_code, s3_t + 3.4, dy, rows3, hi_rows={16}, hi_color=C_ACCENT_SOFT, code_w=89.5)
    draw_diag_stage3(cv, x_diag, s3_t, w_diag, s3_b - s3_t)

    # Stage 4: Official UV Atlas Rasterization & LUT Recovery (top 85.0 .. 104.2, h = 19.2)
    s4_t, s4_b = 85.0, 104.2
    cv.rule(x_left, 98.5, T(84.2), lw=0.4, color=C_SLATE_SOFT)
    draw_stage_rail(cv, x_left, s4_t, s4_b, C_HERO, "Stage 4: Official UV Rasterization & Contact LUT Recovery", fill=None)
    rows4 = [
        (18, [("  face_rgb <- mean(rgb[faces_h], axis=1)", C_INK), ("  # L flips winding", C_MUTED)]),
        (19, [("  L <- ", C_INK), ("cached_label_map", C_HERO, "bold"), ("(official_uv(h), 1024, dilate=1px)", C_INK)]),
        (20, [("  tex <- BG(24,24,28); tex[L>=0] <- face_rgb[L]; tex <- NN(tex, 128)", C_INK)]),
        (21, [("  C <- ", C_INK), ("inv_CONTACT_LUT", C_HERO, "bold"), ("(tex; bg_L1<=18 or res>35^2 -> 0)", C_HERO, "bold")]),
        (22, [("  encode_png(f{t:04d}_{h}.png, C);  return pngs, x'_L, x'_R", C_INK)]),
    ]
    draw_code_block(cv, x_num, x_code, s4_t + 3.4, dy, rows4, hi_rows={21}, hi_color=C_HERO_SOFT, code_w=89.5)
    draw_diag_stage4(cv, x_diag, s4_t, w_diag, s4_b - s4_t)

    # -----------------------------------------------------------------------
    # Stage Connectors (horizontal ticks + right-column vertical flow)
    # -----------------------------------------------------------------------
    for (st, sb, col) in [(s1_t, s1_b, C_SLATE), (s2_t, s2_b, C_HERO), (s3_t, s3_b, C_ACCENT), (s4_t, s4_b, C_HERO)]:
        ym = T((st + sb) / 2.0)
        cv.arrow([(98.8, ym), (101.8, ym)], color=col, lw=LW_BASE, head=0.8)

    for (b_prev, t_next, col) in [(s1_b, s2_t, C_SLATE), (s2_b, s3_t, C_HERO), (s3_b, s4_t, C_ACCENT)]:
        cv.arrow([(128.0, T(b_prev)), (128.0, T(t_next))], color=col, lw=LW_BASE, head=0.8)

    # Bottom booktabs rule
    cv.rule(2.0, W - 2.0, T(105.8), lw=LW_HERO, color=C_INK)

    # -----------------------------------------------------------------------
    # Audits & export (SVG + PNG only, no PDF per AGENTS.md)
    # -----------------------------------------------------------------------
    overlaps = cv.overlaps()
    text_ov = cv.text_overlaps()
    crossings = cv.wire_crossings()
    print(f"Sharpa Alg Audits -> overlaps: {len(overlaps)}, text_overlaps: {len(text_ov)}, wire_crossings: {len(crossings)}")
    if overlaps:
        print("  overlaps:", overlaps)
    if text_ov:
        print("  text_overlaps:", text_ov)
    if crossings:
        print("  wire_crossings:", crossings)

    assert not overlaps and not text_ov and not crossings

    os.makedirs("img", exist_ok=True)
    os.makedirs("assets/alg_sharpa", exist_ok=True)
    out_stem = "img/alg_sharpa_retarget"
    cv.save(out_stem, formats=("svg", "png"))
    print(f"Saved: {out_stem}.svg, {out_stem}.png")


if __name__ == "__main__":
    draw()
