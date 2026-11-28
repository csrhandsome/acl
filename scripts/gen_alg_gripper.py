"""Generate Algorithm 2: Single-Frame Parallel-Gripper Tactile Retargeting (gripper_retarget.py).

Usage:
    uv run --with matplotlib --with numpy --with pillow python scripts/gen_alg_gripper.py
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
# Visual system constants (matching Fig. 1 & Fig. 2, ACL full-width 156 mm)
# ---------------------------------------------------------------------------
W, H = 156.0, 96.0
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


def draw_code_block(cv, x_num, x_code, top_start, dy, rows, hi_rows=None, hi_color=None, code_w=92.0):
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
    """Left vertical stage accent bar + stage eyebrow title (white page ground)."""
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
        Rectangle((x, T(bot_y)), 1.3, bot_y - top_y, fc=color, ec="none", zorder=5)
    )
    cv.text(
        x + 2.8,
        T(top_y + 2.1),
        title,
        size=SZ_BODY,
        weight="bold",
        color=color,
        ha="left",
        z=10,
    )


def draw_diag_stage1(cv, x0, top_y, w, h):
    """Band 1: Two gripper depth maps + flip-only orientation + channel check."""
    cy = T(top_y + h / 2.0)
    cv.box(x0, T(top_y + h), w, h, fill=C_SLATE_FILL, stroke=C_SLATE_SOFT, lw=LW_BASE, radius=1.2, z=2)

    # Mini depth map tiles (thumb & index)
    for k, (lbl, col, off_x) in enumerate([("D_th", C_ACCENT, 2.0), ("D_idx", C_HERO, 13.4)]):
        bx = x0 + off_x
        by = cy - 3.6
        cv.ax.add_patch(Rectangle((bx, by), 9.4, 7.2, fc="#1E293B", ec=col, lw=LW_BASE, zorder=6))
        cv.ax.add_patch(Circle((bx + 4.7 + (0.6 if k else -0.6), by + 3.6), 1.8, fc=col, alpha=0.85, ec="none", zorder=7))
        cv.text(bx + 4.7, by + 3.6, lbl, size=SZ_CODE, color="white", weight="bold", family="monospace", z=9)

    # Arrow -> Transform & verify box
    cv.arrow([(x0 + 23.8, cy), (x0 + 27.6, cy)], color=C_SLATE, lw=LW_BASE, head=0.9)
    cv.box(x0 + 28.2, cy - 4.0, 22.4, 8.0, fill="white", stroke=C_EVAL, lw=LW_BASE, radius=1.0, z=6)
    cv.text(x0 + 39.4, cy + 1.6, "flip_x/y/180 (no scale)", size=SZ_CODE, color=C_INK, z=10)
    cv.text(x0 + 39.4, cy - 1.9, "ref_ch check (L->R)", size=SZ_CODE, color=C_EVAL, z=10)


def draw_diag_stage2(cv, x0, top_y, w, h):
    """Band 2: 3D MANO face barycentric -> 2D UV triangle -> quantized texel."""
    cy = T(top_y + h / 2.0)
    cv.box(x0, T(top_y + h), w, h, fill=C_EVAL_FILL, stroke=C_EVAL_SOFT, lw=LW_BASE, radius=1.2, z=2)

    # Left: 3D MANO mesh triangle with barycentric weights
    tri3d = [(x0 + 3.2, cy - 4.8), (x0 + 15.2, cy - 4.2), (x0 + 8.6, cy + 4.8)]
    cv.ax.add_patch(Polygon(tri3d, closed=True, fc="white", ec=C_SLATE, lw=LW_BASE, zorder=6))
    p3 = (x0 + 9.0, cy - 0.8)
    for v in tri3d:
        cv.line([p3, v], color=C_EVAL, lw=LW_BASE, dashed=True)
    cv.ax.add_patch(Circle(p3, 0.8, fc=C_ACCENT, ec="white", lw=0.35, zorder=9))
    cv.text(x0 + 8.8, cy - 6.6, "3D face (b0,b1,b2)", size=SZ_CODE, color=C_INK, z=10)

    cv.arrow([(x0 + 16.2, cy), (x0 + 20.6, cy)], color=C_EVAL, lw=LW_BASE, head=0.9)
    cv.text(x0 + 18.4, cy + 2.6, "b", size=SZ_CODE, color=C_EVAL, weight="bold", z=10)

    # Middle: 2D UV triangle
    tri2d = [(x0 + 21.8, cy - 4.6), (x0 + 33.4, cy - 4.6), (x0 + 27.6, cy + 4.6)]
    cv.ax.add_patch(Polygon(tri2d, closed=True, fc=C_HERO_FILL, ec=C_HERO, lw=LW_BASE, zorder=6))
    p2 = (x0 + 27.6, cy - 0.8)
    for v in tri2d:
        cv.line([p2, v], color=C_HERO, lw=LW_BASE, dashed=True)
    cv.ax.add_patch(Circle(p2, 0.8, fc=C_ACCENT, ec="white", lw=0.35, zorder=9))
    cv.text(x0 + 27.6, cy - 6.6, "UV = Σ b_k T_uv", size=SZ_CODE, color=C_HERO, z=10)

    cv.arrow([(x0 + 34.4, cy), (x0 + 38.4, cy)], color=C_HERO, lw=LW_BASE, head=0.9)

    # Right: Quantized S x S texel cell
    gx, gy = x0 + 39.4, cy - 4.4
    cell = 2.8
    for r in range(4):
        for c in range(4):
            is_hit = (r == 2 and c == 1)
            cv.ax.add_patch(
                Rectangle(
                    (gx + c * cell, gy + r * cell),
                    cell,
                    cell,
                    fc=C_ACCENT if is_hit else "white",
                    ec=C_SLATE_SOFT,
                    lw=0.4,
                    zorder=6,
                )
            )
    cv.text(gx + 5.6, cy - 6.6, "texel (x, y)", size=SZ_CODE, color=C_INK, z=10)


def draw_diag_stage3(cv, x0, top_y, w, h):
    """Band 3: Per-finger bincount mean -> equal-weight merge (NaN preserved)."""
    cy = T(top_y + h / 2.0)
    cv.box(x0, T(top_y + h), w, h, fill=C_EVAL_FILL, stroke=C_EVAL, lw=LW_HERO, radius=1.2, z=2)

    # Left: two mini finger grids (R_th, R_idx) stacked
    for idx, (lbl, col, y_off) in enumerate([("R_th", C_ACCENT, 4.4), ("R_idx", C_HERO, -4.4)]):
        gx, gy = x0 + 2.4, cy + y_off - 3.0
        cell = 1.9
        for r in range(3):
            for c in range(3):
                active = (r == 1 and c == 1) or (idx == 0 and r == 2 and c == 1) or (idx == 1 and r == 1 and c == 2)
                fc = col if active else "#E2E8F0"
                cv.ax.add_patch(Rectangle((gx + c * cell, gy + r * cell), cell, cell, fc=fc, ec="white", lw=0.35, zorder=6))
        cv.text(gx + 7.2, gy + 2.8, lbl, size=SZ_CODE, color=col, weight="bold", ha="left", family="monospace", z=10)

    # Merge arrows into equal-weight combiner
    cv.arrow([(x0 + 16.5, cy + 4.0), (x0 + 22.0, cy + 1.2)], color=C_SLATE, lw=LW_BASE, head=0.85)
    cv.arrow([(x0 + 16.5, cy - 4.0), (x0 + 22.0, cy - 1.2)], color=C_SLATE, lw=LW_BASE, head=0.85)
    cv.box(x0 + 22.4, cy - 3.6, 10.6, 7.2, fill="white", stroke=C_EVAL, lw=LW_BASE, radius=1.0, z=6)
    cv.text(x0 + 27.7, cy + 1.3, "sum / n", size=SZ_CODE, weight="bold", color=C_EVAL, z=10)
    cv.text(x0 + 27.7, cy - 1.8, "n_obs > 0", size=SZ_CODE, color=C_MUTED, z=10)
    cv.arrow([(x0 + 33.4, cy), (x0 + 36.8, cy)], color=C_EVAL, lw=LW_BASE, head=0.9)

    # Right: Merged 4x4 raster distinguishing Contact, Obs-0, and NaN (hatched)
    gx, gy = x0 + 37.6, cy - 2.6
    cell = 2.95
    pattern = [
        ["nan", "nan", "obs0", "nan"],
        ["nan", "hot", "hot", "obs0"],
        ["obs0", "hot", "obs0", "nan"],
        ["nan", "nan", "nan", "nan"],
    ]
    for r in range(4):
        for c in range(4):
            kind = pattern[3 - r][c]
            fc = C_ACCENT if kind == "hot" else ("#334155" if kind == "obs0" else "#F1F5F9")
            ec = "#94A3B8" if kind == "nan" else "white"
            hatch = "////" if kind == "nan" else None
            cv.ax.add_patch(
                Rectangle(
                    (gx + c * cell, gy + r * cell),
                    cell,
                    cell,
                    fc=fc,
                    ec=ec,
                    lw=0.35,
                    hatch=hatch,
                    zorder=6,
                )
            )
    # Compact legend under merged grid
    cv.text(x0 + 26.0, T(top_y + h - 2.1), "orange: contact   dark: obs-0   hatched: NaN (unknown)", size=SZ_CODE, color=C_INK, z=10)


def draw_diag_stage4(cv, x0, top_y, w, h):
    """Band 4: Fixed-range encoding -> uint8 Contact Proxy + Binary Valid Mask."""
    cy = T(top_y + h / 2.0)
    cv.box(x0, T(top_y + h), w, h, fill=C_HERO_FILL, stroke=C_HERO, lw=LW_HERO, radius=1.2, z=2)

    # Left: Fixed global ramp [0.001, 0.4] -> [0, 255]
    bx, by = x0 + 2.4, cy - 4.2
    grad = np.linspace(0, 1, 32).reshape(-1, 1)
    cv.ax.imshow(grad, cmap="inferno", extent=[bx, bx + 2.6, by, by + 9.2], origin="lower", zorder=7)
    cv.ax.add_patch(Rectangle((bx, by), 2.6, 9.2, fc="none", ec=C_SLATE, lw=0.45, zorder=8))
    cv.text(bx + 3.4, by + 8.4, "0.400", size=SZ_CODE, color=C_INK, ha="left", family="monospace", z=10)
    cv.text(bx + 3.4, by + 0.8, "0.001", size=SZ_CODE, color=C_INK, ha="left", family="monospace", z=10)
    cv.text(bx + 1.3, by - 2.2, "fixed", size=SZ_CODE, color=C_HERO, weight="bold", z=10)

    cv.arrow([(x0 + 13.8, cy + 0.6), (x0 + 17.6, cy + 0.6)], color=C_HERO, lw=LW_BASE, head=0.9)

    # Contact proxy PNG card
    cx1, cy1 = x0 + 18.4, cy - 3.6
    cv.ax.add_patch(Rectangle((cx1, cy1), 14.0, 9.2, fc="#18181C", ec=C_HERO, lw=LW_BASE, zorder=6))
    cv.ax.add_patch(Circle((cx1 + 4.4, cy1 + 5.6), 2.0, fc=C_ACCENT, ec="none", zorder=7))
    cv.ax.add_patch(Circle((cx1 + 9.6, cy1 + 3.6), 1.6, fc="#FDBA74", ec="none", zorder=7))
    cv.text(cx1 + 7.0, cy1 - 2.2, "contact.png (u8)", size=SZ_CODE, color=C_HERO, weight="bold", z=10)

    # Binary valid mask card
    cx2, cy2 = x0 + 35.4, cy - 3.6
    cv.ax.add_patch(Rectangle((cx2, cy2), 14.0, 9.2, fc="#18181C", ec=C_EVAL, lw=LW_BASE, zorder=6))
    cv.ax.add_patch(Circle((cx2 + 4.4, cy2 + 5.6), 2.4, fc="white", ec="none", zorder=7))
    cv.ax.add_patch(Circle((cx2 + 9.6, cy2 + 3.6), 2.0, fc="white", ec="none", zorder=7))
    cv.text(cx2 + 7.0, cy2 - 2.2, "valid.png (0/255)", size=SZ_CODE, color=C_EVAL, weight="bold", z=10)


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
            ("Algorithm 2  ", C_EVAL, "bold"),
            ("Single-Frame Parallel-Gripper Tactile Retargeting  ", C_INK, "bold"),
            ("(gripper_retarget.py)", C_MUTED, "normal"),
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
            ("FixedMapping M (778-vert MANO, face_id, barycentric b); float32 depths {D_th, D_idx}; side; flips tau; size S=128; mode", C_INK),
        ],
        size=SZ_CODE,
        ha="left",
    )
    cv.rich(
        2.4,
        T(12.8),
        [
            ("Output: ", C_INK, "bold"),
            ("Canonical S x S float64 raster R (NaN = unknown); encoded contact/depth PNGs; binary observation mask M_valid", C_INK),
        ],
        size=SZ_CODE,
        ha="left",
    )
    cv.rule(2.0, W - 2.0, T(15.0), lw=LW_BASE, color=C_SLATE_SOFT)

    # Column headers
    cv.text(50.0, T(17.1), "Pseudocode  (retarget_frame)", size=SZ_BODY, weight="bold", color=C_SLATE)
    cv.text(128.2, T(17.1), "Aligned Geometric & Tensor Flow", size=SZ_BODY, weight="bold", color=C_EVAL)

    # -----------------------------------------------------------------------
    # Stage Bands (exact vertical alignment between Left Code and Right Diagrams)
    # -----------------------------------------------------------------------
    x_left = 2.0
    x_num = 7.4
    x_code = 8.6
    x_diag = 102.0
    w_diag = 52.0
    dy = 2.95

    # Stage 1: Channel Check & Flip-Only Orientation (top 19.2 .. 31.8, h = 12.6)
    s1_t, s1_b = 19.2, 31.8
    draw_stage_rail(cv, x_left, s1_t, s1_b, C_SLATE, "Stage 1: Channel Check & Flip-Only Orientation", fill=None)
    rows1 = [
        (1, [("for ", C_HERO, "bold"), ("f in {thumb, index} ", C_INK), ("do", C_HERO, "bold")]),
        (2, [("  assert ", C_HERO, "bold"), ("channel[f] == M[f].ref_channel", C_INK), ("   # R reuses L map", C_MUTED)]),
        (3, [("  D_f <- ", C_INK), ("transform_depth", C_EVAL, "bold"), ("(D_f, tau[f])", C_INK), ("   # flip only, no scale", C_MUTED)]),
    ]
    draw_code_block(cv, x_num, x_code, s1_t + 3.4, dy, rows1, code_w=89.5)
    draw_diag_stage1(cv, x_diag, s1_t, w_diag, s1_b - s1_t)

    # Stage 2: Static Barycentric -> Official UV Texel Lookup (top 33.4 .. 51.8, h = 18.4)
    s2_t, s2_b = 33.4, 51.8
    cv.rule(x_left, 98.5, T(32.6), lw=0.4, color=C_SLATE_SOFT)
    draw_stage_rail(cv, x_left, s2_t, s2_b, C_EVAL, "Stage 2: Static Barycentric to Official UV Texel Precomputation", fill=None)
    rows2 = [
        (4, [("T_uv, F_uv <- ", C_INK), ("official_uv", C_HERO, "bold"), ("(side, M.faces)", C_INK), ("   # MANO_UV_{side}.obj", C_MUTED)]),
        (5, [("for ", C_HERO, "bold"), ("f in {thumb, index} ", C_INK), ("do", C_HERO, "bold")]),
        (6, [("  I_v <- ", C_INK), ("flatnonzero", C_SLATE), ("(M[f].mapping_valid)", C_INK)]),
        (7, [("  uv <- ", C_INK), ("einsum", C_EVAL, "bold"), ("(b[I_v], T_uv[F_uv[face_id[I_v]]])", C_INK), ("  # in [0, 1]^2", C_MUTED)]),
        (8, [("  texel[f] <- ", C_INK), ("rint((1 - v)*(S-1))*S + rint(u*(S-1))", C_HERO)]),
    ]
    draw_code_block(cv, x_num, x_code, s2_t + 3.4, dy, rows2, hi_rows={7, 8}, hi_color=C_EVAL_SOFT, code_w=89.5)
    draw_diag_stage2(cv, x_diag, s2_t, w_diag, s2_b - s2_t)

    # Stage 3: Per-Finger Bincount Mean & Equal-Weight Merge (top 53.4 .. 74.6, h = 21.2)
    s3_t, s3_b = 53.4, 74.6
    cv.rule(x_left, 98.5, T(52.6), lw=0.4, color=C_SLATE_SOFT)
    draw_stage_rail(cv, x_left, s3_t, s3_b, C_EVAL, "Stage 3: Per-Finger Texel Averaging & NaN-Preserving Merge", fill=None)
    rows3 = [
        (9, [("total <- 0_{S x S};  count <- 0_{S x S}", C_INK)]),
        (10, [("for ", C_HERO, "bold"), ("f in {thumb, index} ", C_INK), ("do", C_HERO, "bold")]),
        (11, [("  v <- D_f.flat[M[f].src_px];  v[~valid | ~isfinite(v)] <- NaN", C_INK)]),
        (12, [("  R_f[t] <- ", C_INK), ("bincount_sum", C_EVAL, "bold"), ("(texel[f], v) / bincount_n  (else NaN)", C_INK)]),
        (13, [("  total[fin] += R_f[fin];  count[fin] += 1", C_INK), ("  # equal weight", C_MUTED)]),
        (14, [("R <- NaN_{S x S};  R[count > 0] <- total / count", C_ACCENT, "bold"), ("  # unk != 0", C_MUTED)]),
    ]
    draw_code_block(cv, x_num, x_code, s3_t + 3.4, dy, rows3, hi_rows={14}, hi_color=C_ACCENT_SOFT, code_w=89.5)
    draw_diag_stage3(cv, x_diag, s3_t, w_diag, s3_b - s3_t)

    # Stage 4: Fixed-Range Encoding & Explicit Validity Mask (top 76.2 .. 93.4, h = 17.2)
    s4_t, s4_b = 76.2, 93.4
    cv.rule(x_left, 98.5, T(75.4), lw=0.4, color=C_SLATE_SOFT)
    draw_stage_rail(cv, x_left, s4_t, s4_b, C_HERO, "Stage 4: Fixed-Range Proxy Encoding & Explicit Validity Mask", fill=None)
    rows4 = [
        (15, [("valid <- ", C_INK), ("isfinite", C_EVAL, "bold"), ("(R);  ", C_INK), ("M_valid <- encode_png(valid * 255)", C_EVAL, "bold")]),
        (16, [("if ", C_HERO, "bold"), ("mode in {contact, both} ", C_INK), ("then", C_HERO, "bold")]),
        (17, [("  C[valid] <- rint(255 * clip((R - 0.001) / 0.399, 0, 1))", C_HERO, "bold")]),
        (18, [("if ", C_HERO, "bold"), ("mode in {depth, both} ", C_INK), ("then", C_HERO, "bold")]),
        (19, [("  Q[valid] <- rint((R + 0.001) / 0.401 * 65534) + 1", C_SLATE), ("  # 0=unk", C_MUTED)]),
    ]
    draw_code_block(cv, x_num, x_code, s4_t + 3.2, dy, rows4, hi_rows={15, 17}, hi_color=C_HERO_SOFT, code_w=89.5)
    draw_diag_stage4(cv, x_diag, s4_t, w_diag, s4_b - s4_t)

    # -----------------------------------------------------------------------
    # Stage Connectors (horizontal ticks + right-column vertical flow)
    # -----------------------------------------------------------------------
    for (st, sb, col) in [(s1_t, s1_b, C_SLATE), (s2_t, s2_b, C_EVAL), (s3_t, s3_b, C_EVAL), (s4_t, s4_b, C_HERO)]:
        ym = T((st + sb) / 2.0)
        cv.arrow([(98.8, ym), (101.8, ym)], color=col, lw=LW_BASE, head=0.8)

    # Vertical arrows connecting the 4 diagram bands
    for (b_prev, t_next, col) in [(s1_b, s2_t, C_SLATE), (s2_b, s3_t, C_EVAL), (s3_b, s4_t, C_HERO)]:
        cv.arrow([(128.0, T(b_prev)), (128.0, T(t_next))], color=col, lw=LW_BASE, head=0.8)

    # Bottom booktabs rule
    cv.rule(2.0, W - 2.0, T(94.6), lw=LW_HERO, color=C_INK)

    # -----------------------------------------------------------------------
    # Audits & export (SVG + PNG only, no PDF per AGENTS.md)
    # -----------------------------------------------------------------------
    overlaps = cv.overlaps()
    text_ov = cv.text_overlaps()
    crossings = cv.wire_crossings()
    print(f"Gripper Alg Audits -> overlaps: {len(overlaps)}, text_overlaps: {len(text_ov)}, wire_crossings: {len(crossings)}")
    if overlaps:
        print("  overlaps:", overlaps)
    if text_ov:
        print("  text_overlaps:", text_ov)
    if crossings:
        print("  wire_crossings:", crossings)

    assert not overlaps and not text_ov and not crossings

    os.makedirs("img", exist_ok=True)
    os.makedirs("assets/alg_gripper", exist_ok=True)
    out_stem = "img/alg_gripper_retarget"
    cv.save(out_stem, formats=("svg", "png"))
    print(f"Saved: {out_stem}.svg, {out_stem}.png")


if __name__ == "__main__":
    draw()
