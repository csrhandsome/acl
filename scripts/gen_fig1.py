"""Generate Figure 1: ForceBench Overview (vector PDF + editable SVG + 600-dpi PNG).

Usage:
    uv run --with matplotlib --with numpy --with pillow python scripts/gen_fig1.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

SKILL_SCRIPTS = os.path.expanduser("~/.agents/skills/cs-paper-figure/scripts")
sys.path.insert(0, SKILL_SCRIPTS)
from paperkit import Canvas  # noqa: E402

# ---------------------------------------------------------------------------
# Visual system constants (Plate register, ACL full-width 156 mm)
# ---------------------------------------------------------------------------
W, H = 156.0, 68.0
T = lambda top_mm: H - top_mm  # top-down mm helper  # noqa: E731

# Budget: 2 chromatic hues (Blue = Unification/Hero, Teal = Evaluation) + Neutral Slate
C_SLATE = "#475569"
C_SLATE_FILL = "#EDF1F5"   # Soft cool grey ground so 3D robot renders have crisp contrast
C_SLATE_SOFT = "#E2E8F0"
C_INK = "#0F172A"
C_MUTED = "#64748B"

C_HERO = "#1D4ED8"        # Blue: Canonical MANO UV Atlas & Data Contract
C_HERO_FILL = "#EFF6FF"
C_HERO_SOFT = "#BFDBFE"

C_EVAL = "#0F766E"        # Teal: L0-L2 Evaluation Hierarchy
C_EVAL_FILL = "#F0FDFA"
C_EVAL_SOFT = "#CCFBF1"

C_ACCENT = "#D97706"      # Subtle warm highlight for fingertip contact pads

# Budget: 2 stroke weights, 3 text sizes (all >= 5.8 pt)
LW_BASE = 0.6
LW_HERO = 0.85
SZ_HEAD = 7.8
SZ_BODY = 6.5
SZ_SUB = 5.8


def make_mano_mask_asset(out_dir):
    """Extract the official bimanual MANO UV mask from _mano_uv_atlas.png."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    atlas_path = os.path.join(out_dir, "_mano_uv_atlas.png")
    mask_path = os.path.join(out_dir, "_mano_mask_sample.png")

    arr = np.asarray(Image.open(atlas_path).convert("RGB"))
    left = arr[:, : arr.shape[1] // 2]
    bg = left[0, 0].astype(np.float32)
    diff = np.linalg.norm(left.astype(np.float32) - bg, axis=-1)
    hand_mask = diff > 40.0
    bimanual_mask = np.concatenate([hand_mask, hand_mask], axis=1)

    mask_rgb = np.zeros((bimanual_mask.shape[0], bimanual_mask.shape[1], 3), dtype=np.float32)
    mask_rgb[bimanual_mask] = [0.95, 0.97, 1.0]
    mask_rgb[~bimanual_mask] = [0.09, 0.12, 0.18]
    plt.imsave(mask_path, mask_rgb)

    return mask_path


def draw_smooth_heatmap(cv, x0, y0, w_mm, h_mm, hx=0.5, hy=0.5, alpha=0.90, z=10, border=True):
    """Draw a smooth continuous tactile contact heatmap (green->yellow->red, no gridlines)."""
    res = 64
    yy, xx = np.mgrid[0:res, 0:res]
    nx = (xx + 0.5) / res
    ny = (yy + 0.5) / res
    d2 = ((nx - hx) / 0.34) ** 2 + ((ny - hy) / 0.34) ** 2
    val = np.clip(0.12 + 0.88 * np.exp(-d2 * 1.45), 0.0, 1.0)
    cmap = plt.get_cmap("RdYlGn_r")
    rgba = cmap(val)
    rgba[..., 3] = alpha
    cv.ax.imshow(
        rgba,
        extent=[x0, x0 + w_mm, y0, y0 + h_mm],
        origin="lower",
        interpolation="bilinear",
        zorder=z,
    )
    if border:
        cv.ax.add_patch(
            Rectangle((x0, y0), w_mm, h_mm, fc="none", ec=C_SLATE, lw=0.35, zorder=z + 1)
        )


def draw_hand_with_tactile(cv, img_path, cx, cy, w_mm, h_mm, tip_pads):
    """Embed a cutout hand image with smooth tactile contact heatmaps on fingertips."""
    cv.image(img_path, cx=cx, cy=cy, w=w_mm, z=8, knockout_white=False, crop=False)
    x0 = cx - w_mm / 2.0
    y0 = cy - h_mm / 2.0

    for gx, gy, gw, gh, hx, hy in tip_pads:
        draw_smooth_heatmap(
            cv,
            x0 + gx * w_mm,
            y0 + gy * h_mm,
            gw * w_mm,
            gh * h_mm,
            hx=hx,
            hy=hy,
            alpha=0.82,
            z=9,
            border=False,
        )


def place_rgba_image(cv, path, cx, cy, w, z=8):
    """Place a transparent RGBA PNG directly onto the canvas without white-background compositing."""
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    h = w * (im.height / im.width)
    arr = np.asarray(im)
    cv.ax.imshow(
        arr,
        extent=[cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0],
        origin="upper",
        interpolation="lanczos",
        zorder=z,
    )
    return h


def draw_dexterous_hands(cv, cx, cy):
    """Place URDF-rendered Wuji Hand and Sharpa Wave with 3D surface-mapped fingertip halos."""
    # _urdf_wuji.png: 726 x 969, _urdf_sharpa.png: 664 x 969
    place_rgba_image(cv, "assets/fig1/_urdf_wuji.png", cx=cx - 6.8, cy=cy, w=9.8, z=8)
    place_rgba_image(cv, "assets/fig1/_urdf_sharpa.png", cx=cx + 6.8, cy=cy, w=9.0, z=8)


def draw_encoder_schematics(cv, x0, y_cnn, y_vit, y_jepa):
    """Draw compact vector micro-schematics for CNN, ViT, and JEPA inside Frozen Encoders."""
    lw_s = 0.45

    # Row 1: CNN (hierarchical shrinking feature maps)
    cv.text(x0, y_cnn, "CNN", size=SZ_SUB, weight="bold", color=C_INK, ha="left")
    cnn_x = x0 + 6.8
    for idx, (dx, bw, bh) in enumerate([(0.0, 1.7, 3.6), (2.9, 1.7, 2.5), (5.8, 1.7, 1.5)]):
        cv.ax.add_patch(
            FancyBboxPatch(
                (cnn_x + dx, y_cnn - bh / 2.0), bw, bh,
                boxstyle="round,pad=0.0,rounding_size=0.25",
                fc="white", ec=C_SLATE, lw=lw_s, zorder=6,
            )
        )
        if idx < 2:
            x_a = cnn_x + dx + bw
            cv.line([(x_a, y_cnn), (x_a + 1.2, y_cnn)], color=C_SLATE, lw=lw_s)
    cv.line([(cnn_x + 7.5, y_cnn), (cnn_x + 8.7, y_cnn)], color=C_SLATE, lw=lw_s)

    # Row 2: ViT (2x2 patch tokens -> Transformer block)
    cv.text(x0, y_vit, "ViT", size=SZ_SUB, weight="bold", color=C_INK, ha="left")
    vit_x = x0 + 6.8
    for r in range(2):
        for c in range(2):
            px = vit_x + c * 1.15
            py = y_vit - 1.05 + r * 1.15
            cv.ax.add_patch(
                Rectangle((px, py), 0.9, 0.9, fc="white", ec=C_SLATE, lw=lw_s, zorder=6)
            )
    cv.line([(vit_x + 2.15, y_vit), (vit_x + 3.4, y_vit)], color=C_SLATE, lw=lw_s)
    cv.ax.add_patch(
        FancyBboxPatch(
            (vit_x + 3.4, y_vit - 1.6), 4.0, 3.2,
            boxstyle="round,pad=0.0,rounding_size=0.35",
            fc="white", ec=C_SLATE, lw=lw_s, zorder=6,
        )
    )
    # Two internal layer bars inside Transformer block
    cv.line([(vit_x + 4.1, y_vit + 0.55), (vit_x + 6.7, y_vit + 0.55)], color=C_MUTED, lw=0.4)
    cv.line([(vit_x + 4.1, y_vit - 0.55), (vit_x + 6.7, y_vit - 0.55)], color=C_MUTED, lw=0.4)
    cv.line([(vit_x + 7.4, y_vit), (vit_x + 8.7, y_vit)], color=C_SLATE, lw=lw_s)

    # Row 3: JEPA (online encoder + predictor vs. EMA target encoder)
    cv.text(x0, y_jepa, "JEPA", size=SZ_SUB, weight="bold", color=C_INK, ha="left")
    j_x = x0 + 6.8
    y_top = y_jepa + 1.05
    y_bot = y_jepa - 1.05
    # Online context encoder -> Predictor
    cv.ax.add_patch(
        FancyBboxPatch(
            (j_x, y_top - 0.65), 2.3, 1.3,
            boxstyle="round,pad=0.0,rounding_size=0.25",
            fc="white", ec=C_SLATE, lw=lw_s, zorder=6,
        )
    )
    cv.line([(j_x + 2.3, y_top), (j_x + 3.4, y_top)], color=C_SLATE, lw=lw_s)
    cv.ax.add_patch(
        FancyBboxPatch(
            (j_x + 3.4, y_top - 0.65), 2.2, 1.3,
            boxstyle="round,pad=0.0,rounding_size=0.25",
            fc=C_HERO_SOFT, ec=C_HERO, lw=lw_s, zorder=6,
        )
    )
    # Target EMA encoder (dashed)
    cv.ax.add_patch(
        FancyBboxPatch(
            (j_x, y_bot - 0.65), 2.3, 1.3,
            boxstyle="round,pad=0.0,rounding_size=0.25",
            fc="white", ec=C_MUTED, lw=lw_s, ls="--", zorder=6,
        )
    )
    # Latent comparison node
    lat_x = j_x + 7.2
    cv.line([(j_x + 5.6, y_top), (lat_x, y_jepa + 0.35)], color=C_HERO, lw=lw_s)
    cv.line([(j_x + 2.3, y_bot), (lat_x, y_bot), (lat_x, y_jepa - 0.35)], color=C_MUTED, lw=lw_s, dashed=True)
    cv.ax.add_patch(Circle((lat_x, y_jepa), 0.55, fc=C_HERO_FILL, ec=C_HERO, lw=lw_s, zorder=7))
    cv.line([(lat_x + 0.55, y_jepa), (j_x + 8.7, y_jepa)], color=C_SLATE, lw=lw_s)


def draw_probe_schematics(cv, cx_list, y_icon, y_label):
    """Draw 1-row x 3-col micro-schematics for Projected, k-NN, and Attentive probes."""
    lw_s = 0.45
    c_proj, c_knn, c_attn = cx_list

    # Col 1: Projected (input vector -> linear layer -> output)
    for dy in (-1.1, 0.0, 1.1):
        cv.ax.add_patch(Circle((c_proj - 1.5, y_icon + dy), 0.28, fc="white", ec=C_SLATE, lw=lw_s, zorder=6))
        cv.line([(c_proj - 1.22, y_icon + dy), (c_proj - 0.35, y_icon)], color=C_MUTED, lw=0.35)
    cv.ax.add_patch(
        FancyBboxPatch(
            (c_proj - 0.35, y_icon - 1.1), 1.0, 2.2,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            fc=C_SLATE_FILL, ec=C_SLATE, lw=lw_s, zorder=6,
        )
    )
    cv.line([(c_proj + 0.65, y_icon), (c_proj + 1.35, y_icon)], color=C_SLATE, lw=lw_s)
    cv.ax.add_patch(Circle((c_proj + 1.6, y_icon), 0.35, fc=C_HERO_SOFT, ec=C_HERO, lw=lw_s, zorder=7))
    cv.text(c_proj, y_label, "Proj.", size=SZ_SUB, color=C_INK)

    # Col 2: k-NN (query point + dashed neighborhood circle + k nearest neighbors)
    cv.ax.add_patch(
        Circle((c_knn, y_icon), 1.45, fc=C_HERO_FILL, ec=C_HERO, lw=0.4, ls="--", zorder=5)
    )
    cv.ax.add_patch(Circle((c_knn, y_icon), 0.35, fc=C_HERO, ec="none", zorder=7))
    for dx, dy in [(-0.75, 0.65), (0.8, 0.55), (-0.45, -0.85)]:
        cv.ax.add_patch(Circle((c_knn + dx, y_icon + dy), 0.26, fc="white", ec=C_SLATE, lw=lw_s, zorder=6))
    for dx, dy in [(-1.9, -1.2), (1.85, -1.0)]:
        cv.ax.add_patch(Circle((c_knn + dx, y_icon + dy), 0.22, fc=C_SLATE_SOFT, ec=C_MUTED, lw=0.35, zorder=6))
    cv.text(c_knn, y_label, "k-NN", size=SZ_SUB, color=C_INK)

    # Col 3: Attentive (query node attending to multiple token keys)
    q_x, q_y = c_attn, y_icon + 1.25
    cv.ax.add_patch(Circle((q_x, q_y), 0.38, fc=C_HERO_SOFT, ec=C_HERO, lw=lw_s, zorder=7))
    for dx in (-1.45, -0.48, 0.48, 1.45):
        kx, ky = c_attn + dx, y_icon - 1.2
        cv.line([(kx, ky + 0.3), (q_x, q_y - 0.38)], color=C_HERO if abs(dx) < 1.0 else C_MUTED, lw=0.4)
        cv.ax.add_patch(
            Rectangle((kx - 0.28, ky - 0.35), 0.56, 0.65, fc="white", ec=C_SLATE, lw=lw_s, zorder=6)
        )
    cv.text(c_attn, y_label, "Attn.", size=SZ_SUB, color=C_INK)


def draw_hierarchy_schematics(cv, x_center, y_l0, y_l1, y_l2):
    """Draw compact vector micro-schematics on the right side of L0, L1, and L2."""
    lw_s = 0.45

    # L0: Latent z -> Decoder trapezoid -> Reconstructed contact map
    x0 = x_center - 4.0
    cv.ax.add_patch(
        FancyBboxPatch(
            (x0, y_l0 - 1.6), 1.0, 3.2,
            boxstyle="round,pad=0.0,rounding_size=0.2",
            fc=C_EVAL_SOFT, ec=C_EVAL, lw=lw_s, zorder=6,
        )
    )
    cv.line([(x0 + 1.0, y_l0), (x0 + 1.8, y_l0)], color=C_EVAL, lw=lw_s)
    dec_poly = Polygon(
        [(x0 + 1.8, y_l0 - 1.1), (x0 + 3.6, y_l0 - 1.9), (x0 + 3.6, y_l0 + 1.9), (x0 + 1.8, y_l0 + 1.1)],
        fc="white", ec=C_EVAL, lw=lw_s, zorder=6,
    )
    cv.ax.add_patch(dec_poly)
    cv.line([(x0 + 3.6, y_l0), (x0 + 4.4, y_l0)], color=C_EVAL, lw=lw_s)
    # Mini reconstructed UV map square with contact hotspot
    cv.ax.add_patch(Rectangle((x0 + 4.4, y_l0 - 2.0), 3.8, 4.0, fc="#1E293B", ec=C_EVAL, lw=lw_s, zorder=6))
    cv.ax.add_patch(Circle((x0 + 5.7, y_l0 + 0.5), 0.85, fc=C_ACCENT, alpha=0.85, ec="none", zorder=7))
    cv.ax.add_patch(Circle((x0 + 6.9, y_l0 - 0.5), 0.65, fc="#38BDF8", alpha=0.85, ec="none", zorder=7))

    # L1: Force / Temporal Dynamics sparkline + force arrow
    ax_x0, ax_y0 = x_center - 3.8, y_l1 - 2.4
    ax_w, ax_h = 7.6, 4.8
    cv.ax.add_patch(Rectangle((ax_x0, ax_y0), ax_w, ax_h, fc="white", ec=C_EVAL_SOFT, lw=0.4, zorder=5))
    # Axes
    cv.line([(ax_x0 + 0.6, ax_y0 + 0.6), (ax_x0 + 0.6, ax_y0 + ax_h - 0.5)], color=C_SLATE, lw=0.4)
    cv.line([(ax_x0 + 0.6, ax_y0 + 0.6), (ax_x0 + ax_w - 0.5, ax_y0 + 0.6)], color=C_SLATE, lw=0.4)
    # Force / slip curve F(t)
    xs = np.linspace(ax_x0 + 0.9, ax_x0 + ax_w - 0.7, 14)
    ys = ax_y0 + 1.0 + 2.4 * np.exp(-((xs - (ax_x0 + 3.6)) / 1.8) ** 2) + 0.35 * np.sin((xs - ax_x0) * 2.2)
    cv.ax.plot(xs, ys, color=C_EVAL, lw=0.75, zorder=7)
    # Peak marker + vertical delta indicator
    px_i = int(np.argmax(ys))
    cv.ax.add_patch(Circle((xs[px_i], ys[px_i]), 0.38, fc=C_ACCENT, ec="none", zorder=8))

    # L2: Cross-Source (Hand <-> Gripper) + Perturbation halo
    gx = x_center - 2.4
    hx = x_center + 2.4
    # Left node: 2-jaw parallel gripper icon
    cv.ax.add_patch(Circle((gx, y_l2), 1.65, fc="white", ec=C_EVAL, lw=lw_s, zorder=6))
    cv.line([(gx - 0.6, y_l2 - 0.8), (gx - 0.6, y_l2 + 0.7)], color=C_SLATE, lw=0.55)
    cv.line([(gx + 0.6, y_l2 - 0.8), (gx + 0.6, y_l2 + 0.7)], color=C_SLATE, lw=0.55)
    cv.line([(gx - 0.6, y_l2 - 0.8), (gx + 0.6, y_l2 - 0.8)], color=C_SLATE, lw=0.55)
    # Right node: 5-finger hand schematic (palm + 4/5 rays)
    cv.ax.add_patch(Circle((hx, y_l2), 1.65, fc="white", ec=C_EVAL, lw=lw_s, ls="--", zorder=6))
    cv.ax.add_patch(Circle((hx, y_l2 - 0.3), 0.45, fc=C_EVAL_SOFT, ec=C_SLATE, lw=0.4, zorder=7))
    for ang in (-0.85, -0.3, 0.25, 0.8):
        cv.line(
            [(hx + 0.35 * np.sin(ang), y_l2 - 0.1 + 0.35 * np.cos(ang)),
             (hx + 1.05 * np.sin(ang), y_l2 - 0.1 + 1.05 * np.cos(ang))],
            color=C_SLATE, lw=0.45,
        )
    # Bidirectional transfer arrow
    cv.arrow([(gx + 1.75, y_l2), (hx - 1.75, y_l2)], color=C_EVAL, lw=0.55, head=0.75, both=True)


def draw_parallel_gripper(cv, cx, cy):
    """Place URDF/CAD-rendered GelSight Mini + WSG 50 with 3D pad halo and contact thumbnails."""
    # _urdf_gelsight.png: 796 x 918
    gx = cx - 7.6
    wg = 10.6
    hg = place_rgba_image(cv, "assets/fig1/_urdf_gelsight.png", cx=gx, cy=cy, w=wg, z=8)

    x0 = gx - wg / 2.0
    y0 = cy - hg / 2.0

    # Callout zoom triangle from the visible left 3D GelSight pad halo to contact thumbnails
    tx0, ty0 = x0 + 0.44 * wg, y0 + 0.83 * hg
    cv.ax.add_patch(
        Polygon([(tx0, ty0), (cx + 1.2, cy + 4.8), (cx + 1.2, cy - 0.3)],
                fc="#CBD5E1", alpha=0.28, ec=C_SLATE, lw=0.35, zorder=7)
    )

    # Right: Real visuo-tactile contact-surface examples (from high-res uploaded strips)
    cv.image("assets/fig1/_tactile_dot.png", cx=cx + 4.1, cy=cy + 2.2, w=5.3, z=9, knockout_white=True, crop=False)
    cv.image("assets/fig1/_tactile_grating.png", cx=cx + 1.1, cy=cy - 2.9, w=4.9, z=9, knockout_white=True, crop=False)
    cv.image("assets/fig1/_tactile_force.png", cx=cx + 7.0, cy=cy - 2.9, w=4.9, z=9, knockout_white=True, crop=False)


def draw():
    asset_dir = "assets/fig1"
    out_dir = "img"
    mask_path = make_mano_mask_asset(asset_dir)
    uv_path = os.path.join(asset_dir, "_mano_uv_atlas.png")

    cv = Canvas(W, H, family="serif", size=SZ_BODY)

    # -----------------------------------------------------------------------
    # Column 1: Heterogeneous Inputs (x = 2..34)
    # -----------------------------------------------------------------------
    cv.text(18.0, T(4.2), "1. Tactile Inputs", size=SZ_HEAD, weight="bold", color=C_INK)

    # Route A box
    cv.box(
        2.0, T(32.5), 32.0, 24.5,
        fill=C_SLATE_FILL, stroke=C_SLATE, lw=LW_BASE, radius=1.8, z=3,
    )
    cv.text(18.0, T(10.6), "Route A: Dexterous Hands", size=SZ_BODY, weight="bold", color=C_INK)
    cv.text(18.0, T(13.6), "FK + Inverse Retargeting", size=SZ_SUB, color=C_MUTED)
    draw_dexterous_hands(cv, 18.0, T(22.2))
    cv.text(18.0, T(30.4), "5 corpora (22-DoF, 3–5 fingers)", size=SZ_SUB, color=C_SLATE)

    # Route B box
    cv.box(
        2.0, T(60.0), 32.0, 24.5,
        fill=C_SLATE_FILL, stroke=C_SLATE, lw=LW_BASE, radius=1.8, z=3,
    )
    cv.text(18.0, T(38.1), "Route B: Parallel Grippers", size=SZ_BODY, weight="bold", color=C_INK)
    cv.text(18.0, T(41.3), "Static Surface Mapping", size=SZ_SUB, color=C_MUTED)
    draw_parallel_gripper(cv, 18.0, T(50.5))
    cv.text(18.0, T(57.9), "Visuo-tactile pads (Thumb / Index)", size=SZ_SUB, color=C_SLATE)

    # -----------------------------------------------------------------------
    # Column 2 (HERO): Canonical MANO UV Atlas (x = 38.5..75.0)
    # User guidance:
    # - Stack MANO UV Mask -> UV Contact Map vertically with an explicit arrow
    #   to avoid crowding and clearly show the mask-to-UV mapping.
    # - Remove the Data Contract card from Fig 1 (covered in Fig 2 / text).
    # - Visually split the UV image into Left (pink schematic hand topology)
    #   and Right (dark actual contact map fed to encoders) with aligned labels.
    # -----------------------------------------------------------------------
    cv.text(56.75, T(4.2), "2. Unified Representation", size=SZ_HEAD, weight="bold", color=C_HERO)

    cv.box(
        38.5, T(60.0), 36.5, 52.0,
        fill=C_HERO_FILL, stroke=C_HERO, lw=LW_HERO, radius=2.2, z=3,
    )
    cv.text(56.75, T(11.0), "Canonical MANO UV Atlas", size=SZ_BODY, weight="bold", color=C_HERO)
    cv.text(56.75, T(14.2), "Bimanual 128 × 256 · 30 FPS", size=SZ_SUB, color=C_INK)

    # Step 1: MANO UV Validity Mask (top)
    cv.image(mask_path, cx=56.75, cy=T(22.2), w=21.0, z=8, knockout_white=False, crop=False)
    cv.ax.add_patch(
        Rectangle((56.75 - 10.5, T(22.2) - 5.25), 21.0, 10.5, fc="none", ec=C_SLATE, lw=0.45, zorder=9)
    )

    # Explicit Mask -> UV mapping connector
    cv.arrow([(56.75, T(27.8)), (56.75, T(32.8))], color=C_HERO, lw=LW_HERO, head=1.3)
    cv.text(55.0, T(30.3), "MANO UV Mask", size=SZ_SUB, weight="bold", color=C_INK, ha="right")
    cv.text(58.5, T(30.3), "Maps valid surface", size=SZ_SUB, color=C_HERO, style="italic", ha="left")

    # Step 2: Canonical UV Contact Map (split Left = Schematic vs. Right = Actual Input)
    uv_w, uv_h = 31.0, 15.5
    uv_cy = T(44.5)
    uv_x0 = 56.75 - uv_w / 2.0
    uv_y0 = uv_cy - uv_h / 2.0

    cv.text(56.75 - uv_w / 4.0, T(35.0), "Hand Layout", size=SZ_SUB, color=C_MUTED)
    cv.text(56.75 + uv_w / 4.0, T(35.0), "Actual UV Input", size=SZ_SUB, weight="bold", color=C_HERO)

    cv.image(uv_path, cx=56.75, cy=uv_cy, w=uv_w, z=8, knockout_white=False, crop=False)
    # Left half: muted border (schematic hand topology)
    cv.ax.add_patch(
        Rectangle((uv_x0, uv_y0), uv_w / 2.0, uv_h, fc="none", ec=C_SLATE, lw=0.45, zorder=9)
    )
    # Right half: hero blue border (actual dark contact map fed to encoders)
    cv.ax.add_patch(
        Rectangle((56.75, uv_y0), uv_w / 2.0, uv_h, fc="none", ec=C_HERO, lw=0.75, zorder=10)
    )
    cv.ax.plot([56.75, 56.75], [uv_y0, uv_y0 + uv_h], color="white", lw=0.5, ls="--", alpha=0.7, zorder=11)

    cv.text(56.75, T(55.0), "Left: schematic · Right: actual contact map", size=SZ_SUB, color=C_MUTED)

    # Arrows: Route A & B -> Hero
    cv.arrow([(34.0, T(20.2)), (38.5, T(20.2))], color=C_SLATE, lw=LW_BASE, head=1.3)
    cv.arrow([(34.0, T(47.8)), (38.5, T(47.8))], color=C_SLATE, lw=LW_BASE, head=1.3)

    # -----------------------------------------------------------------------
    # Column 3: Frozen Encoders & Probes (x = 79.5..104.5, w = 25.0)
    # -----------------------------------------------------------------------
    cv.text(92.0, T(4.2), "3. Frozen Probing", size=SZ_HEAD, weight="bold", color=C_INK)

    # Frozen Encoders
    cv.box(
        79.5, T(35.5), 25.0, 27.5,
        fill=C_SLATE_FILL, stroke=C_SLATE, lw=LW_BASE, radius=1.8, z=3,
    )
    cv.icon_snow(83.5, T(11.1), 2.5, color=C_HERO, lw=0.55)
    cv.icon_lock(100.5, T(11.1), 2.3, color=C_SLATE, lw=0.55)
    cv.text(92.0, T(10.8), "Frozen", size=SZ_BODY, weight="bold", color=C_INK)
    cv.text(92.0, T(13.8), "Encoders", size=SZ_BODY, weight="bold", color=C_INK)
    cv.text(92.0, T(16.6), "Pre-trained, then Frozen", size=SZ_SUB, color=C_HERO, style="italic")
    cv.rule(81.5, 102.5, T(18.5), lw=0.45, color=C_SLATE_SOFT)
    draw_encoder_schematics(cv, 83.2, T(21.4), T(26.8), T(32.2))

    # Low-Capacity Probes
    cv.box(
        79.5, T(60.0), 25.0, 20.5,
        fill="white", stroke=C_SLATE, lw=LW_BASE, radius=1.8, z=3,
    )
    cv.text(92.0, T(42.0), "Low-Capacity", size=SZ_BODY, weight="bold", color=C_INK)
    cv.text(92.0, T(44.9), "Probes", size=SZ_BODY, weight="bold", color=C_INK)
    cv.text(92.0, T(47.6), "Trainable / Fitted", size=SZ_SUB, color=C_MUTED, style="italic")
    cv.rule(81.5, 102.5, T(49.5), lw=0.45, color=C_SLATE_SOFT)
    draw_probe_schematics(cv, [84.5, 92.0, 99.5], T(53.4), T(57.8))

    # Arrows: Hero -> Frozen Encoders -> Probes
    cv.arrow([(75.0, T(20.2)), (79.5, T(20.2))], color=C_HERO, lw=LW_HERO, head=1.4)
    cv.arrow([(92.0, T(35.5)), (92.0, T(39.5))], color=C_SLATE, lw=LW_BASE, head=1.3)

    # -----------------------------------------------------------------------
    # Column 4: Evaluation Hierarchy L0-L2 (x = 109.5..154.0, w = 44.5)
    # -----------------------------------------------------------------------
    cv.text(131.75, T(4.2), "4. ForceBench Hierarchy", size=SZ_HEAD, weight="bold", color=C_EVAL)

    # L0 band
    cv.box(
        109.5, T(21.5), 44.5, 13.5,
        fill=C_EVAL_FILL, stroke=C_EVAL, lw=LW_BASE, radius=1.8, z=3,
    )
    cv.box(109.5, T(21.5), 8.2, 13.5, fill=C_EVAL, stroke="none", radius=1.8, z=4)
    cv.text(113.6, T(14.75), "L0", size=SZ_HEAD, weight="bold", color="white")
    cv.text(119.2, T(11.2), "Information Retention", size=SZ_BODY, weight="bold", color=C_EVAL, ha="left")
    cv.text(119.2, T(15.2), "L0.1 Native · L0.2 Unified", size=SZ_SUB, color=C_INK, ha="left")
    cv.text(119.2, T(18.6), "MAE, Dice, IoU, Load Err.", size=SZ_SUB, color=C_MUTED, ha="left")

    # L1 band
    cv.box(
        109.5, T(43.0), 44.5, 19.0,
        fill=C_EVAL_FILL, stroke=C_EVAL, lw=LW_BASE, radius=1.8, z=3,
    )
    cv.box(109.5, T(43.0), 8.2, 19.0, fill=C_EVAL, stroke="none", radius=1.8, z=4)
    cv.text(113.6, T(33.5), "L1", size=SZ_HEAD, weight="bold", color="white")
    cv.text(119.2, T(27.0), "Frozen Readability", size=SZ_BODY, weight="bold", color=C_EVAL, ha="left")
    cv.text(119.2, T(31.2), "L1a Contact (load, area)", size=SZ_SUB, color=C_INK, ha="left")
    cv.text(119.2, T(35.0), "L1b Force / Wrench", size=SZ_SUB, color=C_INK, ha="left")
    cv.text(119.2, T(38.8), "L1c Dynamics (ΔF, slip)", size=SZ_SUB, color=C_INK, ha="left")

    # L2 band
    cv.box(
        109.5, T(60.0), 44.5, 14.5,
        fill=C_EVAL_FILL, stroke=C_EVAL, lw=LW_BASE, radius=1.8, z=3,
    )
    cv.box(109.5, T(60.0), 8.2, 14.5, fill=C_EVAL, stroke="none", radius=1.8, z=4)
    cv.text(113.6, T(52.75), "L2", size=SZ_HEAD, weight="bold", color="white")
    cv.text(119.2, T(48.6), "Generalization & Robustness", size=SZ_BODY, weight="bold", color=C_EVAL, ha="left")
    cv.text(119.2, T(52.8), "Cross-Src · Hand ↔ Gripper", size=SZ_SUB, color=C_INK, ha="left")
    cv.text(119.2, T(56.4), "Cross-Qty · Perturbations", size=SZ_SUB, color=C_MUTED, ha="left")

    draw_hierarchy_schematics(cv, 148.0, T(15.2), T(34.5), T(53.8))

    # Fan-out arrows from Probes to L0, L1, L2 (using a clean shared vertical bus at x=107.0)
    cv.line([(104.5, T(49.0)), (107.0, T(49.0))], color=C_EVAL, lw=LW_BASE)
    cv.line([(107.0, T(14.8)), (107.0, T(52.8))], color=C_EVAL, lw=LW_BASE)
    for target_y in (T(14.8), T(33.5), T(52.8)):
        cv.arrow([(107.0, target_y), (109.5, target_y)], color=C_EVAL, lw=LW_BASE, head=1.2)

    # -----------------------------------------------------------------------
    # Bottom Key Strip (y = T(64.5))
    # -----------------------------------------------------------------------
    cv.rule(2.0, 154.0, T(62.2), lw=0.45, color=C_SLATE_SOFT)
    cv.icon_snow(40.0, T(65.0), 2.4, color=C_HERO, lw=0.5)
    cv.text(42.6, T(65.0), "Frozen encoder weights", size=SZ_SUB, color=C_MUTED, ha="left")

    cv.icon_lock(88.0, T(65.0), 2.3, color=C_SLATE, lw=0.5)
    cv.text(90.5, T(65.0), "Offline latent features", size=SZ_SUB, color=C_MUTED, ha="left")

    # -----------------------------------------------------------------------
    # Geometric audits & export
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

    out_stem = os.path.join(out_dir, "fig1_overview_v2")
    cv.save(out_stem, formats=("svg", "png"))
    print(f"Saved: {out_stem}.svg, {out_stem}.png")


if __name__ == "__main__":
    draw()
