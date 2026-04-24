"""
Visualization of vanishing points from MPIIFaceGaze calibration data.

Approach adapted from:
  Kocur et al., "Traffic Camera Calibration via Vehicle Vanishing Point Detection"
  https://github.com/kocurvik/deep_vp

Three-panel figure:
  1. Synthetic full-frame canvas with fan lines converging to each VP
  2. Diamond-space scatter (bounded VP representation from the paper)
  3. Sample face patch from the dataset
"""

import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

DATASET_ROOT = Path('data_sets/MPIIFaceGaze')

VP_COLORS_MPL = {
    'horizontal': (0.15, 0.75, 0.15),
    'vertical':   (0.85, 0.15, 0.15),
    'normal':     (0.15, 0.45, 0.90),
}
VP_COLORS_BGR = {
    'horizontal': (0,   200,   0),
    'vertical':   (0,     0, 200),
    'normal':     (200, 110,   0),
}


# ---------------------------------------------------------------------------
# Diamond-space helpers  (from deep_vp/utils/diamond_space.py)
# ---------------------------------------------------------------------------

def diamond_coords_from_original(p, d=1.0):
    """Map a 2-D point in image coords to diamond space."""
    p = np.array([p[0], p[1], 1.0])
    p_d = np.array([
        -d**2 * p[2],
        -d * p[0],
        np.sign(p[0] * p[1]) * p[0] + p[1] + np.sign(p[1]) * d * p[2],
    ])
    if abs(p_d[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    return p_d[:2] / p_d[2]


def vp_to_diamond(vp, scale=1.0):
    """Return the rotated diamond coordinate for a VP (normalized by scale)."""
    vp_scaled = np.array(vp) * scale
    dc = diamond_coords_from_original(vp_scaled, d=1.0)
    R = np.array([[1., -1.], [1., 1.]])
    return R @ dc


# ---------------------------------------------------------------------------
# Drawing helpers  (pretty_line from deep_vp/preview_heatmap.py)
# ---------------------------------------------------------------------------

def pretty_line(img, p1, p2, color_bgr, thickness=2):
    """Draw a line with a black outline and a coloured centre."""
    p1i = (int(p1[0]), int(p1[1]))
    p2i = (int(p2[0]), int(p2[1]))
    cv2.line(img, p1i, p2i, (0, 0, 0), thickness + 2)
    cv2.line(img, p1i, p2i, color_bgr, thickness)


def _ray_to_border(origin, direction, w, h):
    """Find the first image-border crossing for a ray from origin in direction."""
    ox, oy = origin
    dx, dy = direction
    ts = []
    for num, den in [(-ox, -dx), (w - ox, dx), (-oy, -dy), (h - oy, dy)]:
        if abs(den) > 1e-9:
            t = num / den
            x = ox + t * dx
            y = oy + t * dy
            if -1 <= x <= w + 1 and -1 <= y <= h + 1 and t > 1e-6:
                ts.append(t)
    if not ts:
        return None
    t = min(ts)
    return np.array([ox + t * dx, oy + t * dy])


def draw_vp_fan(canvas, vp, color_bgr, w, h, n_lines=9):
    """Draw n perspective-fan lines that converge at vp, clipped to the image."""
    vp = np.array(vp, dtype=float)
    in_frame = 0 <= vp[0] <= w and 0 <= vp[1] <= h

    # Sample spread points on the image boundary to fan toward
    border_ts = np.linspace(0, 1, n_lines + 1)[:-1]
    perimeter = 2 * (w + h)
    border_pts = []
    for t in border_ts:
        s = t * perimeter
        if s < w:
            border_pts.append([s, 0])
        elif s < w + h:
            border_pts.append([w, s - w])
        elif s < 2 * w + h:
            border_pts.append([2 * w + h - s, h])
        else:
            border_pts.append([0, perimeter - s])

    for bp in border_pts:
        bp = np.array(bp, dtype=float)
        if in_frame:
            start = vp
            direction = bp - vp
        else:
            # Ray from outside: start at vp, draw through image
            direction = bp - vp
            start = _ray_to_border(vp, direction, w, h)
            if start is None:
                continue
        end = _ray_to_border(start, direction, w, h) if not in_frame else bp
        if end is None:
            end = bp
        pretty_line(canvas, start, end, color_bgr, thickness=1)


# ---------------------------------------------------------------------------
# Calibration + VP computation
# ---------------------------------------------------------------------------

def load_calibration(participant_dir: Path):
    mat_cam = scipy.io.loadmat(str(participant_dir / 'Calibration' / 'Camera.mat'))
    mat_mon = scipy.io.loadmat(str(participant_dir / 'Calibration' / 'monitorPose.mat'))
    return mat_cam, mat_mon


def compute_vanishing_points(mat_cam: dict, mat_mon: dict) -> dict:
    K = mat_cam['cameraMatrix']
    R, _ = cv2.Rodrigues(mat_mon['rvects'].flatten())

    def project(d):
        p = K @ (R @ d)
        return np.array([p[0] / p[2], p[1] / p[2]])

    return {
        'horizontal': project(np.array([1., 0., 0.])),   # monitor X axis
        'vertical':   project(np.array([0., 1., 0.])),   # monitor Y axis
        'normal':     project(np.array([0., 0., 1.])),   # monitor normal
    }


# ---------------------------------------------------------------------------
# Focal length estimation from orthogonal vanishing points
# ---------------------------------------------------------------------------

def focal_length_from_vp_pair(vp1, vp2, cx, cy):
    """
    Estimate focal length from two vanishing points of orthogonal 3-D directions.

    Derivation: for orthogonal directions d1, d2 and intrinsic matrix K,
      (K^{-1} v1)^T (K^{-1} v2) = 0  =>  (vx1-cx)(vx2-cx) + (vy1-cy)(vy2-cy) + f^2 = 0
    """
    dot = (vp1[0] - cx) * (vp2[0] - cx) + (vp1[1] - cy) * (vp2[1] - cy)
    if dot >= 0:
        return None  # geometry violated — VPs not truly orthogonal given this cx,cy
    return float(np.sqrt(-dot))


def focal_length_from_vanishing_points(vps: dict, cx: float, cy: float) -> dict:
    """
    Estimate focal length from all three pairs of the three orthogonal VPs.
    Returns per-pair estimates and their mean.
    """
    pairs = [
        ('horizontal', 'vertical'),
        ('horizontal', 'normal'),
        ('vertical',   'normal'),
    ]
    estimates = {}
    for a, b in pairs:
        f = focal_length_from_vp_pair(vps[a], vps[b], cx, cy)
        estimates[f'{a[0].upper()}-{b[0].upper()}'] = f

    valid = [v for v in estimates.values() if v is not None]
    estimates['mean'] = float(np.mean(valid)) if valid else None
    return estimates


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def visualize_vanishing_points(participant: str = 'p00',
                               sample_image: str = 'day01/0005.jpg',
                               save_path: str = 'vp_visualization.png'):
    participant_dir = DATASET_ROOT / participant
    mat_cam, mat_mon = load_calibration(participant_dir)

    K = mat_cam['cameraMatrix']
    w, h = int(K[0, 2] * 2), int(K[1, 2] * 2)
    scale = 1.0 / max(w, h)          # normalise VP coords for diamond space

    vps = compute_vanishing_points(mat_cam, mat_mon)

    # ------------------------------------------------------------------ #
    # Panel 1 – synthetic full-frame canvas with fan lines               #
    # ------------------------------------------------------------------ #
    canvas = np.full((h, w, 3), 235, dtype=np.uint8)

    for name, vp in vps.items():
        draw_vp_fan(canvas, vp, VP_COLORS_BGR[name], w, h, n_lines=10)

    for name, vp in vps.items():
        if 0 <= vp[0] <= w and 0 <= vp[1] <= h:
            pt = (int(vp[0]), int(vp[1]))
            cv2.circle(canvas, pt, 11, (0, 0, 0), -1)
            cv2.circle(canvas, pt,  8, VP_COLORS_BGR[name], -1)
            cv2.putText(canvas, f'VP_{name[0].upper()}',
                        (pt[0] + 13, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (60, 60, 60), 2)

    # ------------------------------------------------------------------ #
    # Panel 2 – diamond-space scatter                                     #
    # ------------------------------------------------------------------ #
    diamond_pts = {}
    for name, vp in vps.items():
        dc = vp_to_diamond(vp, scale=scale)
        if np.isfinite(dc).all() and np.abs(dc).max() < 4.5:
            diamond_pts[name] = dc

    # ------------------------------------------------------------------ #
    # Panel 3 – face patch                                                #
    # ------------------------------------------------------------------ #
    img_path = participant_dir / sample_image
    face_img = cv2.imread(str(img_path))

    # ------------------------------------------------------------------ #
    # Assemble figure                                                      #
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.2, 1], wspace=0.25)
    ax_frame   = fig.add_subplot(gs[0])
    ax_diamond = fig.add_subplot(gs[1])
    ax_face    = fig.add_subplot(gs[2])

    # --- frame panel ---
    ax_frame.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax_frame.set_title(f'Vanishing Points in Camera Frame  [{participant}]', fontsize=11)
    ax_frame.set_xlabel('x (px)')
    ax_frame.set_ylabel('y (px)')

    legend_handles = []
    for name, color in VP_COLORS_MPL.items():
        vp = vps[name]
        in_frame = 0 <= vp[0] <= w and 0 <= vp[1] <= h
        label = f'VP_{name[0].upper()}  ({vp[0]:.0f}, {vp[1]:.0f} px)'
        if not in_frame:
            label += '  [off-screen]'
        legend_handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=label))
    ax_frame.legend(handles=legend_handles, loc='lower left', fontsize=8)

    # --- diamond panel ---
    ax_diamond.set_xlim(-2.5, 2.5)
    ax_diamond.set_ylim(-2.5, 2.5)
    ax_diamond.set_aspect('equal')
    ax_diamond.set_title('Diamond Space', fontsize=11)
    ax_diamond.set_xlabel('u')
    ax_diamond.set_ylabel('v')

    # Draw diamond boundary
    d_corners = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    ax_diamond.plot(d_corners[:, 0], d_corners[:, 1], 'k--', lw=1, alpha=0.6, label='VP domain')
    ax_diamond.axhline(0, color='gray', lw=0.5, alpha=0.4)
    ax_diamond.axvline(0, color='gray', lw=0.5, alpha=0.4)

    for name, dc in diamond_pts.items():
        ax_diamond.plot(dc[0], dc[1], 'o', color=VP_COLORS_MPL[name], markersize=11,
                        zorder=5, label=f'VP_{name[0].upper()}')
        ax_diamond.annotate(f'VP_{name[0].upper()}', dc, fontsize=8,
                            color=VP_COLORS_MPL[name],
                            xytext=(6, 6), textcoords='offset points')
    ax_diamond.legend(fontsize=8, loc='upper right')

    # --- face panel ---
    if face_img is not None:
        ax_face.imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        ax_face.set_title(f'Face Patch\n{sample_image}', fontsize=10)
    else:
        ax_face.text(0.5, 0.5, 'Image not found',
                     ha='center', va='center', transform=ax_face.transAxes)
    ax_face.axis('off')

    plt.suptitle('Perspective-Aware Gaze Estimation — Vanishing Point Visualization',
                 fontsize=12, y=1)

    plt.show()

    # Print summary
    print(f'\nFrame size: {w} x {h} px')
    print(f'Vanishing points for {participant}:')
    for name, vp in vps.items():
        in_frame = 0 <= vp[0] <= w and 0 <= vp[1] <= h
        dc_str = ''
        if name in diamond_pts:
            dc = diamond_pts[name]
            dc_str = f'  diamond=({dc[0]:.3f}, {dc[1]:.3f})'
        print(f'  VP_{name:12s}: ({vp[0]:8.1f}, {vp[1]:8.1f}) px'
              f'  {"[in-frame] " if in_frame else "[off-screen]"}{dc_str}')

    cx, cy = float(K[0, 2]), float(K[1, 2])
    f_gt   = float(K[0, 0])
    f_est  = focal_length_from_vanishing_points(vps, cx, cy)
    print(f'\nFocal length estimation (principal point: cx={cx:.1f}, cy={cy:.1f}):')
    for pair, f in f_est.items():
        if pair == 'mean':
            continue
        tag = f'  [{pair}]'
        val = f'{f:.2f} px' if f is not None else 'invalid (dot >= 0)'
        print(f'  {tag:20s} {val}')
    print(f'  {"[mean estimate]":20s} {f_est["mean"]:.2f} px')
    print(f'  {"[ground truth K]":20s} {f_gt:.2f} px  (error: {abs(f_est["mean"] - f_gt):.2f} px)'
          if f_est['mean'] is not None else '')


if __name__ == '__main__':
    visualize_vanishing_points(participant='p01', sample_image='day01/0005.jpg')
