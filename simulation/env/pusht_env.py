"""
Minimal PushT Environment

A 2D environment where a circular end-effector pushes a T-shaped block
toward a target pose. Renders to numpy arrays suitable for VLA input.

State space: [ee_x, ee_y, block_x, block_y, block_theta]
Action space: [delta_x, delta_y] — end-effector position delta
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Affine2D


# Environment constants
WORKSPACE = 512          # workspace is [0, 512] x [0, 512]
EE_RADIUS = 15           # end-effector circle radius
BLOCK_W = 80             # T-block horizontal bar width
BLOCK_H = 30             # T-block bar thickness
PUSH_COEFF = 0.15        # how much the block moves when pushed (lower = heavier block)
FRICTION = 0.7           # velocity damping per step (lower = more friction)


def make_t_shape(cx, cy, theta):
    """Return T-shaped polygon vertices centered at (cx, cy) rotated by theta.

    The T is made of two rectangles:
      - horizontal bar: BLOCK_W x BLOCK_H, centered at top
      - vertical stem:  BLOCK_H x BLOCK_W*0.6, hanging down from center of bar
    """
    hw = BLOCK_W / 2
    hh = BLOCK_H / 2
    stem_w = BLOCK_H / 2
    stem_h = BLOCK_W * 0.4

    # T-shape vertices (centered at origin, top bar + stem)
    verts = np.array([
        [-hw, hh],          # top-left of bar
        [hw, hh],           # top-right of bar
        [hw, -hh],          # bottom-right of bar
        [stem_w, -hh],      # right notch
        [stem_w, -hh - stem_h],   # bottom-right of stem
        [-stem_w, -hh - stem_h],  # bottom-left of stem
        [-stem_w, -hh],     # left notch
        [-hw, -hh],         # bottom-left of bar
    ])

    # Rotate
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T

    # Translate
    verts[:, 0] += cx
    verts[:, 1] += cy

    return verts


def point_in_polygon(px, py, verts):
    """Ray casting test for point in polygon."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def circle_polygon_overlap(cx, cy, radius, verts):
    """Approximate check: does circle overlap polygon?"""
    # Check if center is inside
    if point_in_polygon(cx, cy, verts):
        return True
    # Check if any polygon edge is close to circle center
    n = len(verts)
    for i in range(n):
        ax, ay = verts[i]
        bx, by = verts[(i + 1) % n]
        # Closest point on segment to circle center
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-8:
            continue
        t = max(0, min(1, ((cx - ax) * dx + (cy - ay) * dy) / seg_len_sq))
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
        if dist_sq < radius * radius:
            return True
    return False


class PushTEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        # End-effector starts at center-bottom
        self.ee_x = WORKSPACE / 2
        self.ee_y = WORKSPACE * 0.75

        # Block starts at a random position near center
        self.block_x = WORKSPACE / 2 + self.rng.uniform(-50, 50)
        self.block_y = WORKSPACE / 2 + self.rng.uniform(-50, 50)
        self.block_theta = self.rng.uniform(-0.3, 0.3)

        # Block velocity (for momentum)
        self.block_vx = 0.0
        self.block_vy = 0.0
        self.block_vtheta = 0.0

        # Target pose (fixed)
        self.target_x = WORKSPACE / 2
        self.target_y = WORKSPACE * 0.35
        self.target_theta = 0.0

        self.ee_trail = [(self.ee_x, self.ee_y)]
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return np.array([
            self.ee_x / WORKSPACE,
            self.ee_y / WORKSPACE,
            self.block_x / WORKSPACE,
            self.block_y / WORKSPACE,
            self.block_theta / np.pi,
        ], dtype=np.float32)

    def step(self, action):
        """Apply action [dx, dy] to end-effector, simulate push physics."""
        dx, dy = action[0], action[1]

        # Move end-effector
        old_ee_x, old_ee_y = self.ee_x, self.ee_y
        self.ee_x = np.clip(self.ee_x + dx, EE_RADIUS, WORKSPACE - EE_RADIUS)
        self.ee_y = np.clip(self.ee_y + dy, EE_RADIUS, WORKSPACE - EE_RADIUS)
        ee_dx = self.ee_x - old_ee_x
        ee_dy = self.ee_y - old_ee_y

        # Check collision with T-block
        verts = make_t_shape(self.block_x, self.block_y, self.block_theta)
        if circle_polygon_overlap(self.ee_x, self.ee_y, EE_RADIUS, verts):
            # Push ee out of block (resolve overlap)
            sep_x = self.ee_x - self.block_x
            sep_y = self.ee_y - self.block_y
            sep_dist = max(np.sqrt(sep_x**2 + sep_y**2), 1e-6)
            # Nudge ee outward along separation vector
            self.ee_x += sep_x / sep_dist * 2.0
            self.ee_y += sep_y / sep_dist * 2.0
            self.ee_x = np.clip(self.ee_x, EE_RADIUS, WORKSPACE - EE_RADIUS)
            self.ee_y = np.clip(self.ee_y, EE_RADIUS, WORKSPACE - EE_RADIUS)

            # Push block in direction of ee movement
            self.block_vx += ee_dx * PUSH_COEFF
            self.block_vy += ee_dy * PUSH_COEFF

            # Torque: cross product of (ee_pos - block_center) x push_dir
            rx = self.ee_x - self.block_x
            ry = self.ee_y - self.block_y
            torque = (rx * ee_dy - ry * ee_dx) * 0.0001
            self.block_vtheta += torque

        # Apply block velocity
        self.block_x += self.block_vx
        self.block_y += self.block_vy
        self.block_theta += self.block_vtheta

        # Damping
        self.block_vx *= FRICTION
        self.block_vy *= FRICTION
        self.block_vtheta *= FRICTION

        # Keep block in workspace
        margin = BLOCK_W / 2 + 10
        self.block_x = np.clip(self.block_x, margin, WORKSPACE - margin)
        self.block_y = np.clip(self.block_y, margin, WORKSPACE - margin)

        self.ee_trail.append((self.ee_x, self.ee_y))
        if len(self.ee_trail) > 100:  # keep last 100 positions
            self.ee_trail.pop(0)
        self.step_count += 1
        return self.get_state()

    def render(self, dpi=72, info=None):
        """Render environment to RGB numpy array [H, W, 3] uint8.

        info: optional dict with overlay text, e.g.
              {"inference_ms": 1.4, "chunk": 3}
        """
        fig, ax = plt.subplots(1, 1, figsize=(512/dpi, 512/dpi), dpi=dpi)
        ax.set_xlim(0, WORKSPACE)
        ax.set_ylim(0, WORKSPACE)
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f0f0")
        ax.axis("off")

        # Draw target T (outline)
        target_verts = make_t_shape(self.target_x, self.target_y, self.target_theta)
        target_poly = plt.Polygon(target_verts, fill=False, edgecolor="#90EE90",
                                  linewidth=2, linestyle="--", zorder=1)
        ax.add_patch(target_poly)

        # Draw end-effector trail
        if len(self.ee_trail) > 1:
            trail = np.array(self.ee_trail)
            ax.plot(trail[:, 0], trail[:, 1], color="#FF6B6B", alpha=0.3,
                    linewidth=2, zorder=1.5)

        # Draw T-block (filled)
        block_verts = make_t_shape(self.block_x, self.block_y, self.block_theta)
        block_poly = plt.Polygon(block_verts, facecolor="#4A90D9", edgecolor="#2C5F8A",
                                 linewidth=1.5, zorder=2)
        ax.add_patch(block_poly)

        # Draw end-effector
        ee_circle = plt.Circle((self.ee_x, self.ee_y), EE_RADIUS,
                               facecolor="#FF6B6B", edgecolor="#CC4444",
                               linewidth=1.5, zorder=3)
        ax.add_patch(ee_circle)

        # Info overlay (top-left)
        lines = [f"step {self.step_count}"]
        if info:
            if "inference_ms" in info:
                lines.append(f"inference: {info['inference_ms']:.1f} ms/step")
            if "chunk" in info:
                lines.append(f"chunk {info['chunk']}")
            if "backend" in info:
                lines.append(info["backend"])
        ax.text(10, WORKSPACE - 15, "\n".join(lines),
                fontsize=8, color="#444444", fontfamily="monospace",
                verticalalignment="top", zorder=4)

        fig.tight_layout(pad=0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        plt.close(fig)
        return img[:, :, :3]  # drop alpha


    def render_for_vla(self):
        """Render as [3, 512, 512] float32 normalized image for VLA input."""
        img = self.render(dpi=72)  # [H, W, 3] uint8
        # Resize to 512x512 if needed (should already be close)
        from PIL import Image
        img_pil = Image.fromarray(img).resize((512, 512), Image.BILINEAR)
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        # HWC -> CHW
        return img_np.transpose(2, 0, 1)
