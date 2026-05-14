"""
Multi-target PushT environment.

Extends PushTEnv with 4 named target zones controlled by text command.
State is 7D: [ee_x, ee_y, block_x, block_y, theta, target_x, target_y].
"""

import numpy as np
from env.pusht_env import PushTEnv, WORKSPACE, make_t_shape

COMMANDS = ["up", "down", "left", "right"]

TARGETS = {
    "up":    (WORKSPACE * 0.50, WORKSPACE * 0.20),  # (256, 102)
    "down":  (WORKSPACE * 0.50, WORKSPACE * 0.78),  # (256, 400)
    "left":  (WORKSPACE * 0.20, WORKSPACE * 0.50),  # (102, 256)
    "right": (WORKSPACE * 0.80, WORKSPACE * 0.50),  # (410, 256)
}

COMMAND_PROMPTS = {
    "up":    "push the block up",
    "down":  "push the block down",
    "left":  "push the block left",
    "right": "push the block right",
}

# Colors per zone for rendering
ZONE_COLORS = {
    "up":    "#22c55e",   # green
    "down":  "#3b82f6",   # blue
    "left":  "#f59e0b",   # amber
    "right": "#ef4444",   # red
}


class PushTEnvMulti(PushTEnv):
    """PushT with 4 named target zones selected by text command."""

    def reset(self, command: str = None, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Pick random command if not specified
        if command is None:
            command = self.rng.choice(COMMANDS)
        self.command = command

        # Run parent reset (sets default target, ee, block positions)
        super().reset()

        # Override target with command-specific zone
        self.target_x, self.target_y = TARGETS[command]
        self.target_theta = 0.0

        return self.get_state()

    def get_state(self):
        """7D state: [ee_x, ee_y, block_x, block_y, theta, target_x, target_y]."""
        return np.array([
            self.ee_x / WORKSPACE,
            self.ee_y / WORKSPACE,
            self.block_x / WORKSPACE,
            self.block_y / WORKSPACE,
            self.block_theta / np.pi,
            self.target_x / WORKSPACE,
            self.target_y / WORKSPACE,
        ], dtype=np.float32)

    def render(self, dpi=72, info=None):
        """Render with all 4 zones visible; active zone highlighted, others faint."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from env.pusht_env import make_t_shape, WORKSPACE, EE_RADIUS

        fig, ax = plt.subplots(1, 1, figsize=(512 / dpi, 512 / dpi), dpi=dpi)
        ax.set_xlim(0, WORKSPACE)
        ax.set_ylim(0, WORKSPACE)
        ax.set_aspect("equal")
        ax.set_facecolor("#f0f0f0")
        ax.axis("off")

        cmd = getattr(self, "command", "")

        # Draw all 4 target zones
        for zone, (tx, ty) in TARGETS.items():
            verts = make_t_shape(tx, ty, 0.0)
            color = ZONE_COLORS[zone]
            if zone == cmd:
                # Active zone: filled + solid border
                poly = plt.Polygon(verts, facecolor=color + "55", edgecolor=color,
                                   linewidth=2.5, linestyle="-", zorder=1)
            else:
                # Inactive zones: outline only, very faint
                poly = plt.Polygon(verts, fill=False, edgecolor=color + "55",
                                   linewidth=1.5, linestyle="--", zorder=1)
            ax.add_patch(poly)

            # Zone label
            label_y = ty + 38 if ty < WORKSPACE / 2 else ty - 48
            ax.text(tx, label_y, zone, fontsize=7, color=color,
                    ha="center", va="center", fontfamily="monospace",
                    alpha=1.0 if zone == cmd else 0.4, zorder=5)

        # EE trail
        if len(self.ee_trail) > 1:
            trail = np.array(self.ee_trail)
            ax.plot(trail[:, 0], trail[:, 1], color="#FF6B6B", alpha=0.3,
                    linewidth=2, zorder=1.5)

        # Block
        block_verts = make_t_shape(self.block_x, self.block_y, self.block_theta)
        ax.add_patch(plt.Polygon(block_verts, facecolor="#4A90D9", edgecolor="#2C5F8A",
                                 linewidth=1.5, zorder=2))

        # End-effector
        ax.add_patch(plt.Circle((self.ee_x, self.ee_y), EE_RADIUS,
                                facecolor="#FF6B6B", edgecolor="#CC4444",
                                linewidth=1.5, zorder=3))

        # Command label
        color = ZONE_COLORS.get(cmd, "#22c55e")
        ax.text(10, WORKSPACE - 12,
                f'"{COMMAND_PROMPTS.get(cmd, cmd)}"',
                fontsize=9, color=color, fontfamily="monospace",
                verticalalignment="top", zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#000000aa", edgecolor="none"))

        # Step info
        lines = [f"step {self.step_count}"]
        if info:
            if "inference_ms" in info:
                lines.append(f"{info['inference_ms']:.0f} ms/step")
            if "chunk" in info:
                lines.append(f"chunk {info['chunk']}")
        ax.text(10, 10, "  ".join(lines), fontsize=7, color="#666666",
                fontfamily="monospace", verticalalignment="bottom", zorder=6)

        fig.tight_layout(pad=0)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        plt.close(fig)
        return buf[:, :, :3]
