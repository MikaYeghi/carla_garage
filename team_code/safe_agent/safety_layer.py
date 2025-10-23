
# -*- coding: utf-8 -*-
"""
SafetyLayer: faithful Python reimplementation of the C++ DepthGroundRemover pipeline
from depth_clustering (Bogoslavskyi & Stachniss), focused on ground removal.

Implements the following (mirroring the C++):
- Range image construction (partial FOV safe)
- RepairDepth (two-sided vertical inpaint)
- CreateAngleImage (using row-angle sines/cosines)
- Savitzky–Golay smoothing (row-wise)
- Bottom-seeded BFS ground labeling with angle threshold
- Two-tap vertical dilation of ground labels
- Nonground mask output

Also provides manual-call visualizations:
- visualize_range_image(ri)
- visualize_angle_image(angle_image)
- visualize_ground_overlay(points, ri_idx, nonground_mask)

Downstream functions (e.g., obstacle boxes, velocity limiting) are stubbed.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import math
import numpy as np

# Visualizations (call manually)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# --- Public dataclass for output ---
@dataclass
class Obstacle:
    center: Tuple[float, float, float]
    extent: Tuple[float, float, float]
    yaw: float
    num_points: int
    cell_indices: List[Tuple[int, int]]  # (row, col) pixels that formed this cluster

@dataclass
class LidarGeometry:
    n_rings: int
    n_cols: int
    vert_fov_up: float      # radians
    vert_fov_down: float    # radians
    max_range: float        # meters
    full_azimuth: bool = False   # if True, wrap columns (360°); else partial sweep (no wrap)


class SafetyLayer:
    """
    Faithful reimplementation of:
      - depth_ground_remover.{RepairDepth, CreateAngleImage, ApplySavitskyGolaySmoothing, ZeroOutGroundBFS}
      - image_based_clusterer + linear_image_labeler (angles-based diff, 4-neighborhood)

    Inputs:
      lidar_data = (frame_or_ts, Nx(3 or 4) array [x,y,z,(intensity)])
    Returns:
      List[Obstacle]
    """

    # ---- Defaults from depth_clustering/parameter.h (with sensible CARLA sensor defaults) ----
    def __init__(self,
                 n_rings: int = 64,
                 n_cols: int = 235,              # ~¼ turn for your stream (~15k pts @ 64 rings)
                 vert_fov_up_deg: float = 10.0,
                 vert_fov_down_deg: float = -30.0,
                 max_range_m: float = 85.0,
                 # depth_clustering defaults:
                 angle_ground_remove_deg: float = 9.0,   # _ground_remove_angle
                 angle_cluster_deg: float = 10.0,        # _angle_tollerance
                 size_smooth_window: int = 5,            # SG window
                 min_cluster_size: int = 10,
                 max_cluster_size: int = 20000,
                 wrap_cols: bool = False                 # set True if full 360° sweep
                 ) -> None:

        self.n_rings = int(n_rings)
        self.n_cols = int(n_cols)
        self.vert_fov_up = math.radians(vert_fov_up_deg)
        self.vert_fov_down = math.radians(vert_fov_down_deg)
        self.max_range = float(max_range_m)

        self.alpha_ground = math.radians(angle_ground_remove_deg)
        self.alpha_cluster = math.radians(angle_cluster_deg)
        self.sg_window = int(size_smooth_window)

        self.min_cluster_size = int(min_cluster_size)
        self.max_cluster_size = int(max_cluster_size)
        self.wrap_cols = bool(wrap_cols)

        # IMPORTANT: row 0 is TOP ring (largest elevation), last row is BOTTOM ring (smallest).
        # This matches the C++ ProjectionParams row convention used in CreateAngleImage().
        self._row_angles = np.linspace(self.vert_fov_up, self.vert_fov_down,
                                       self.n_rings, dtype=np.float32)

    # -------------------- Public API --------------------

    def detect_obstacles(self, lidar_data: Tuple[int | float, np.ndarray]) -> List[Obstacle]:
        frame_id, pts = self._parse_lidar(lidar_data)

        # 1) Build range image + back-index to original points
        ri, ri_idx = self._range_image_partial_sweep(pts)

        # 2) Ground removal (faithful to DepthGroundRemover)
        #    2.1 Repair zeros vertically
        repaired = self._repair_depth(ri, step=5, depth_threshold=1.0)
        #    2.2 Make angle image between vertical neighbors using row-angle sines/cosines
        angle_img = self._create_angle_image(repaired)
        #    2.3 Optional Savitzky–Golay smoothing (vertical)
        angle_used = self._apply_savgol(angle_img, self.sg_window)

        #    2.4 BFS from bottom valid pixel in each column using SimpleDiff(angle) + threshold
        ground_mask = self._zero_out_ground_bfs(repaired, angle_used,
                                                radians_threshold=self.alpha_ground,
                                                kernel_size=self.sg_window)

        # 3) Image-based clustering on NON-GROUND using angles diff (LinearImageLabeler, 4-neigh)
        nonground_mask = (repaired > 0.001) & (~ground_mask)
        labels, sizes, nlabels = self._label_components_angles(repaired, nonground_mask,
                                                               radians_threshold=self.alpha_cluster,
                                                               wrap_cols=self.wrap_cols)

        # 4) Build 3D clusters, filter by size, make AABBs
        obstacles = self._clusters_to_obstacles(labels, sizes, nlabels, ri_idx, pts)
        return obstacles

    def ground_removal(self, ri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """High-level ground removal that returns (non_ground_mask, angle_image)."""
        repaired = self._repair_depth_two_sided(ri, step=5, depth_threshold=1.0)
        angle_image = self._create_angle_image(repaired, self._row_angles)

        if self.sg_window in (5, 7, 9, 11):
            angle_used = self._savitzky_golay_smooth(angle_image, self.sg_window)
        else:
            angle_used = angle_image

        labels = self._zero_out_ground_bfs(repaired, angle_used,
                                           threshold=self.alpha_thresh,
                                           kernel_size=self.sg_window)
        nonground = self._dilate_and_compose(repaired, labels, kernel_size=self.sg_window)
        return nonground, angle_image
    
    def _range_image_partial_sweep(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ProjectionParams-like range image for a partial azimuth sweep.
        - Rows are rings (top->bottom).
        - Columns are evenly divided over the *observed* azimuth span (no 360 wrap).
        """
        xyz = pts[:, :3]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.sqrt((x*x + y*y + z*z))
        keep = (r > 0.1) & (r < self.max_range) & np.isfinite(r)
        x, y, z, r = x[keep], y[keep], z[keep], r[keep]
        kept_idx = np.nonzero(keep)[0]

        # azimuth in [0, 2π)
        az = np.arctan2(y, x)
        az[az < 0.0] += 2.0 * math.pi
        # dynamic span for partial sweep
        az_min, az_max = az.min(), az.max()
        span = max(az_max - az_min, 1e-3)
        cols = np.floor((az - az_min) / span * (self.n_cols - 1)).astype(np.int32)

        # elevation -> row index (top row = largest elevation)
        el = np.arctan2(z, np.sqrt(x*x + y*y))
        rows = np.floor((self.vert_fov_up - el) /
                        (self.vert_fov_up - self.vert_fov_down) *
                        (self.n_rings - 1)).astype(np.int32)
        np.clip(rows, 0, self.n_rings - 1, out=rows)
        np.clip(cols, 0, self.n_cols - 1, out=cols)

        ri = np.full((self.n_rings, self.n_cols), 0.0, dtype=np.float32)  # C++ depth images use 0 for empty
        ri_idx = np.full((self.n_rings, self.n_cols), -1, dtype=np.int32)

        # fill *nearest* first: sort by range ascending
        order = np.argsort(r)
        filled = np.zeros(self.n_rings * self.n_cols, dtype=bool)
        for k in order:
            rr, cc = int(rows[k]), int(cols[k])
            f = rr * self.n_cols + cc
            if not filled[f]:
                filled[f] = True
                ri[rr, cc] = r[k]
                ri_idx[rr, cc] = int(kept_idx[k])

        return ri, ri_idx

    def range_image(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[float,float]]:
        """Build range image and index map from point cloud (N x >=3).

        Returns:
            ri: (R x C) depth image (meters), np.inf where empty
            ri_idx: (R x C) indices into input points (-1 where empty)
            az_span: (az_min, az_max) radians used for column mapping (partial sweeps)
        """
        g = self.geom
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x*x + y*y + z*z)
        keep = (r > self.eps_depth) & (r < g.max_range) & np.isfinite(r)
        if not np.any(keep):
            R, C = g.n_rings, g.n_cols
            return (np.full((R, C), np.inf, dtype=np.float32),
                    np.full((R, C), -1, dtype=np.int32),
                    (0.0, 0.0))

        x, y, z, r = x[keep], y[keep], z[keep], r[keep]
        idx_kept = np.nonzero(keep)[0]

        # Angles
        az = np.arctan2(y, x)              # [-pi, pi]
        az[az < 0.0] += 2.0 * math.pi      # [0, 2pi)
        el = np.arctan2(z, np.sqrt(x*x + y*y))  # elevation

        # Rows: linear map from [vert_fov_up .. vert_fov_down] -> [0 .. R-1]
        # row 0 = top, row R-1 = bottom
        R = g.n_rings
        C = g.n_cols
        row = np.floor((g.vert_fov_up - el) / (g.vert_fov_up - g.vert_fov_down) * (R - 1)).astype(np.int32)
        row = np.clip(row, 0, R - 1)

        # Cols:
        if g.full_azimuth:
            col = np.floor(az / (2.0 * math.pi) * C).astype(np.int32)
        else:
            az_min = az.min()
            az_max = az.max()
            span = max(1e-6, az_max - az_min)
            col = np.floor((az - az_min) / span * (C - 1)).astype(np.int32)
        col = np.clip(col, 0, C - 1)

        # Fill RI with nearest-first policy
        ri = np.full((R, C), np.inf, dtype=np.float32)
        ri_idx = np.full((R, C), -1, dtype=np.int32)

        flat = row * C + col
        order = np.argsort(r)  # nearest first
        seen = np.zeros(R * C, dtype=bool)
        for k in order:
            f = int(flat[k])
            if not seen[f]:
                seen[f] = True
                rr = int(row[k]); cc = int(col[k])
                ri[rr, cc] = r[k]
                ri_idx[rr, cc] = int(idx_kept[k])

        return ri, ri_idx, (float(az.min()), float(az.max()))

    # -------------------- Visualizations (manual) --------------------

    @staticmethod
    def visualize_range_image(ri: np.ndarray, vmax: Optional[float] = None) -> None:
        """Show range image (depth meters); np.inf shown as white."""
        depth = ri.copy()
        depth[~np.isfinite(depth)] = np.nan
        plt.figure(figsize=(10, 4))
        plt.imshow(depth, cmap='viridis', origin='upper', vmin=0.0, vmax=vmax)
        plt.colorbar(label='Depth [m]')
        plt.title('Range Image (row 0 = top ring)')
        plt.xlabel('Azimuth columns')
        plt.ylabel('Rings')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_angle_image(angle_image: np.ndarray, max_angle_deg: float = 30.0) -> None:
        """Show angle image (radians) as degrees."""
        angle_deg = np.degrees(angle_image)
        angle_deg = np.clip(angle_deg, 0, max_angle_deg)
        plt.figure(figsize=(10, 4))
        plt.imshow(angle_deg, cmap='turbo', origin='upper')
        plt.colorbar(label='Inclination angle [deg]')
        plt.title('Angle Image (vertical steepness)')
        plt.xlabel('Azimuth columns')
        plt.ylabel('Rings')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_ground_overlay(points: np.ndarray,
                                 ri_idx: np.ndarray,
                                 nonground_mask: np.ndarray) -> None:
        """3D scatter with ground (green) vs non-ground (red)."""
        if points.shape[1] > 3:
            points = points[:, :3]

        H, W = ri_idx.shape
        valid = ri_idx >= 0
        nonground = nonground_mask & valid
        ground = (~nonground) & valid

        g_idx = ri_idx[ground].ravel()
        ng_idx = ri_idx[nonground].ravel()

        pts_g = points[g_idx]
        pts_ng = points[ng_idx]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=40)
        if len(pts_g) > 0:
            ax.scatter(pts_g[:, 0], pts_g[:, 1], pts_g[:, 2], c='green', s=1, label='Ground')
        if len(pts_ng) > 0:
            ax.scatter(pts_ng[:, 0], pts_ng[:, 1], pts_ng[:, 2], c='red', s=1, label='Non-ground')
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        ax.set_title('Ground Segmentation Overlay')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_obstacles_3d(points: np.ndarray,
                               obstacles: List["Obstacle"],
                               ri_idx: Optional[np.ndarray] = None,
                               show_ground: bool = False,
                               nonground_mask: Optional[np.ndarray] = None) -> None:
        """
        3D scatter visualization of LiDAR point cloud with obstacle bounding boxes.
        - points: (N,3) array of LiDAR XYZ points.
        - obstacles: list of Obstacle dataclasses from detect_obstacles().
        - ri_idx: optional range-image index map (to filter valid points).
        - show_ground: if True, color ground green/red using nonground_mask like visualize_ground_overlay.
        - nonground_mask: optional mask for ground highlighting.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.cm as cm

        if points.shape[1] > 3:
            points = points[:, :3]

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=40)

        # Plot points
        if show_ground and nonground_mask is not None and ri_idx is not None:
            H, W = ri_idx.shape
            valid = ri_idx >= 0
            nonground = nonground_mask & valid
            ground = (~nonground) & valid
            g_idx = ri_idx[ground].ravel()
            ng_idx = ri_idx[nonground].ravel()
            if len(g_idx) > 0:
                ax.scatter(points[g_idx, 0], points[g_idx, 1], points[g_idx, 2],
                           c='green', s=1, label='Ground')
            if len(ng_idx) > 0:
                ax.scatter(points[ng_idx, 0], points[ng_idx, 1], points[ng_idx, 2],
                           c='red', s=1, label='Non-ground')
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                       c='gray', s=0.5, alpha=0.5, label='All points')

        # Distinct colors for obstacles
        n_obs = len(obstacles)
        cmap = cm.get_cmap('tab20', n_obs)

        def draw_bbox(ax, center, extent, color):
            cx, cy, cz = center
            ex, ey, ez = extent
            # corners of box
            corners = np.array([
                [cx - ex, cy - ey, cz - ez],
                [cx + ex, cy - ey, cz - ez],
                [cx + ex, cy + ey, cz - ez],
                [cx - ex, cy + ey, cz - ez],
                [cx - ex, cy - ey, cz + ez],
                [cx + ex, cy - ey, cz + ez],
                [cx + ex, cy + ey, cz + ez],
                [cx - ex, cy + ey, cz + ez]
            ])
            # faces (6)
            faces = [
                [corners[j] for j in [0, 1, 2, 3]],  # bottom
                [corners[j] for j in [4, 5, 6, 7]],  # top
                [corners[j] for j in [0, 1, 5, 4]],  # front
                [corners[j] for j in [2, 3, 7, 6]],  # back
                [corners[j] for j in [1, 2, 6, 5]],  # right
                [corners[j] for j in [4, 7, 3, 0]],  # left
            ]
            box = Poly3DCollection(faces, alpha=0.15, facecolor=color, edgecolor=color)
            ax.add_collection3d(box)

        # Plot bounding boxes
        for i, obs in enumerate(obstacles):
            color = cmap(i % cmap.N)
            draw_bbox(ax, obs.center, obs.extent, color)
            cx, cy, cz = obs.center
            ax.text(cx, cy, cz + obs.extent[2] + 0.2,
                    f'#{i+1}', color='black', fontsize=8,
                    ha='center', va='bottom')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title(f'Detected Obstacles: {len(obstacles)}')
        ax.legend(loc='upper right', markerscale=4)
        plt.tight_layout()
        plt.show()


    # -------------------- Internals: C++ faithful parts --------------------

    @staticmethod
    def _parse_lidar(lidar_data: Tuple[int | float, np.ndarray]) -> Tuple[int | float, np.ndarray]:
        if not isinstance(lidar_data, (tuple, list)) or len(lidar_data) != 2:
            raise ValueError("lidar_data must be (frame_or_ts, points)")
        frame_id, pts = lidar_data
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"points must have shape (N, >=3); got {pts.shape}")
        return frame_id, pts
    
    @staticmethod
    def _repair_depth(depth: np.ndarray, step: int = 5, depth_threshold: float = 1.0) -> np.ndarray:
        """C++ RepairDepth(no_ground_image, step, depth_threshold)"""
        R, C = depth.shape
        out = depth.astype(np.float32, copy=True)
        # treat NaN/Inf as zero
        out[~np.isfinite(out)] = 0.0

        for c in range(C):
            col = out[:, c]
            for r in range(R):
                if col[r] >= 0.001:
                    continue
                counter = 0
                s = 0.0
                for i in range(1, step):
                    ru = r - i
                    if ru < 0:
                        continue
                    prev = col[ru]
                    if prev <= 0.001:
                        continue
                    for j in range(1, step):
                        rd = r + j
                        if rd >= R:
                            continue
                        nxt = col[rd]
                        if nxt > 0.001 and abs(prev - nxt) < depth_threshold:
                            s += prev + nxt
                            counter += 2
                if counter > 0:
                    col[r] = s / counter
            out[:, c] = col
        return out

    def _repair_depth_two_sided(self, image: np.ndarray, step: int, depth_threshold: float) -> np.ndarray:
        """C++ RepairDepth(no_ground_image, 5, 1.0f). Fill small vertical gaps if neighbors agree."""
        eps = self.eps_depth
        depth = image.astype(np.float32, copy=True)
        depth[~np.isfinite(depth)] = 0.0

        R, C = depth.shape
        for c in range(C):
            col = depth[:, c]
            for r in range(R):
                if col[r] < eps:
                    counter = 0
                    acc = 0.0
                    for i in range(1, step):
                        ru = r - i
                        if ru < 0:
                            continue
                        prev = col[ru]
                        if prev <= eps:
                            continue
                        for j in range(1, step):
                            rd = r + j
                            if rd >= R:
                                continue
                            nxt = col[rd]
                            if nxt > eps and abs(prev - nxt) < depth_threshold:
                                acc += prev + nxt
                                counter += 2
                    if counter > 0:
                        col[r] = acc / counter
            depth[:, c] = col
        return depth

    def _create_angle_image(self, depth: np.ndarray) -> np.ndarray:
        """C++ CreateAngleImage: angle between vertical neighbors in (x,z)-like space using row sines/cosines."""
        R, C = depth.shape
        angle_img = np.zeros((R, C), dtype=np.float32)
        sin_row = np.sin(self._row_angles)[:, None]  # (R,1)
        cos_row = np.cos(self._row_angles)[:, None]

        # x_row = depth * cos(theta_row), y_row = depth * sin(theta_row)
        x_mat = depth * cos_row
        y_mat = depth * sin_row

        dx = np.abs(x_mat[1:, :] - x_mat[:-1, :])
        dy = np.abs(y_mat[1:, :] - y_mat[:-1, :])
        angle_img[1:, :] = np.arctan2(dy, dx).astype(np.float32)
        return angle_img

    @staticmethod
    def _sg_kernel(window: int) -> np.ndarray:
        if window not in (5, 7, 9, 11):
            raise ValueError("Savitzky–Golay window must be one of {5,7,9,11}")
        if window == 5:
            k = np.array([-3, 12, 17, 12, -3], np.float32) / 35.0
        elif window == 7:
            k = np.array([-2, 3, 6, 7, 6, 3, -2], np.float32) / 21.0
        elif window == 9:
            k = np.array([-21, 14, 39, 54, 59, 54, 39, 14, -21], np.float32) / 231.0
        else:
            k = np.array([-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36], np.float32) / 429.0
        return k

    @staticmethod
    def _reflect101_index(i: int, n: int) -> int:
        """OpenCV BORDER_REFLECT101 behavior for 1D index."""
        if n == 1:
            return 0
        if i < 0:
            i = -i
        block = n * 2 - 2
        i_mod = i % block
        return i_mod if i_mod < n else block - i_mod
    
    def _apply_savgol(self, angle_img: np.ndarray, window: int) -> np.ndarray:
        """C++ ApplySavitskyGolaySmoothing on columns (vertical), BORDER_REFLECT101."""
        if window not in (5, 7, 9, 11):
            return angle_img
        R, C = angle_img.shape
        k = self._sg_kernel(window)
        half = window // 2
        out = np.zeros_like(angle_img, dtype=np.float32)

        # BORDER_REFLECT101 padding emulation
        def ref_idx(i: int, n: int) -> int:
            if n == 1:
                return 0
            if i < 0:
                i = -i
            block = n * 2 - 2
            t = i % block
            return t if t < n else block - t

        for r in range(R):
            acc = np.zeros(C, dtype=np.float32)
            for t in range(window):
                rr = ref_idx(r + t - half, R)
                acc += k[t] * angle_img[rr, :]
            out[r, :] = acc
        return out

    def _savitzky_golay_smooth(self, image: np.ndarray, window: int) -> np.ndarray:
        """1D vertical convolution with SG kernel, reflect padding like OpenCV."""
        R, C = image.shape
        k = self._sg_kernel(window)
        half = window // 2
        out = np.zeros_like(image, dtype=np.float32)
        for r in range(R):
            acc = np.zeros(C, dtype=np.float32)
            for t in range(window):
                rr = self._reflect101_index(r + (t - half), R)
                acc += k[t] * image[rr, :]
            out[r, :] = acc
        return out

    def _zero_out_ground_bfs(self,
                             depth: np.ndarray,
                             angle_img: np.ndarray,
                             radians_threshold: float,
                             kernel_size: int) -> np.ndarray:
        """
        C++ ZeroOutGroundBFS: seed BFS at bottom valid pixel in each column,
        expand where SimpleDiff(angle) < threshold; then dilate vertically with two-tap kernel.
        Returns: ground_mask (True for ground).
        """
        R, C = depth.shape
        valid = depth > 0.001
        labels = np.zeros((R, C), dtype=np.uint16)

        from collections import deque
        def simple_diff_ok(r_from: int, r_to: int, c: int) -> bool:
            # Angle index is the *greater* row of the two (as in C++)
            rr = r_to if r_to > r_from else r_from
            if rr <= 0 or rr >= R:
                return False
            v = angle_img[rr, c]
            return np.isfinite(v) and (v < radians_threshold)

        # Seed per column at bottom-most valid pixel
        next_label = 1
        for c in range(C):
            r = R - 1
            while r > 0 and not valid[r, c]:
                r -= 1
            if r <= 0:
                continue
            if labels[r, c] > 0:
                continue

            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                if labels[rr, cc] > 0 or not valid[rr, cc]:
                    continue
                labels[rr, cc] = next_label
                # 4-neighborhood with column wrap disabled here (ground removal uses per-column BFS)
                for rn, cn in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if rn < 0 or rn >= R or cn < 0 or cn >= C:
                        continue
                    if labels[rn, cn] > 0 or not valid[rn, cn]:
                        continue
                    # vertical move needs angle check
                    if rn != rr and not simple_diff_ok(rr, rn, cn):
                        continue
                    q.append((rn, cn))
            next_label += 1

        # Dilate vertically with "uniform kernel": [1, 0, …, 0, 1]
        ksz = max(kernel_size - 2, 3)
        if ksz % 2 == 0:
            ksz += 1
        shift = ksz - 1
        ground = labels > 0
        if shift > 0:
            up = np.zeros_like(ground, dtype=bool)
            down = np.zeros_like(ground, dtype=bool)
            up[:-shift, :] = ground[shift:, :]
            down[shift:, :] = ground[:-shift, :]
            ground = ground | up | down

        return ground

    def _dilate_and_compose(self, depth_image: np.ndarray,
                            labels: np.ndarray,
                            kernel_size: int) -> np.ndarray:
        """C++: kernel_size = max(window_size - 2, 3); two-tap vertical dilation; compose masks."""
        R, C = labels.shape
        kernel_size = max(int(kernel_size) - 2, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        shift = kernel_size - 1

        ground = labels > 0
        if shift > 0:
            up = np.zeros_like(ground, dtype=bool)
            down = np.zeros_like(ground, dtype=bool)
            up[:-shift, :] = ground[shift:, :]
            down[shift:, :] = ground[:-shift, :]
            ground = ground | up | down

        valid = depth_image > self.eps_depth
        nonground = valid & (~ground)
        return nonground
    
    def _label_components_angles(self,
                                 depth: np.ndarray,
                                 mask: np.ndarray,
                                 radians_threshold: float,
                                 wrap_cols: bool) -> Tuple[np.ndarray, Dict[int, int], int]:
        """
        Faithful to ImageBasedClusterer + LinearImageLabeler:
          - BFS over 4-neighbors (up,down,left,right)
          - diff = angle between vertical neighbors (ANGLES_PRECOMPUTED equivalent)
          - 'mask' gates which pixels are allowed (here: NON-GROUND & valid)
          - wrap in columns only if 360° FOV
        Returns:
          labels(int32), sizes{label:size}, nlabels
        """
        R, C = depth.shape
        labels = np.zeros((R, C), dtype=np.int32)
        sizes: Dict[int, int] = {}
        next_label = 1

        # Precompute vertical-pair angle exactly as in CreateAngleImage()
        angle_img = self._create_angle_image(depth)

        from collections import deque

        def satisfies(current: Tuple[int, int], neigh: Tuple[int, int]) -> bool:
            rr, cc = current
            rn, cn = neigh
            # depth gate
            if depth[rn, cn] <= 0.001:
                return False
            # angle gate: like SimpleDiff(angle) < threshold
            if rn != rr:
                # angle between rn and rr -> use greater row index
                r_use = rn if rn > rr else rr
                if r_use <= 0 or r_use >= R:
                    return False
                if not (np.isfinite(angle_img[r_use, cn]) and angle_img[r_use, cn] < radians_threshold):
                    return False
            else:
                # horizontal neighbor: keep it (matches the C++ "diff_at" using angle precomputed
                # where horizontal steps are allowed without extra angle rejection)
                pass
            return True

        for r in range(R):
            for c in range(C):
                if not mask[r, c] or labels[r, c] != 0:
                    continue
                # seed BFS
                q = deque([(r, c)])
                labels[r, c] = next_label
                size = 0
                while q:
                    rr, cc = q.popleft()
                    size += 1

                    neighs = [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]
                    if wrap_cols:
                        neighs = [(a, (b + C) % C) if (a, b) == (rr, cc - 1) or (a, b) == (rr, cc + 1) else (a, b)
                                  for (a, b) in neighs]

                    for rn, cn in neighs:
                        if rn < 0 or rn >= R or cn < 0 or cn >= C:
                            continue
                        if labels[rn, cn] != 0 or not mask[rn, cn]:
                            continue
                        if not satisfies((rr, cc), (rn, cn)):
                            continue
                        labels[rn, cn] = next_label
                        q.append((rn, cn))

                sizes[next_label] = size
                next_label += 1

        return labels, sizes, (next_label - 1)
    
    def _clusters_to_obstacles(self,
                               labels: np.ndarray,
                               sizes: Dict[int, int],
                               nlabels: int,
                               ri_idx: np.ndarray,
                               pts: np.ndarray) -> List[Obstacle]:
        obstacles: List[Obstacle] = []
        H, W = labels.shape
        for lab in range(1, nlabels + 1):
            sz = sizes.get(lab, 0)
            if sz < self.min_cluster_size or sz > self.max_cluster_size:
                continue

            # collect all original point indices belonging to this label
            rows, cols = np.where(labels == lab)
            if rows.size == 0:
                continue
            pix_ids = ri_idx[rows, cols]
            pix_ids = pix_ids[pix_ids >= 0]
            if pix_ids.size == 0:
                continue

            xyz = pts[pix_ids, :3]
            xyz_min = xyz.min(axis=0)
            xyz_max = xyz.max(axis=0)

            center = (xyz_min + xyz_max) * 0.5
            extent = (xyz_max - xyz_min) * 0.5

            obstacles.append(Obstacle(center=tuple(map(float, center)),
                                      extent=tuple(map(float, extent)),
                                      yaw=0.0,
                                      num_points=int(xyz.shape[0]),
                                      cell_indices=[(int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist())]))
        return obstacles

    # -------------------- Placeholders for downstream steps --------------------

    def detect_faults(self, mission_detections, safety_obstacles) -> bool:
        """Placeholder: compare mission vs safety layer outputs."""
        return False

    def assess_collision_risk(self, safety_obstacles) -> bool:
        """Placeholder: Perception Simplex existence-region overlap."""
        return False

    def override_control(self):
        """Placeholder: return an emergency VehicleControl (to be provided by caller/carla)."""
        return None

    def limit_velocity(self, control_mission):
        """Placeholder: clamp velocity if needed."""
        return control_mission