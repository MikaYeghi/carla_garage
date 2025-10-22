
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


@dataclass
class LidarGeometry:
    n_rings: int
    n_cols: int
    vert_fov_up: float      # radians
    vert_fov_down: float    # radians
    max_range: float        # meters
    full_azimuth: bool = False   # if True, wrap columns (360°); else partial sweep (no wrap)


class SafetyLayer:
    def __init__(self,
                 n_rings: int = 64,
                 n_cols: int = 235,
                 vert_fov_up_deg: float = 10.0,
                 vert_fov_down_deg: float = -30.0,
                 max_range_m: float = 85.0,
                 alpha_thresh_deg: float = 0.1,
                 sg_window: int = 5,
                 eps_depth: float = 0.001,
                 full_azimuth: bool = False) -> None:
        """
        Args:
            n_rings: vertical channels (rows)
            n_cols:  columns to discretize current azimuth span (for partial sweeps)
            vert_fov_up_deg: top vertical FOV (degrees, positive)
            vert_fov_down_deg: bottom vertical FOV (degrees, negative)
            max_range_m: max usable range (meters)
            alpha_thresh_deg: angle threshold (degrees) for BFS linkage (ground growth)
            sg_window: Savitzky–Golay window {5,7,9,11}; others -> smoothing disabled
            eps_depth: minimum valid depth in meters
            full_azimuth: if True, assumes 360° and wraps columns in BFS; else no wrap
        """
        self.geom = LidarGeometry(
            n_rings=int(n_rings),
            n_cols=int(n_cols),
            vert_fov_up=math.radians(vert_fov_up_deg),
            vert_fov_down=math.radians(vert_fov_down_deg),
            max_range=float(max_range_m),
            full_azimuth=bool(full_azimuth),
        )
        self.alpha_thresh = math.radians(alpha_thresh_deg)
        self.sg_window = int(sg_window)
        self.eps_depth = float(eps_depth)

        # Row-angle vector: row 0 = TOP ring (sky), row R-1 = BOTTOM (ground)
        self._row_angles = np.linspace(self.geom.vert_fov_up,
                                       self.geom.vert_fov_down,
                                       self.geom.n_rings).astype(np.float32)

    # -------------------- Public API --------------------

    def detect_obstacles(self, lidar_data: Tuple[int | float, np.ndarray]) -> List:
        """Placeholder: obstacle extraction is not part of the provided C++ snippet."""
        _ts, pts = self._parse_lidar(lidar_data)
        ri, ri_idx, _ = self.range_image(pts)
        nonground_mask, angle_image = self.ground_removal(ri)
        self.visualize_ground_overlay(pts, ri_idx, nonground_mask)
        # NOTE: obstacle extraction will be added later; return empty list for now.
        return []

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

    # -------------------- Internals: C++ faithful parts --------------------

    @staticmethod
    def _parse_lidar(lidar_data: Tuple[int | float, np.ndarray]) -> Tuple[int | float, np.ndarray]:
        if not isinstance(lidar_data, (tuple, list)) or len(lidar_data) != 2:
            raise ValueError("lidar_data must be a (frame_id_or_ts, points) tuple")
        frame_id, pts = lidar_data
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"points must have shape (N, >=3); got {pts.shape}")
        return frame_id, pts

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

    @staticmethod
    def _create_angle_image(depth_image: np.ndarray, row_angles: np.ndarray) -> np.ndarray:
        """C++ CreateAngleImage: atan2(|Δy|, |Δx|), where x = d*cos(row_angle), y = d*sin(row_angle)."""
        R, C = depth_image.shape
        angle_image = np.zeros((R, C), dtype=np.float32)
        s = np.sin(row_angles, dtype=np.float32)[:, None]
        c = np.cos(row_angles, dtype=np.float32)[:, None]
        x_mat = depth_image * c
        y_mat = depth_image * s
        dx = np.abs(x_mat[1:, :] - x_mat[:-1, :])
        dy = np.abs(y_mat[1:, :] - y_mat[:-1, :])
        angle_image[1:, :] = np.arctan2(dy, dx).astype(np.float32)
        return angle_image

    @staticmethod
    def _sg_kernel(window: int) -> np.ndarray:
        if window == 5:
            k = np.array([-3.0, 12.0, 17.0, 12.0, -3.0], dtype=np.float32) / 35.0
        elif window == 7:
            k = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0], dtype=np.float32) / 21.0
        elif window == 9:
            k = np.array([-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0], dtype=np.float32) / 231.0
        elif window == 11:
            k = np.array([-36.0, 9.0, 44.0, 69.0, 84.0, 89.0, 84.0, 69.0, 44.0, 9.0, -36.0], dtype=np.float32) / 429.0
        else:
            raise ValueError("Savitzky–Golay window must be one of {5,7,9,11}")
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
                             depth_image: np.ndarray,
                             angle_image: np.ndarray,
                             threshold: float,
                             kernel_size: int) -> np.ndarray:
        """C++ ZeroOutGroundBFS: bottom-seeded BFS per column using angle threshold.

        Returns:
            labels: uint16 label image (ground labels > 0; unlabeled = 0)
        """
        eps = self.eps_depth
        R, C = depth_image.shape
        labels = np.zeros((R, C), dtype=np.uint16)
        valid = depth_image > eps

        from collections import deque

        def neighbors(r: int, c: int):
            # 4-neighborhood with optional azimuth wrap if full FOV
            cols = [c - 1, c + 1]
            rows = [r - 1, r + 1]
            for rr in rows:
                if 0 <= rr < R:
                    yield rr, c
            for cc in cols:
                if self.geom.full_azimuth:
                    cc = (cc + C) % C
                    yield r, cc
                else:
                    if 0 <= cc < C:
                        yield r, cc

        def angle_satisfies(r_from: int, r_to: int, c: int) -> bool:
            if r_from == r_to:
                return True  # lateral step: C++ uses SimpleDiff on angle; we follow their BFS that seeds vertically
            rr = max(r_from, r_to)
            if rr <= 0 or rr >= R:
                return False
            a = angle_image[rr, c]
            return np.isfinite(a) and (a < threshold)

        for c in range(C):
            r = R - 1
            while r > 0 and not valid[r, c]:
                r -= 1
            if r <= 0:
                continue
            if labels[r, c] > 0:
                continue

            q = deque()
            q.append((r, c))
            while q:
                rr, cc = q.popleft()
                if labels[rr, cc] > 0 or not valid[rr, cc]:
                    continue
                labels[rr, cc] = 1
                for rn, cn in neighbors(rr, cc):
                    if labels[rn, cn] > 0 or not valid[rn, cn]:
                        continue
                    # Require angle constraint on vertical motion (matches CreateAngleImage semantics)
                    if rn != rr and not angle_satisfies(rr, rn, cn):
                        continue
                    q.append((rn, cn))

        return labels

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