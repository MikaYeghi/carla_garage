# Re-executing the SafetyLayer implementation cell (previous session reset).

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import numpy as np

from srunner.scenariomanager.timer import GameTime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_lidar_and_boxes(points: np.ndarray, obstacles: list):
    """Visualize LiDAR points and obstacle bounding boxes in 3D."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Downsample to avoid heavy rendering
    if len(points) > 50000:
        idx = np.random.choice(len(points), 50000, replace=False)
        pts = points[idx]
    else:
        pts = points

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='k', s=0.1, alpha=0.3)

    for ob in obstacles:
        cx, cy, cz = ob.center
        ex, ey, ez = ob.extent
        # 8 corners of the AABB
        corners = np.array([
            [cx-ex, cy-ey, cz-ez],
            [cx+ex, cy-ey, cz-ez],
            [cx+ex, cy+ey, cz-ez],
            [cx-ex, cy+ey, cz-ez],
            [cx-ex, cy-ey, cz+ez],
            [cx+ex, cy-ey, cz+ez],
            [cx+ex, cy+ey, cz+ez],
            [cx-ex, cy+ey, cz+ez],
        ])
        # faces
        faces = [
            [corners[j] for j in [0,1,2,3]],
            [corners[j] for j in [4,5,6,7]],
            [corners[j] for j in [0,1,5,4]],
            [corners[j] for j in [2,3,7,6]],
            [corners[j] for j in [1,2,6,5]],
            [corners[j] for j in [4,7,3,0]],
        ]
        box = Poly3DCollection(faces, alpha=0.2, facecolor='red')
        ax.add_collection3d(box)
        ax.text(cx, cy, cz+0.5, f"{ob.num_points}", color='r', fontsize=6)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('LiDAR Point Cloud with Detected Obstacles')
    ax.set_box_aspect([1,1,0.5])
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_ground_segmentation(pts: np.ndarray,
                             ri_idx: np.ndarray,
                             nonground_mask: np.ndarray):
    """
    Visualize ground vs non-ground points in 3D.
    
    Args:
        pts: (N, 3) or (N, 4) numpy array of lidar points [x, y, z, (intensity)].
        ri_idx: (H, W) array of indices from range image (as returned by _range_image()).
        nonground_mask: (H, W) boolean mask where True = non-ground (obstacle).
    """
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    H, W = ri_idx.shape
    assert nonground_mask.shape == ri_idx.shape, "nonground_mask and ri_idx must have same shape."

    # Mask out invalid pixels (ri_idx == -1)
    valid_mask = ri_idx >= 0
    nonground_mask = nonground_mask & valid_mask
    ground_mask = (~nonground_mask) & valid_mask

    # Gather ground and non-ground point indices
    ground_indices = ri_idx[ground_mask].ravel()
    nonground_indices = ri_idx[nonground_mask].ravel()

    # Extract points
    pts_ground = pts[ground_indices]
    pts_nonground = pts[nonground_indices]

    # Plot in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=40)

    ax.scatter(pts_ground[:, 0], pts_ground[:, 1], pts_ground[:, 2],
               c='green', s=1, label='Ground')
    ax.scatter(pts_nonground[:, 0], pts_nonground[:, 1], pts_nonground[:, 2],
               c='red', s=1, label='Non-ground')

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("LiDAR Ground Segmentation")
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


@dataclass
class Obstacle:
    center: Tuple[float, float, float]
    extent: Tuple[float, float, float]
    yaw: float
    num_points: int
    cell_indices: List[Tuple[int, int]]

def visualize_angle_image(angle_image: np.ndarray, max_angle_deg: float = 30.0) -> None:
    """
    Visualize the angle image (vertical steepness) used in ground removal.
    Each pixel shows the inclination angle between adjacent LiDAR rings.

    Args:
        angle_image: 2D numpy array (R x C), in radians.
        max_angle_deg: upper bound for color scaling to avoid outlier saturation.
    """
    angle_deg = np.degrees(angle_image)
    angle_deg = np.clip(angle_deg, 0, max_angle_deg)

    plt.figure(figsize=(10, 5))
    plt.imshow(angle_deg, cmap="turbo", origin="upper")
    plt.colorbar(label="Inclination angle (degrees)")
    plt.title("LiDAR Vertical Angle Image (Steepness Map)")
    plt.xlabel("Azimuth columns")
    plt.ylabel("Vertical rings (0 = top, N = bottom)")
    plt.tight_layout()
    plt.show()


class SafetyLayer:
    def __init__(self,
                 n_rings: int = 64,
                 n_cols: int = 235,
                 vert_fov_up_deg: float = 10.0,
                 vert_fov_down_deg: float = -30.0,
                 max_range_m: float = 85.0,
                 alpha_thresh_deg: float = 5.0,
                 min_cluster_size: int = 20,
                 max_cluster_size: int = 7500) -> None:
        self.n_rings = int(n_rings)
        self.n_cols = int(n_cols)
        self.vert_fov_up = math.radians(vert_fov_up_deg)
        self.vert_fov_down = math.radians(vert_fov_down_deg)
        self.max_range = float(max_range_m)
        self.alpha_thresh = math.radians(alpha_thresh_deg)
        self.min_cluster_size = int(min_cluster_size)
        self.max_cluster_size = int(max_cluster_size)

        # self._row_angles = np.linspace(self.vert_fov_down, self.vert_fov_up, self.n_rings, dtype=np.float32)
        self._row_angles = np.linspace(self.vert_fov_up, self.vert_fov_down, self.n_rings, dtype=np.float32)

    def detect_obstacles(self, lidar_data: Tuple[int | float, np.ndarray]) -> List[Obstacle]:
        frame_id, pts = self._parse_lidar(lidar_data)
        ri, ri_idx = self._range_image(pts)
        nonground_mask = self._ground_removal(ri)
        labels, num_labels, sizes = self._label_cc_4(nonground_mask)
        obstacles = self._build_boxes(labels, num_labels, sizes, ri_idx, pts)

        # === Visualization ===
        # if GameTime.get_time() >= 0:
        # plot_ground_segmentation(pts, ri_idx, nonground_mask)
        # plot_lidar_and_boxes(pts, obstacles)

        return obstacles
    
    def detect_faults(self, mission_detections, safety_obstacles):
        return False
    
    def assess_collision_risk(self, safety_obstacles):
        return False
    
    def limit_velocity(self, control_mission):
        return control_mission

    @staticmethod
    def _parse_lidar(lidar_data: Tuple[int | float, np.ndarray]) -> Tuple[int | float, np.ndarray]:
        if not isinstance(lidar_data, (tuple, list)) or len(lidar_data) != 2:
            raise ValueError("lidar_data must be a (frame_id_or_ts, points) tuple")
        frame_id, pts = lidar_data
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"points must have shape (N, >=3); got {pts.shape}")
        return frame_id, pts

    def _range_image(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x*x + y*y + z*z)
        keep = (r > 0.1) & (r < self.max_range) & np.isfinite(r)
        x, y, z, r = x[keep], y[keep], z[keep], r[keep]
        idx_kept = np.nonzero(keep)[0]

        # NOTE: this version of the code led to a range image shrinking to eithe right or left half.
        # az = np.arctan2(y, x)
        # az[az < 0.0] += 2.0 * math.pi
        # el = np.arctan2(z, np.sqrt(x*x + y*y))
        # row = np.floor((self.vert_fov_up - el) /
        #        (self.vert_fov_up - self.vert_fov_down) *
        #        (self.n_rings - 1)).astype(np.int32)
        # col = np.floor(az / (2.0 * math.pi) * self.n_cols).astype(np.int32)
        # np.clip(row, 0, self.n_rings - 1, out=row)
        # np.clip(col, 0, self.n_cols - 1, out=col)

        # NOTE: this version of the code seems to work, although it is monkey patching.
        az = np.arctan2(y, x)
        az[az < 0.0] += 2.0 * math.pi  # wrap to [0, 2π)
        az_min, az_max = az.min(), az.max()
        az_span = az_max - az_min
        if az_span < 1e-3:
            raise ValueError("LiDAR azimuth span too small!")
        col = np.floor((az - az_min) / az_span * (self.n_cols - 1)).astype(np.int32)

        el = np.arctan2(z, np.sqrt(x*x + y*y))
        row = np.floor((self.vert_fov_up - el) /
                    (self.vert_fov_up - self.vert_fov_down) *
                    (self.n_rings - 1)).astype(np.int32)
        np.clip(row, 0, self.n_rings - 1, out=row)
        np.clip(col, 0, self.n_cols - 1, out=col)

        ri = np.full((self.n_rings, self.n_cols), np.inf, dtype=np.float32)
        ri_idx = np.full((self.n_rings, self.n_cols), -1, dtype=np.int32)

        flat_idx = row * self.n_cols + col
        order = np.argsort(r)  # nearest first
        seen = np.zeros(self.n_rings * self.n_cols, dtype=bool)
        for k in order:
            f = int(flat_idx[k])
            if not seen[f]:
                seen[f] = True
                rr = int(row[k]); cc = int(col[k])
                ri[rr, cc] = r[k]
                ri_idx[rr, cc] = int(idx_kept[k])

        return ri, ri_idx
    
    def _ground_removal(self, ri: np.ndarray) -> np.ndarray:
        """
        DepthGroundRemover reimplementation:
        - Repairs depth (fills small zero/invalid holes vertically)
        - Builds angle image using row-angle sines/cosines
        - Optional Savitzky–Golay smoothing on the angle image (row-wise)
        - Bottom-seeded BFS per column with angle threshold
        - Vertical dilation with a two-tap kernel (top/bottom = 1)
        Returns: boolean mask of non-ground cells (valid & not ground)
        """
        depth = ri.astype(np.float32, copy=True)
        R, C = depth.shape

        # --- parameters (fallbacks if not set on the instance) ---
        row_angles = getattr(self, "_row_angles", None)
        if row_angles is None or len(row_angles) != R:
            raise ValueError("self._row_angles (radians) must be set and match ri rows.")
        row_angles = row_angles.astype(np.float32)

        alpha_thresh = float(getattr(self, "alpha_thresh", math.radians(5.0)))  # radians
        sg_window = int(getattr(self, "sg_window", 5))  # one of {5,7,9,11}; else bypass smoothing
        # RepairDepth params (match C++ call RepairDepth(image, 5, 1.0f))
        repair_step = 5
        repair_depth_thresh = 1.0

        # ---------- RepairDepth(no_ground_image, 5, 1.0f) ----------
        # Fill zero/near-zero depths using vertical neighbors if consistent
        repaired = depth.copy()
        # treat invalid/inf as zero so we can repair
        repaired[~np.isfinite(repaired)] = 0.0

        for c in range(C):
            col = repaired[:, c]
            for r in range(R):
                if col[r] < 0.001:
                    counter = 0
                    acc = 0.0
                    # i steps up, j steps down (1..step-1)
                    for i in range(1, repair_step):
                        ru = r - i
                        if ru < 0:
                            continue
                        prev = col[ru]
                        if prev <= 0.001:
                            continue
                        for j in range(1, repair_step):
                            rd = r + j
                            if rd >= R:
                                continue
                            nxt = col[rd]
                            if nxt > 0.001 and abs(prev - nxt) < repair_depth_thresh:
                                acc += (prev + nxt)
                                counter += 2
                    if counter > 0:
                        col[r] = acc / counter
            repaired[:, c] = col

        # ---------- CreateAngleImage(depth_image) ----------
        # angle_image[r,c] = atan2(|y_r,c - y_{r-1},c|, |x_r,c - x_{r-1},c|)
        # where x_r,c = depth[r,c] * cos(row_angle[r]), y_r,c = depth[r,c] * sin(row_angle[r])
        sin_x = np.sin(row_angles, dtype=np.float32)[:, None]  # (R,1)
        cos_x = np.cos(row_angles, dtype=np.float32)[:, None]  # (R,1)
        x_mat = repaired * cos_x
        y_mat = repaired * sin_x

        angle_image = np.zeros_like(repaired, dtype=np.float32)
        # rows 1..R-1: use previous row for vertical diff
        dx = np.abs(x_mat[1:, :] - x_mat[:-1, :])
        dy = np.abs(y_mat[1:, :] - y_mat[:-1, :])
        angle_image[1:, :] = np.arctan2(dy, dx).astype(np.float32)

        # ---------- Optional: Savitzky–Golay smoothing along rows ----------
        if sg_window in (5, 7, 9, 11):
            # 1D vertical convolution by the SG kernel; reflect padding like OpenCV BORDER_REFLECT101
            kernel = self._sg_kernel(sg_window).astype(np.float32).reshape(-1)
            half = sg_window // 2
            smoothed = np.zeros_like(angle_image, dtype=np.float32)
            # reflect index helper
            def _ref(i, n):
                # BORDER_REFLECT101 behavior
                if n == 1:  # degenerate
                    return 0
                if i < 0:
                    i = -i
                block = n * 2 - 2
                i_mod = i % block
                return i_mod if i_mod < n else block - i_mod
            for r in range(R):
                acc = np.zeros(C, dtype=np.float32)
                for k in range(sg_window):
                    rr = _ref(r + (k - half), R)
                    acc += kernel[k] * angle_image[rr, :]
                smoothed[r, :] = acc
            angle_used = smoothed
        else:
            angle_used = angle_image

        # ---------- ZeroOutGroundBFS ----------
        # Mimic LinearImageLabeler + SimpleDiff:
        # For each column: seed at bottom-most valid depth; BFS with predicate angle <= threshold.
        # We'll consider 4-neighborhood with preference to vertical growth; angle condition applied
        # on the higher of two rows (since angle is defined between r and r-1).
        valid = repaired > 0.001
        labels = np.zeros((R, C), dtype=np.uint16)  # 0 = unlabeled, >0 ground
        # A simple BFS per seed column (label id is always 1, like in C++)
        from collections import deque
        def angle_ok(r_from, r_to, c):
            # vertical step up/down: pick max(r_from, r_to) for angle index
            rr = r_to if r_to > r_from else r_from
            if rr <= 0 or rr >= R:
                return False
            a = angle_used[rr, c]
            return np.isfinite(a) and (a <= alpha_thresh)

        for c in range(C):
            # find bottom-most valid pixel
            r = R - 1
            while r > 0 and not valid[r, c]:
                r -= 1
            if r <= 0:
                continue
            if labels[r, c] > 0:
                continue
            # BFS
            q = deque()
            q.append((r, c))
            while q:
                rr, cc = q.popleft()
                if labels[rr, cc] > 0 or not valid[rr, cc]:
                    continue
                labels[rr, cc] = 1
                # neighbors (favor vertical; include slight lateral spread)
                for rn, cn in ((rr-1, cc), (rr, cc-1), (rr, cc+1), (rr+1, cc)):
                    if rn < 0 or rn >= R or cn < 0 or cn >= C:
                        continue
                    if labels[rn, cn] > 0 or not valid[rn, cn]:
                        continue
                    # require the vertical angle constraint if step changes row
                    if rn != rr:
                        if not angle_ok(rr, rn, cn):
                            continue
                    q.append((rn, cn))

        # ---------- Dilation with GetUniformKernel(kernel_size, CV_8U) ----------
        # C++: kernel has ones only at first and last positions (two-tap vertical).
        # Then cv::dilate(label_image, dilated, kernel).
        kernel_size = int(getattr(self, "sg_window", 5))  # they pass window_size to ZeroOutGroundBFS and then kernel_size = max(window_size-2, 3)
        kernel_size = max(kernel_size - 2, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        shift = kernel_size - 1  # distance between the two taps
        dilated = (labels > 0).copy()
        if shift > 0:
            # two-tap vertical dilation: OR with version shifted up and down by 'shift'
            up = np.zeros_like(dilated, dtype=bool)
            down = np.zeros_like(dilated, dtype=bool)
            up[:-shift, :] = dilated[shift:, :]
            down[shift:, :] = dilated[:-shift, :]
            dilated = dilated | up | down

        # ---------- Compose outputs ----------
        ground_mask = dilated
        nonground_mask = valid & (~ground_mask)

        return nonground_mask

    # ---- helpers mirroring the C++ kernels ----
    def _sg_kernel(self, window: int) -> np.ndarray:
        if window not in (5, 7, 9, 11):
            raise ValueError("Savitzky–Golay window must be one of {5,7,9,11}")
        if window == 5:
            k = np.array([-3.0, 12.0, 17.0, 12.0, -3.0], dtype=np.float32) / 35.0
        elif window == 7:
            k = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0], dtype=np.float32) / 21.0
        elif window == 9:
            k = np.array([-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0], dtype=np.float32) / 231.0
        else:  # 11
            k = np.array([-36.0, 9.0, 44.0, 69.0, 84.0, 89.0, 84.0, 69.0, 44.0, 9.0, -36.0], dtype=np.float32) / 429.0
        return k

    def _ground_removal_legacy(self, ri: np.ndarray) -> np.ndarray:
        R, C = ri.shape
        xi = self._row_angles
        sin_xi = np.sin(xi)[:, None]
        cos_xi = np.cos(xi)[:, None]

        depth = ri
        valid = np.isfinite(depth)

        alpha = np.zeros_like(depth, dtype=np.float32)
        for r in range(1, R):
            r0 = depth[r - 1, :]
            r1 = depth[r, :]
            vmask = valid[r - 1, :] & valid[r, :]
            if not np.any(vmask):
                continue
            dz = np.abs(r0 * sin_xi[r - 1, 0] - r1 * sin_xi[r, 0])
            dx = np.abs(r0 * cos_xi[r - 1, 0] - r1 * cos_xi[r, 0])
            a = np.zeros(C, dtype=np.float32)
            a[vmask] = np.arctan2(dz[vmask], dx[vmask]).astype(np.float32)
            alpha[r, :] = a

        delta_alpha = np.zeros_like(alpha)
        delta_alpha[1:, :] = np.abs(alpha[1:, :] - alpha[:-1, :])

        ground = np.zeros_like(depth, dtype=bool)
        start_row = 0
        ground[start_row, :] = valid[start_row, :]

        for r in range(start_row + 1, R):
            cond = ground[r - 1, :] & valid[r, :] & (delta_alpha[r, :] <= self.alpha_thresh)
            ground[r, cond] = True

        nonground = valid & (~ground)
        return nonground

    @staticmethod
    def _label_cc_4(mask: np.ndarray) -> Tuple[np.ndarray, int, Dict[int, int]]:
        R, C = mask.shape
        labels = np.zeros((R, C), dtype=np.int32)
        current = 0
        sizes: Dict[int, int] = {}

        for r in range(R):
            row_mask = mask[r]
            if not row_mask.any():
                continue
            for c in np.nonzero(row_mask)[0]:
                if labels[r, c] != 0:
                    continue
                current += 1
                stack = [(r, c)]
                labels[r, c] = current
                size = 0
                while stack:
                    rr, cc = stack.pop()
                    size += 1
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < R and 0 <= nc < C and mask[nr, nc] and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            stack.append((nr, nc))
                sizes[current] = size

        return labels, current, sizes

    def _build_boxes(self,
                     labels: np.ndarray,
                     num_labels: int,
                     sizes: Dict[int, int],
                     ri_idx: np.ndarray,
                     pts: np.ndarray) -> List[Obstacle]:
        obstacles: List[Obstacle] = []
        for lab in range(1, num_labels + 1):
            sz = sizes.get(lab, 0)
            if sz < self.min_cluster_size or sz > self.max_cluster_size:
                continue
            rc = np.argwhere(labels == lab)
            if rc.size == 0:
                continue
            idxs = []
            for rr, cc in rc:
                pid = ri_idx[rr, cc]
                if pid >= 0:
                    idxs.append(pid)
            if not idxs:
                continue
            idxs = np.array(idxs, dtype=np.int64)
            xyz = pts[idxs, :3]
            xyz_min = xyz.min(axis=0)
            xyz_max = xyz.max(axis=0)
            center = (xyz_min + xyz_max) * 0.5
            extent = (xyz_max - xyz_min) * 0.5
            obstacles.append(
                Obstacle(center=tuple(map(float, center)),
                         extent=tuple(map(float, extent)),
                         yaw=0.0,
                         num_points=int(xyz.shape[0]),
                         cell_indices=[(int(r), int(c)) for r, c in rc.tolist()])
            )
        return obstacles


# Synthetic demo to validate the pipeline
def _make_mock_cloud_two_boxes() -> Tuple[int, np.ndarray]:
    rng = np.random.default_rng(0)

    xs_ground = rng.uniform(-5.0, 5.0, size=6000)
    ys_ground = rng.uniform(-5.0, 5.0, size=6000)
    zs_ground = 0.02 * np.ones_like(xs_ground)
    ground = np.stack([xs_ground, ys_ground, zs_ground, np.ones_like(xs_ground)], axis=1).astype(np.float32)

    xa = rng.uniform(1.0, 4.0, size=600)
    ya = rng.uniform(1.0, 3.0, size=600)
    za = rng.uniform(0.5, 1.0, size=600)
    A = np.stack([xa, ya, za, np.ones_like(xa)], axis=1).astype(np.float32)

    xb = rng.uniform(-2.0, -1.0, size=800)
    yb = rng.uniform(-2.0, -1.0, size=800)
    zb = rng.uniform(0.5, 1.0, size=800)
    B = np.stack([xb, yb, zb, np.ones_like(xb)], axis=1).astype(np.float32)

    # pts = np.vstack([ground, A, B]).astype(np.float32)
    pts = np.vstack([A, B]).astype(np.float32)
    # pts = np.vstack([ground]).astype(np.float32)
    frame = (123, pts)
    return frame

def make_mock_lidar_ground():
    rings = 32
    cols = 512
    vert_fov_up, vert_fov_down = 10.0, -30.0
    row_angles = np.linspace(np.radians(vert_fov_down),
                             np.radians(vert_fov_up), rings)
    col_angles = np.linspace(0, 2*np.pi, cols, endpoint=False)
    R, C = np.meshgrid(np.ones_like(row_angles)*10.0, col_angles, indexing='ij')
    x = R*np.cos(col_angles)[None, :]
    y = R*np.sin(col_angles)[None, :]
    z = np.zeros_like(x)
    pts = np.stack([x, y, z, np.ones_like(x)], axis=-1).reshape(-1,4)
    return (0, pts.astype(np.float32))


# # Run the sanity check
# sl = SafetyLayer(n_rings=32, n_cols=720, vert_fov_up_deg=10.0, vert_fov_down_deg=-30.0,
#                  alpha_thresh_deg=5.0, min_cluster_size=20)
# # demo_frame = _make_mock_cloud_two_boxes()
# demo_frame = make_mock_lidar_ground()
# demo_obstacles = sl.detect_obstacles(demo_frame)
# print(len(demo_obstacles), [(o.center, o.extent, o.num_points) for o in demo_obstacles[:3]])
