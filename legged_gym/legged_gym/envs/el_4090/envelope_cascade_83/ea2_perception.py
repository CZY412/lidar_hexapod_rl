"""EA2 187-channel perception for the cascade demo.

Replicates the *reduced raycast* path EA2 trains with
(``El4090EA2Cfg.lidar.use_reduced_raycast = True``): the 187 fixed
sensor-frame ray directions from ``selected_airy_channels.pt`` are launched
directly against a Warp mesh via
``LidarWarpKernels.draw_optimized_kernel_pointcloud`` — no ``LidarSensor``
instance is constructed, so there is no ray-regeneration hook to neutralise.

Timing matches EA2 exactly: the range image refreshes every
``round(1 / (dt * update_frequency_hz))`` control steps (10 Hz at 50 Hz
control, timer initialised to ``decimation - 1`` so the first step
refreshes), while the policy consumes the (stale-carrying) image at full
control rate.  An env reset marks its image rows empty (``= max_range``)
until the next global scan — EA2's empty-frame contract.

Observation assembly reuses EA2's own helpers
(``build_selected_range_image`` / ``assemble_observation``), so the
cascade consumes byte-identical observation semantics.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import warp as wp

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.el_4090.envelope_adaptive_2.airy_mount import (
    load_selected_channels,
    self_check_selected_channels,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.el_4090_ea2_env import (
    assemble_observation,
    refresh_range_image_from_scan,
)
from legged_gym.envs.el_4090.envelope_adaptive_2.range_image import (
    build_selected_range_image,
)


class Ea2Perception:
    """187-ray LiDAR perception driving the EA2 policy observation."""

    def __init__(
        self,
        num_envs: int,
        device,
        cfg,
        dt: float,
        terrain_vertices: Optional[np.ndarray] = None,
        terrain_triangles: Optional[np.ndarray] = None,
        border_size: float = 0.0,
    ) -> None:
        """``cfg`` is the cascade ``cfg.ea2`` namespace.

        Mesh source: either the Isaac terrain arrays (``mesh_type='trimesh'``
        path, ``border_size`` shifted like the inherited LiDAR code does) or
        explicitly injected arrays (tests).
        """
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._wp_device = str(self.device)  # warp rejects torch.device objects
        self.cfg = cfg
        self.dt = float(dt)
        self._decimation = max(1, int(round(1.0 / (self.dt * float(cfg.update_frequency_hz)))))
        self._timer = self._decimation - 1

        wp.init()
        if terrain_vertices is not None:
            vertices = torch.as_tensor(terrain_vertices, device=self.device, dtype=torch.float32).clone()
            vertices[:, 0] -= border_size
            vertices[:, 1] -= border_size
            triangles = np.asarray(terrain_triangles, dtype=np.int32).flatten()
        else:
            raise ValueError("Ea2Perception requires terrain_vertices/triangles")

        self._wp_mesh = wp.Mesh(
            points=wp.from_torch(vertices, dtype=wp.vec3),
            indices=wp.from_numpy(triangles, dtype=wp.int32, device=self._wp_device),
        )
        self.mesh_ids = wp.array([self._wp_mesh.id], dtype=wp.uint64, device=self._wp_device)

        channel_file = str(cfg.channel_file).format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        channels = load_selected_channels(channel_file)
        self_check_selected_channels(channels)
        self.range_max = float(channels["max_range"])
        ray_directions = channels["ray_directions"].to(self.device)
        ray_dir_tensor = ray_directions.unsqueeze(1).contiguous()  # (187, 1, 3)
        self._ray_vectors_wp = wp.from_torch(ray_dir_tensor, dtype=wp.vec3)

        from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul

        self._quat_apply = quat_apply
        self._quat_mul = quat_mul
        rpy = [float(v) for v in cfg.sensor_offset_rpy]
        offset_q = quat_from_euler_xyz(
            torch.tensor(rpy[0], device=self.device),
            torch.tensor(rpy[1], device=self.device),
            torch.tensor(rpy[2], device=self.device),
        )
        self._offset_quat = offset_q.view(1, 4).repeat(self.num_envs, 1)
        self._translation = (
            torch.tensor([float(v) for v in cfg.offset_pos], device=self.device)
            .view(1, 3)
            .repeat(self.num_envs, 1)
        )

        self.sensor_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.sensor_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self._pos_wp = wp.from_torch(self.sensor_pos.view(self.num_envs, 1, 3), dtype=wp.vec3)
        self._quat_wp = wp.from_torch(self.sensor_quat.view(self.num_envs, 1, 4), dtype=wp.quat)

        n_sel = int(ray_directions.shape[0])
        if n_sel != 187:
            raise ValueError(f"expected 187 selected channels, got {n_sel}")
        self._points = torch.zeros(
            (self.num_envs, 1, n_sel, 1, 3), dtype=torch.float32, device=self.device
        )
        self._dists = torch.zeros((self.num_envs, 1, n_sel, 1), dtype=torch.float32, device=self.device)
        self._points_wp = wp.from_torch(self._points, dtype=wp.vec3)
        self._dists_wp = wp.from_torch(self._dists, dtype=wp.float32)

        self._noise_ctx = SimpleNamespace(
            sensor_cfg=SimpleNamespace(
                enable_sensor_noise=bool(cfg.enable_sensor_noise),
                pixel_std_dev_multiplier=float(cfg.pixel_std_dev_multiplier),
                pixel_dropout_prob=float(cfg.pixel_dropout_prob),
            ),
            far_plane=float(cfg.far_plane),
        )
        from legged_gym.utils.LidarSensor import LidarSensor

        self._apply_noise = LidarSensor.apply_noise

        self.range_image = torch.full(
            (self.num_envs, n_sel), self.range_max, dtype=torch.float32, device=self.device
        )
        self.stale = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._debug_points: Optional[torch.Tensor] = None
        self._debug_dists: Optional[torch.Tensor] = None
        self.refresh_count = 0

    # ── pose ────────────────────────────────────────────────────────────

    def update_pose(self, base_pos: torch.Tensor, base_quat: torch.Tensor) -> None:
        """Place the sensor from the live robot pose.

        Default is the physical full ``base_quat`` (pitch/roll included);
        ``cfg.yaw_only`` reproduces EA2's yaw-only training mount for
        attributing OOD to body tilt.
        """
        if bool(self.cfg.yaw_only):
            q = yaw_quat(base_quat)
        else:
            q = base_quat
        self.sensor_quat.copy_(self._quat_mul(q, self._offset_quat))
        self.sensor_pos.copy_(base_pos + self._quat_apply(q, self._translation))

    # ── raycast ─────────────────────────────────────────────────────────

    def refresh(self) -> bool:
        """Advance the 10 Hz clock; on a tick raycast and rebuild the image."""
        self._timer += 1
        if self._timer % self._decimation != 0:
            return False

        from legged_gym.utils.LidarSensor.sensor_kernels.lidar_kernels_warp import (
            LidarWarpKernels,
        )

        wp.launch(
            kernel=LidarWarpKernels.draw_optimized_kernel_pointcloud,
            dim=(self.num_envs, 1, 187, 1),
            inputs=[
                self.mesh_ids,
                self._pos_wp,
                self._quat_wp,
                self._ray_vectors_wp,
                float(self.cfg.far_plane),
                self._points_wp,
                self._dists_wp,
                False,
            ],
            device=self._wp_device,
        )
        points = wp.to_torch(self._points_wp)
        dists = wp.to_torch(self._dists_wp)
        if getattr(self._noise_ctx.sensor_cfg, "enable_sensor_noise", False):
            points, dists = self._apply_noise(self._noise_ctx, points, dists)
        distsflat = dists.view(self.num_envs, 187)
        fresh = build_selected_range_image(distsflat, self.range_max, far_plane=float(self.cfg.far_plane))
        refresh_range_image_from_scan(self.range_image, fresh, self.stale)
        # Debug cloud: the kernel emits SENSOR-frame pixels (dist * ray_dir);
        # convert to BODY frame exactly like EA2's reduced-path debug does
        # (offset rotation + mount translation) so viewer-side transforms
        # (base_pos + base_quat) land the points in the world correctly.
        flat = points.reshape(-1, 3)
        points_body = self._quat_apply(
            self._offset_quat[0:1].expand(flat.shape[0], 4), flat
        ).reshape(self.num_envs, 187, 3) + self._translation.unsqueeze(1)
        self._debug_points = points_body
        self._debug_dists = distsflat.clone()
        self.refresh_count += 1
        return True

    def mark_stale(self, env_ids: torch.Tensor) -> None:
        """EA2 empty-frame contract: reset envs keep an empty image until
        the next global scan."""
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(self.device, dtype=torch.long)
        self.range_image[env_ids] = self.range_max
        self.stale[env_ids] = True

    # ── observation ─────────────────────────────────────────────────────

    @staticmethod
    def ego_motion_from(base_lin_vel: torch.Tensor, base_ang_vel: torch.Tensor) -> torch.Tensor:
        """Body-frame ``[vx, vy, wz]`` — the measured analogue of EA2's
        scripted ego motion."""
        return torch.cat((base_lin_vel[:, :2], base_ang_vel[:, 2:3]), dim=-1)

    def observe(self, ego_motion: torch.Tensor) -> torch.Tensor:
        """Assemble the 190-dim EA2 observation (187 range + 3 ego)."""
        return assemble_observation(
            self.range_image,
            ego_motion,
            max_range=self.range_max,
            ego_scales=tuple(float(v) for v in self.cfg.ego_scales),
        )

    def debug_cloud(self):
        """Latest noisy 187-point body-frame cloud ``(points, dists)`` or None."""
        if self._debug_points is None:
            return None
        return self._debug_points, self._debug_dists


def yaw_quat(base_quat: torch.Tensor) -> torch.Tensor:
    """Project a quaternion onto its yaw component (xyzw, same convention as
    ``EA2.yaw_quat_from_heading``)."""
    x, y, z, w = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return torch.stack(
        [torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)],
        dim=-1,
    )
