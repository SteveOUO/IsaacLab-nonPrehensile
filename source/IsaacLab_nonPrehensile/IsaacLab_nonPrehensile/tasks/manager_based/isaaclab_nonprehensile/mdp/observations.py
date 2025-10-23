# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat
from scipy.spatial.transform import Rotation as R
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.cloud import Cloud
import IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.mdp as mdp

# Lightweight profiling utilities for observation functions
import time
from functools import wraps
import torch

def _ensure_obs_timers(env: "ManagerBasedRLEnv") -> dict:
    if not hasattr(env, "_obs_timers"):
        env._obs_timers = {}
    return env._obs_timers

def profile_obs(fn):
    @wraps(fn)
    def wrapper(env, *args, **kwargs):
        timers = _ensure_obs_timers(env)
        name = fn.__name__
        t0 = time.perf_counter()
        result = fn(env, *args, **kwargs)
        dt = time.perf_counter() - t0
        entry = timers.get(name)
        if entry is None:
            timers[name] = {"time": dt, "count": 1}
        else:
            entry["time"] += dt
            entry["count"] += 1
        return result
    return wrapper

def print_obs_timers(env: "ManagerBasedRLEnv") -> None:
    timers = getattr(env, "_obs_timers", {})
    if not timers:
        print("[obs timers] no data collected yet")
        return
    print("[obs timers] summary:")
    for name, entry in timers.items():
        total = entry["time"]
        count = entry["count"]
        avg = total / count if count > 0 else 0.0
        print(f"  {name}: total={total:.6f}s count={count} avg={avg:.6f}s")

# Debug: print observation stats every N calls
DEBUG_OBS_EVERY = 10000000

_HAND_GOAL_MEAN = torch.tensor([0.5, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # z mean = 0.15
_HAND_GOAL_STD = torch.tensor([0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

def _dbg(env: "ManagerBasedRLEnv", name: str, tensor: torch.Tensor) -> torch.Tensor:
    cnt = getattr(env, "_dbg_obs_cnt", 0)
    if cnt % DEBUG_OBS_EVERY == 0:
        t = tensor
        if t.dim() >= 2:
            mins = t.min(dim=0).values
            maxs = t.max(dim=0).values
            print(f"[obs] {name}: min={mins.tolist()} max={maxs.tolist()} shape={tuple(t.shape)}")
        else:
            print(f"[obs] {name}: min={t.min().item():.3f}, max={t.max().item():.3f}, shape={tuple(t.shape)}")
    env._dbg_obs_cnt = cnt + 1
    return tensor


def _dbg_cloud(env: "ManagerBasedRLEnv", name: str, cloud_env: torch.Tensor) -> None:
    cnt = getattr(env, "_dbg_cloud_cnt", 0)
    if cnt % DEBUG_OBS_EVERY == 0:
        # cloud_env shape: (num_envs, num_points, 3)
        x = cloud_env[..., 0]
        y = cloud_env[..., 1]
        z = cloud_env[..., 2]
        print(
            f"[obs] {name}: x[min={x.min().item():.3f}, max={x.max().item():.3f}] "
            f"y[min={y.min().item():.3f}, max={y.max().item():.3f}] "
            f"z[min={z.min().item():.3f}, max={z.max().item():.3f}], shape={tuple(cloud_env.shape)}"
        )
    env._dbg_cloud_cnt = cnt + 1


@profile_obs
def hand_state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Hand state observation (9D: position[3] + rotation_matrix[6]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 9) containing [x,y,z, r11,r12,r13, r21,r22,r23]
    """
    ee_frame = env.scene[ee_frame_cfg.name]
    
    # Get EE position and orientation in world coordinates
    ee_pos_l_w = ee_frame.data.target_pos_w[..., 1, :]  # (num_envs, 3)
    ee_pos_r_w = ee_frame.data.target_pos_w[..., 2, :]  # (num_envs, 3)
    ee_pos_w = (ee_pos_l_w + ee_pos_r_w) / 2
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)

    # Convert to environment coordinates
    ee_pos_env = ee_pos_w - env.scene.env_origins
    
    # Rotation as 6D
    rot_matrix = matrix_from_quat(ee_quat_w)  # (num_envs, 3, 3)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    hand_state_9d = torch.cat([ee_pos_env, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters (similar to corn config)
        device = hand_state_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        hand_state_9d = (hand_state_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "hand_state", hand_state_9d)


@profile_obs
def robot_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot state observation (14D: joint_positions[7] + joint_velocities[7]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 14)
    """
    asset = env.scene[asset_cfg.name]
    
    # Get joint positions and velocities
    joint_pos = asset.data.joint_pos[:, :7]
    joint_vel = asset.data.joint_vel[:, :7]
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Normalize joint positions using soft limits around default pos -> [-1,1]
        default_pos = asset.data.default_joint_pos[:, :7]
        soft_limits = asset.data.soft_joint_pos_limits[:, :7, :]
        mins = soft_limits[..., 0]
        maxs = soft_limits[..., 1]
        centers = default_pos
        half_ranges = torch.clamp((maxs - mins) * 0.5, min=1e-6)
        pos_norm = torch.clamp((joint_pos - centers) / half_ranges, -1.0, 1.0)

        # Normalize joint velocities to [0,1] using soft velocity limits
        vel_limits = torch.clamp(asset.data.soft_joint_vel_limits[:, :7], min=1e-6)
        vel_norm = torch.clamp(joint_vel / vel_limits, -1.0, 1.0)
        vel_norm = (vel_norm + 1.0) * 0.5
        
        return _dbg(env, "robot_state", torch.cat([pos_norm, vel_norm], dim=1))
    else:
        # Return raw joint states without normalization
        return _dbg(env, "robot_state", torch.cat([joint_pos, joint_vel], dim=1))


@profile_obs
def abs_pose_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
) -> torch.Tensor:
    """Absolute pose goal observation (9D: target position[3] + target rotation_matrix[6]).
    
    Returns:
        torch.Tensor: Shape (num_envs, 9)
    """
    from isaaclab.utils.math import quat_from_euler_xyz, matrix_from_quat
    
    target_goal = env.command_manager.get_command(command_name)
    target_pos = target_goal[:, :3]
    target_quat = target_goal[:, 3:7]  # quaternion [w, x, y, z]

    # Command now directly contains quaternions, convert to rotation matrix
    rot_matrix = matrix_from_quat(target_quat)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    goal_9d = torch.cat([target_pos, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use goal-specific normalization parameters (similar to corn config)
        device = goal_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        goal_9d = (goal_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "abs_goal", goal_9d)


def rel_pose_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "target_object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Relative pose goal observation (9D: goal relative to current object pose)."""
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_conjugate, matrix_from_quat

    target_goal = env.command_manager.get_command(command_name)  # (num_envs, 7)
    object_pose_7d = object_pose_in_env_frame(env, object_cfg)
    obj_pos_env = object_pose_7d[:, :3]
    obj_quat_w = object_pose_7d[:, 3:7]

    target_pos = target_goal[:, :3]
    target_quat = target_goal[:, 3:7]  # quaternion [w, x, y, z]

    rel_pos = target_pos - obj_pos_env
    current_quat_inv = quat_conjugate(obj_quat_w)
    rel_quat = quat_mul(target_quat, current_quat_inv)
    rot_matrix = matrix_from_quat(rel_quat)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine relative position and rotation
    rel_pose_9d = torch.cat([rel_pos, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters for relative pose goal
        device = rel_pose_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        rel_pose_9d = (rel_pose_9d - mean) / torch.clamp(std, min=1e-6)
    
    return rel_pose_9d


@profile_obs
def phys_params(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Physical parameters observation for non-prehensile manipulation.
    
    Returns 5D tensor: [object_mass, object_friction, hand_friction, ground_friction, object_restitution]

    Args:
        env: The RL environment
        object_cfg: Configuration for the object asset
        hand_cfg: Configuration for the robot hand
        
    Returns:
        torch.Tensor: Shape (num_envs, 5) containing [object_mass, object_friction, hand_friction, ground_friction, object_restitution]
    """
    object: RigidObject = env.scene[object_cfg.name]
    hand: RigidObject = env.scene[hand_cfg.name]
    
    # 1. Get object mass from IsaacLab's built-in interface
    object_mass = object.root_physx_view.get_masses().squeeze(-1)  # Shape: (num_envs,)

    # 2. Get object material properties from PhysX view
    # Material properties format: [static_friction, dynamic_friction, restitution]
    object_material_props = object.root_physx_view.get_material_properties()  # Shape: (num_envs, num_bodies, 3)
    object_friction = object_material_props[:, :, 0].mean(dim=1)   # (num_envs,) - object static friction
    object_restitution = object_material_props[:, :, 2].mean(dim=1)  # (num_envs,) - object restitution

    # 3. Get hand friction from robot's physics properties
    hand_material_props = hand.root_physx_view.get_material_properties()    # Shape: (num_envs, num_bodies, 3)
    hand_friction = hand_material_props[:, -1, 0]      # Use static friction for hand (last body)

    # 4. Get ground/terrain friction - read actual randomized values from USD prim
    # Read the actual physics material values that were set by randomization
    terrain = env.scene["terrain"]
    
    # Get the actual terrain prim path (same as in randomization)
    terrain_prim_path = terrain.cfg.prim_path + "/terrain"
    physics_material_path = f"{terrain_prim_path}/physicsMaterial"
    
    import isaacsim.core.utils.prims as prim_utils
    from pxr import UsdPhysics
    
    # Read the actual physics material values
    physics_material_prim = prim_utils.get_prim_at_path(physics_material_path)
    physics_material = UsdPhysics.MaterialAPI(physics_material_prim)
    static_friction_attr = physics_material.GetStaticFrictionAttr()
    ground_friction_value = static_friction_attr.Get()
    
    ground_friction = torch.full_like(object_mass, ground_friction_value)
    
    # Stack into observation tensor: [object_mass, object_friction, hand_friction, ground_friction, object_restitution]
    phys_params_tensor = torch.stack([
        object_mass.to(device=object.data.root_pos_w.device),                      # (num_envs,) - object mass [0.1, 0.5]
        object_friction.to(device=object.data.root_pos_w.device),                  # (num_envs,) - object static friction [0.7, 1.0]
        hand_friction.to(device=object.data.root_pos_w.device),                    # (num_envs,) - hand friction coefficient [1.0, 1.5]
        ground_friction.to(device=object.data.root_pos_w.device),                  # (num_envs,) - ground friction coefficient [0.3, 0.8]
        object_restitution.to(device=object.data.root_pos_w.device)                # (num_envs,) - object restitution coefficient [0.1, 0.2]
    ], dim=1)  # (num_envs, 5)
    
    return _dbg(env, "phys_params", phys_params_tensor)


def object_pose_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The pose of the object in the environment coordinate frame.

    Returns:
        torch.Tensor: Shape (num_envs, 7) containing [x, y, z, qw, qx, qy, qz] in environment coordinates
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]  # (num_envs, 3)
    object_quat_w = object.data.root_quat_w       # (num_envs, 4)
    object_pos_env = object_pos_w - env.scene.env_origins  # (num_envs, 3)
    pose_7d = torch.cat([object_pos_env, object_quat_w], dim=1)  # (num_envs, 7)

    if env.cfg.visualize_current_object_pose:
        visualize_object_pose_in_env(env, pose_7d)

    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand-specific normalization parameters for object pose
        device = pose_7d.device
        # For 7D pose: position [x,y,z] + quaternion [qw,qx,qy,qz]
        # Use position normalization from hand_goal params
        pos_mean = _HAND_GOAL_MEAN[:3].to(device).view(1, 3)  # [x, y, z] mean
        pos_std = _HAND_GOAL_STD[:3].to(device).view(1, 3)    # [x, y, z] std
        # For quaternion, use simple normalization (quaternions are already normalized)
        quat_mean = torch.zeros(4, device=device).view(1, 4)  # [qw, qx, qy, qz] mean
        quat_std = torch.ones(4, device=device).view(1, 4)    # [qw, qx, qy, qz] std
        
        # Normalize position and quaternion separately
        pos_norm = (pose_7d[:, :3] - pos_mean) / torch.clamp(pos_std, min=1e-6)
        quat_norm = (pose_7d[:, 3:7] - quat_mean) / torch.clamp(quat_std, min=1e-6)
        
        pose_7d = torch.cat([pos_norm, quat_norm], dim=1)
        
    return pose_7d


def object_pose_9d_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The object's current pose in the environment frame as 9D [x,y,z, r11,r12,r13, r21,r22,r23]."""
    pose_7d = object_pose_in_env_frame(env, object_cfg)
    pos_env = pose_7d[:, :3]
    quat_wxyz = pose_7d[:, 3:7]

    # Convert to rotation matrix
    rot_matrix = matrix_from_quat(quat_wxyz)
    rot_6d = torch.cat([rot_matrix[:, 0, :], rot_matrix[:, 1, :]], dim=1)
    
    # Combine position and rotation
    object_pose_9d = torch.cat([pos_env, rot_6d], dim=1)
    
    # Check normalization setting from environment config
    normalize = getattr(env.cfg, 'normalize_observations', True)
    
    if normalize:
        # Use hand/goal-specific normalization parameters for object pose too
        device = object_pose_9d.device
        mean = _HAND_GOAL_MEAN.to(device).view(1, 9)
        std = _HAND_GOAL_STD.to(device).view(1, 9)
        
        # Z-score normalization: (x - mean) / std
        object_pose_9d = (object_pose_9d - mean) / torch.clamp(std, min=1e-6)
    
    return _dbg(env, "cur_pose", object_pose_9d)


def visualize_object_pose_in_env(
    env: ManagerBasedRLEnv,
    object_pose_7d: torch.Tensor,
    marker_scale: tuple = (0.08, 0.08, 0.08),
) -> None:
    """Visualize the object's current pose in environment coordinates.
    
    This function creates visualization markers to show the object's current pose
    in the environment coordinate frame, using the same approach as the target pose visualization.
    """
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.utils.math import quat_from_euler_xyz
    
    # Create visualization markers if they don't exist (similar to target pose visualization)
    if not hasattr(env, '_current_object_pose_visualizer'):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"  # Different path from target pose
        marker_cfg.markers["frame"].scale = marker_scale  # Make frames visible but distinct
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)

    # Extract position and quaternion (same as target pose visualization)
    local_positions = object_pose_7d[:, :3]  # (num_envs, 3)
    quaternions = object_pose_7d[:, 3:7]  # (num_envs, 4)

    # Convert local positions to world positions by adding environment origins (same as target)
    world_positions = local_positions + env.scene.env_origins
    
    # Visualize current pose frames using world positions (same method as target)
    env._current_object_pose_visualizer.visualize(translations=world_positions, orientations=quaternions)


def create_object_pose_visualizer(env: ManagerBasedRLEnv, marker_scale: tuple = (0.08, 0.08, 0.08)) -> None:
    """Initialize the object pose visualizer. Call this once during environment setup."""
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    
    if not hasattr(env, '_current_object_pose_visualizer'):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"
        marker_cfg.markers["frame"].scale = marker_scale
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)


def update_object_pose_visualization(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> None:
    """Update the object pose visualization. Call this during environment step/reset."""
    from isaaclab.utils.math import quat_from_euler_xyz
    
    if hasattr(env, '_current_object_pose_visualizer'):
        # Get object pose in environment frame
        object_pose_7d = object_pose_in_env_frame(env, object_cfg)
        
        # Extract position and euler angles
        local_positions = object_pose_7d[:, :3]
        world_positions = local_positions + env.scene.env_origins
        
        quaternions = object_pose_7d[:, 3:7]
        
        # Update visualization (same method as target pose)
        env._current_object_pose_visualizer.visualize(translations=world_positions, orientations=quaternions)


def object_pose_with_visualization(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object pose in environment frame and update visualization as a side effect.
    
    This function serves dual purpose:
    1. Returns the object pose for observations
    2. Triggers visualization update each time it's called
    
    Returns:
        torch.Tensor: Shape (num_envs, 6) containing [x, y, z, roll, pitch, yaw] in environment coordinates
    """
    # Initialize visualizer if not already done
    if not hasattr(env, '_current_object_pose_visualizer'):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG
        
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ObjectPose/current_pose"
        marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        
        env._current_object_pose_visualizer = VisualizationMarkers(marker_cfg)
    
    # Get object pose
    object_pose_7d = object_pose_in_env_frame(env, object_cfg)
    
    # Update visualization
    update_object_pose_visualization(env, object_cfg)
    
    return object_pose_7d


def visualize_object_pointcloud(
    env: ManagerBasedRLEnv,
    pointcloud_tensor: torch.Tensor,
    point_size: float = 0.005,
    color: tuple = (0.0, 1.0, 0.0),  # Green color for point cloud
) -> None:
    """Visualize the object's point cloud for debugging purposes.
    
    The point cloud is displayed in world coordinates, showing the actual 
    transformed points at the object's current position and orientation.
    
    Args:
        env: The RL environment
        pointcloud_tensor: Pre-computed point cloud tensor, shape (num_envs, num_points*3)
        point_size: Size of the visualization spheres
        color: RGB color tuple for the point cloud visualization
    """
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils
    
    # Create visualization markers if they don't exist
    if not hasattr(env, '_pointcloud_visualizer'):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PointCloud",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=point_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
            },
        )
        
        env._pointcloud_visualizer = VisualizationMarkers(marker_cfg)
    
    # Reshape flattened point cloud back to (num_envs, num_points, 3) for visualization
    num_envs = pointcloud_tensor.shape[0]
    points_per_env = pointcloud_tensor.shape[1] // 3
    pointcloud_reshaped = pointcloud_tensor.view(num_envs, points_per_env, 3)
    
    # For visualization, show points from the first environment only
    first_env_points = pointcloud_reshaped[1]  # Shape: (num_points, 3)
    
    # Create identity quaternions for all points (spheres don't need rotation)
    num_points = first_env_points.shape[0]
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_points).to(first_env_points.device)
    
    # Visualize the points in world coordinates
    env._pointcloud_visualizer.visualize(
        translations=first_env_points,
        orientations=orientations
    )


@profile_obs
def get_object_pointcloud(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object point cloud transformed to world coordinates.
    
    The point cloud is computed in world coordinates by transforming the object's
    canonical point cloud using the object's current pose (position + orientation).
    
    Args:
        env: The RL environment
        robot_cfg: Robot configuration (currently unused but kept for compatibility)
        object_cfg: Object configuration

    Returns:
        torch.Tensor: Point cloud in world coordinates, shape (num_envs, num_points*3)
                     Each row contains flattened [x1,y1,z1, x2,y2,z2, ...] coordinates for observation concatenation
    """
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get asset configuration - with random_choice=False, we cycle through assets deterministically
    assets_cfg = object.cfg.spawn.assets_cfg
    num_envs = object.data.root_pos_w.shape[0]
    
    # Optimized batch processing by asset type
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env import get_cached_cloud
    
    # Group environments by asset type for batch processing
    env_to_asset = {env_idx: env_idx % len(assets_cfg) for env_idx in range(num_envs)}
    asset_to_envs = {}
    for env_idx, asset_idx in env_to_asset.items():
        if asset_idx not in asset_to_envs:
            asset_to_envs[asset_idx] = []
        asset_to_envs[asset_idx].append(env_idx)
    
    out_tensor = None
    
    # Optionally compile the inner transform function for speed (PyTorch 2.x)
    compile_enabled = getattr(env.cfg, "use_torch_compile", False)

    # Process each asset type in batch
    device = object.data.root_pos_w.device
    num_assets = len(assets_cfg)
    for asset_idx in range(num_assets):
        # Build env indices for this asset by stepping
        env_indices_tensor = torch.arange(asset_idx, num_envs, num_assets, device=device, dtype=torch.long)
        if env_indices_tensor.numel() == 0:
            continue
        obj_path = assets_cfg[asset_idx].obj_path
        
        # Get cached cloud object for this asset
        object_cloud = get_cached_cloud(obj_path)
        
        # Get actual scales from USD objects for the environments using this asset
        # Read cached scales if available; else fallback to API
        if hasattr(env, "_object_scales"):
            scales = env._object_scales[env_indices_tensor]
        else:
            scales = mdp.get_rigid_body_scale(env, SceneEntityCfg("object"), env_indices_tensor.tolist())
        
        # Batch gather poses for all environments using this asset
        batch_pos_w = object.data.root_pos_w[env_indices_tensor, :3].contiguous()  # (batch,3)
        batch_quat_w = object.data.root_quat_w[env_indices_tensor].contiguous()    # (batch,4)
        
        # Batch transform point clouds for all environments using this asset
        if not compile_enabled:
            batch_transformed = object_cloud.get_pointcloud(
                translation=batch_pos_w, 
                rotation=batch_quat_w,
                scale=scales
            )
        else:
            # Prewarm point cache on this device to avoid CPU->GPU tensor construction inside compiled graph
            if object_cloud._points_torch.get(device) is None:
                object_cloud._points_torch[device] = torch.tensor(object_cloud.points, dtype=torch.float32, device=device)
            # Compile and cache per-Cloud callable to avoid passing function objects as dynamic inputs
            if not hasattr(object_cloud, "_compiled_get_pointcloud"):
                def _call(t, r, s, self_ref=object_cloud):
                    return self_ref.get_pointcloud(translation=t, rotation=r, scale=s)
                object_cloud._compiled_get_pointcloud = torch.compile(
                    _call,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
            batch_transformed = object_cloud._compiled_get_pointcloud(batch_pos_w, batch_quat_w, scales)

        # Allocate output tensor if not yet allocated
        if out_tensor is None:
            num_points = batch_transformed.shape[1]
            out_tensor = torch.empty((num_envs, num_points, 3), device=batch_transformed.device, dtype=batch_transformed.dtype)

        # Write this batch back using tensor indexing on (N, M, 3)
        out_tensor[env_indices_tensor] = batch_transformed
    
    # Result tensor: flatten to (N, M*3)
    all_pointclouds = out_tensor.view(num_envs, -1)
    
    # Optional visualization for debugging
    if env.cfg.visualize_object_pointcloud:
        # Visualization expects fp32; cast temporarily to float32
        visualize_object_pointcloud(env, all_pointclouds.float())
    
    return all_pointclouds

@profile_obs
def get_object_pointcloud_in_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Get object point cloud in environment frame with optional normalization."""
    pointcloud_w = get_object_pointcloud(env, object_cfg)
    num_envs, flat_dim = pointcloud_w.shape
    num_points = flat_dim // 3
    pointcloud_w_reshaped = pointcloud_w.view(num_envs, num_points, 3)
    pointcloud_env = pointcloud_w_reshaped - env.scene.env_origins.unsqueeze(1)

    pointcloud_env_flat = pointcloud_env.reshape(num_envs, num_points * 3)
    return pointcloud_env_flat