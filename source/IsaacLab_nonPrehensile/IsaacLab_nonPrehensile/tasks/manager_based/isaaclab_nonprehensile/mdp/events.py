# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for non-prehensile manipulation environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform
import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_initial_object_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset object position using curriculum learning ranges.
    
    This function resets the object position within ranges that can be
    dynamically updated by the curriculum learning system through the command manager.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        asset_cfg: Asset configuration.
    """
    # Get the asset
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # Get curriculum ranges from command manager (preferred) or environment fallback
    stable_pose_term = env.command_manager.get_term("target_object_pose")
    xy_range = stable_pose_term.initial_position_range
    
    # Define base position (center of table in environment coordinates)
    base_x = 0.5  # center of table x-coordinate
    base_y = 0.0  # center of table y-coordinate
    base_z = 0.0  # fixed height on table
    
    # Sample random positions within curriculum ranges
    num_resets = len(env_ids)
    # Create poses: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
    poses = torch.zeros((num_resets, 13), device=env.device)
    
    # Per-env sampling from stable pose with random yaw offset
    from IsaacLab_nonPrehensile.tasks.manager_based.isaaclab_nonprehensile.env import get_cached_cloud
    assets_cfg = asset.cfg.spawn.assets_cfg
    scales = get_rigid_body_scale(env, SceneEntityCfg("object"), env_ids)
    
    for i, env_id in enumerate(env_ids):
        env_id_int = int(env_id.item())
        # XY around base with curriculum range (environment coordinates + origin)
        dx = sample_uniform(-xy_range, xy_range, (1,), device=env.device).squeeze(0)
        dy = sample_uniform(-2*xy_range, 2*xy_range, (1,), device=env.device).squeeze(0)
        # local (environment) coordinates before origin offset
        x_env = torch.as_tensor(base_x, device=env.device) + dx
        y_env = torch.as_tensor(base_y, device=env.device) + dy
        
        # Enforce reachable workspace (project into XY annulus) if configured
        if hasattr(stable_pose_term, "enforce_workspace") and bool(stable_pose_term.enforce_workspace):
            cx = torch.as_tensor(getattr(stable_pose_term, "workspace_center_x", 0.0), device=env.device)
            cy = torch.as_tensor(getattr(stable_pose_term, "workspace_center_y", 0.0), device=env.device)
            rmin = torch.as_tensor(getattr(stable_pose_term, "workspace_radius_min", 0.0), device=env.device)
            rmax = torch.as_tensor(getattr(stable_pose_term, "workspace_radius_max", float("inf")), device=env.device)
            ddx = x_env - cx
            ddy = y_env - cy
            rr = torch.sqrt(ddx * ddx + ddy * ddy).clamp_min(1e-9)
            rr_clamped = torch.clamp(rr, min=rmin, max=rmax)
            scale = rr_clamped / rr
            x_env = cx + ddx * scale
            y_env = cy + ddy * scale
        
        pos_x = x_env + env.scene.env_origins[env_id_int, 0]
        pos_y = y_env + env.scene.env_origins[env_id_int, 1]
        pos_z = torch.as_tensor(base_z, device=env.device)
        
        # Determine this env's asset and sample a stable pose (roll/pitch/yaw)
        asset_idx = env_id_int % len(assets_cfg)
        obj_path = assets_cfg[asset_idx].obj_path

        # Get the actual current scale from USD (dynamic scale system)
        scale_tensor = scales[i]
        scale = tuple(scale_tensor.cpu().numpy())

        object_cloud = get_cached_cloud(obj_path)
        sample_pose = object_cloud.sample_stable_pose_trimesh(scale=scale)
        _, quat = sample_pose  # (position, quaternion)
        
        # Add a random yaw offset to quaternion (rotate around Z-axis)
        yaw_offset = (torch.rand(1, device=env.device) * (2 * torch.pi) - torch.pi).squeeze(0)
        
        # Create rotation quaternion for yaw offset around Z-axis
        yaw_quat = torch.zeros(4, device=env.device)
        yaw_quat[0] = torch.cos(yaw_offset * 0.5)  # w
        yaw_quat[3] = torch.sin(yaw_offset * 0.5)  # z (around Z-axis)
        
        # Convert numpy quaternion to torch tensor
        quat_tensor = torch.as_tensor(quat, device=env.device, dtype=torch.float32)
        
        # Applies yaw rotation to the original orientation
        w1, x1, y1, z1 = yaw_quat[0], yaw_quat[1], yaw_quat[2], yaw_quat[3]
        w2, x2, y2, z2 = quat_tensor[0], quat_tensor[1], quat_tensor[2], quat_tensor[3]
        
        qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
        qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # x
        qy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # y
        qz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # z
        
        # Normalize quaternion after multiplication to ensure it's a unit quaternion
        quat_norm = torch.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if quat_norm > 1e-12:
            qw = qw / quat_norm
            qx = qx / quat_norm
            qy = qy / quat_norm
            qz = qz / quat_norm
        
        # Fill pose row
        poses[i, 0] = pos_x
        poses[i, 1] = pos_y
        poses[i, 2] = pos_z
        poses[i, 3] = qw
        poses[i, 4] = qx
        poses[i, 5] = qy
        poses[i, 6] = qz
        # velocities already zeros
    
    # Apply the new poses
    asset.write_root_state_to_sim(poses, env_ids)
    

def get_rigid_body_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    env_ids: torch.Tensor | list[int] | None = None,
):
    """Get rigid body scales with simple lazy per-env cache (env._scale_cache)."""
    # ensure cache dict
    if not hasattr(env, "_scale_cache") or env._scale_cache is None:
        env._scale_cache = {}

    # normalize requested ids (list[int])
    if isinstance(env_ids, torch.Tensor):
        requested = env_ids.tolist()
    else:
        requested = env_ids

    # find missing ids
    missing = [eid for eid in requested if eid not in env._scale_cache]
    if missing:
        # resolve prim paths once
        asset: RigidObject = env.scene[asset_cfg.name]
        prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
        if len(prim_paths) == 0:
            raise ValueError(f"Could not find prims with path: {asset.cfg.prim_path}")
        import isaacsim.core.utils.prims as prim_utils
        for eid in missing:
            prim = prim_utils.get_prim_at_path(prim_paths[eid])
            scale = prim.GetAttribute("xformOp:scale").Get()
            env._scale_cache[eid] = torch.tensor(scale, device=env.device, dtype=torch.float32)

    # assemble output in request order
    return torch.stack([env._scale_cache[eid] for eid in requested], dim=0)


def randomize_terrain_material(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
):
    """Randomize terrain material properties.
    
    This function randomizes the physics material properties of the terrain mesh.
    Since terrain is not a standard RigidObject, we need to directly access the
    terrain mesh prim and update its material properties.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to randomize (ignored for terrain as it's global).
        static_friction_range: Range for static friction coefficient.
        dynamic_friction_range: Range for dynamic friction coefficient.
        restitution_range: Range for restitution coefficient.
        num_buckets: Number of material buckets for randomization.
    """
    # Get terrain from scene
    terrain = env.scene["terrain"]
    
    # Generate material buckets
    static_friction_buckets = torch.linspace(
        static_friction_range[0], static_friction_range[1], num_buckets, device=env.device
    )
    dynamic_friction_buckets = torch.linspace(
        dynamic_friction_range[0], dynamic_friction_range[1], num_buckets, device=env.device
    )
    restitution_buckets = torch.linspace(
        restitution_range[0], restitution_range[1], num_buckets, device=env.device
    )
    
    # Ensure dynamic friction <= static friction
    dynamic_friction_buckets = torch.min(dynamic_friction_buckets, static_friction_buckets)
    
    # Randomly select material properties
    bucket_id = torch.randint(0, num_buckets, (1,), device=env.device)
    
    static_friction = static_friction_buckets[bucket_id]
    dynamic_friction = dynamic_friction_buckets[bucket_id]
    restitution = restitution_buckets[bucket_id]
    
    # Update terrain physics material
    # For terrain, we need to access the physics material prim directly
    import isaacsim.core.utils.prims as prim_utils
    from pxr import UsdPhysics
    
    # Get the terrain prim path
    # For plane terrain, the actual prim path is {cfg.prim_path}/terrain
    terrain_prim_path = terrain.cfg.prim_path + "/terrain"
    
    # Find the physics material prim
    physics_material_path = f"{terrain_prim_path}/physicsMaterial"
    physics_material_prim = prim_utils.get_prim_at_path(physics_material_path)

    # Create or get the physics material
    physics_material = UsdPhysics.MaterialAPI.Apply(physics_material_prim)
    
    # Set material properties
    physics_material.CreateStaticFrictionAttr().Set(static_friction.item())
    physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction.item())
    physics_material.CreateRestitutionAttr().Set(restitution.item())
    
    # Apply the material to the terrain mesh
    # For plane terrain, we need to find the actual collision prim (Plane type)
    mesh_prim_path = f"{terrain_prim_path}/mesh"
    mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
    
    # Find the collision prim (Plane type) for ground plane
    collision_prim = prim_utils.get_first_matching_child_prim(
        terrain_prim_path, 
        predicate=lambda x: prim_utils.get_prim_type_name(x) == "Plane"
    )
    
    # Use IsaacLab's bind_physics_material function
    import isaaclab.sim as sim_utils
    sim_utils.bind_physics_material(collision_prim.GetPrimPath(), physics_material_path)

