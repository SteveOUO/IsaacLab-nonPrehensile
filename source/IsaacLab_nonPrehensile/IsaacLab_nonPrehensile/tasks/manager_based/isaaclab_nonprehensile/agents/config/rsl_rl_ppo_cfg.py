# from isaaclab.utils import configclass

# from isaaclab_rl.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoAlgorithmCfg,
# )


# @configclass
# class PointNetActorCriticCfg:
#     """Config for PointNet-based Actor-Critic used in non-prehensile task."""

#     class_name: str = "ActorCriticPointNet"
#     # class_name: str = "ActorCritic"
#     pointnet_point_dim: int = 3
#     pointnet_num_points: int = 512
#     fuser_hidden_dims: list[int] = [256, 512, 256]
#     actor_hidden_dims: list[int] = [128, 64]
#     critic_hidden_dims: list[int] = [128, 64]
#     pointnet_output_dim: int = 128
#     activation: str = "elu"
#     init_noise_std: float = 1.0
#     noise_std_type: str = "scalar"


# @configclass
# class NonPrehensilePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     """RSL-RL PPO configuration for the non-prehensile pushing task."""

#     # Training parameters
#     num_steps_per_env = 8
#     max_iterations = 1000000
#     save_interval = 500

#     # Logging / experiment identifiers
#     experiment_name = "franka_nonprehensile"

#     # Observation normalization
#     empirical_normalization = False

#     # Policy network
#     policy = PointNetActorCriticCfg()

#     # PPO algorithm hyper-parameters
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=0.5,         
#         use_clipped_value_loss=True,
#         clip_param=0.3,                
#         entropy_coef=0.006,               
#         num_learning_epochs=8,      
#         num_mini_batches=8,            
#         learning_rate=5.0e-5,        
#         schedule="adaptive",  
#         gamma=0.99,               
#         lam=0.95,                 
#         desired_kl=0.016, 
#         max_grad_norm=1.0,
#     )


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from dataclasses import field


@configclass
class ICPActorCriticCfg:
    """Config for ICP-based Actor-Critic used in manipulation tasks."""

    class_name: str = "ActorCriticICP"
    
    # ICP pretrained weights path
    icp_weights_path: str | None = './ckpts/512-32-balanced-SAM-wd-5e-05-920'
    # icp_weights_path: str | None = None
    freeze_icp: bool = True     # Whether to freeze ICP parameters
    
    icp_point_dim: int = 3  # Only xyz coordinates
    icp_num_points: int = 512  # Number of points in point cloud
    
    # Network architecture
    fuser_hidden_dims: list[int] = [512, 256, 128]  # Feature fusion MLP
    actor_hidden_dims: list[int] = field(default_factory=lambda: [64])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [64])
    
    # Activation and noise configuration
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@configclass
class NonPrehensilePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the non-prehensile pushing task."""

    # Training parameters
    num_steps_per_env = 8
    max_iterations = 1000000
    save_interval = 500

    # Logging / experiment identifiers
    experiment_name = "franka_nonprehensile"

    # Observation normalization
    empirical_normalization = False

    # Policy network
    policy = ICPActorCriticCfg()

    # PPO algorithm hyper-parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,         
        use_clipped_value_loss=True,
        clip_param=0.3,                
        entropy_coef=0.006,               
        num_learning_epochs=8,      
        num_mini_batches=8,            
        learning_rate=5.0e-5,        
        schedule="adaptive",  
        gamma=0.99,               
        lam=0.95,                 
        desired_kl=0.016, 
        max_grad_norm=1.0,
    )