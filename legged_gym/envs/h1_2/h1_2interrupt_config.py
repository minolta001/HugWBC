from legged_gym.envs.h1_2.h1_2_config import H12Cfg, H12CfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

PROPRIOCEPTION_DIM = 87 # From H12Cfg
INTERRUPT_IN_CMD = True
NOISE_IN_PRIVILEGE = False
EXECUTE_IN_PRIVILEGE = False
CMD_DIM = 3 + 4 + 1 + 2 + INTERRUPT_IN_CMD # From H12Cfg, plus interrupt flag
TERRAIN_DIM = 221 # From H12Cfg
PRIVILEGED_DIM = 3 + 1 + 2 + 1 + 6 + 11 # From H12Cfg, can be extended  # 
CLOCK_INPUT = 2 # From H12Cfg

DISTURB_DIM = 14 # H1 V2 has 7 joints per arm

class H12InterruptCfg( H12Cfg ):
    class env( H12Cfg.env ):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT

    class rewards ( H12Cfg.rewards ):
        reward_curriculum_list = ['action_rate_upper', 'action_rate_lower',
                            'feet_stumble',
                            'joint_power_distribution', 'feet_contact_forces',
                            'dof_acc', 'torques',  
                            'base_height', 'collision', 'stand_still',
                            'lin_vel_z', 'base_height_min', 'dof_vel_limits', 
                            'ang_vel_xy', 
                            # 'dof_pos_limits',
                            # Deviation
                            'shoulder_yaw_deviation', 'shoulder_roll_deviation', 
                            'shoulder_pitch_deviation', 'elbow_deviation',
                            # 'hip_yaw_deviation', 'hip_roll_deviation',
                            'torso_deviation',
                            # Mob
                            # 'tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel', 
                            # 'feet_clearance_cmd_linear',
                            # 'feet_clearance_cmd_polynomial', 
                            'hopping_symmetry',
                            'jump',
                            'orientation_control',
                            # 'waist_control',
                            # Task
                            # 'tracking_ang_vel'
                            # Standing
                            # 'standing',
                            'standing_air',
                            'standing_vel',
                            # 'standing_joint_deviation'
                            ]

        # You can override reward scales here if needed, for example:
        class scales( H12Cfg.rewards.scales ):
            action_rate = 0 # Disable the general action rate
            action_rate_lower = -0.01 # Penalize leg action rate
            action_rate_upper = -0.01 # Penalize arm action rate (when not disturbed)
            orientation_control = -10
            base_height = -40.0
            stand_still = -10.0
            standing = 2.0
            standing_air = -2


    class commands( H12Cfg.commands ):
        num_commands = CMD_DIM

    class disturb:
        max_curriculum = 1.0
        use_disturb = True
        disturb_dim = DISTURB_DIM
        disturb_scale = 2
        # NOTE: These scales and bounds need to be defined for all 14 arm joints
        noise_scale = [
            # Left Arm (7 joints)
            5.2, # Left Shoulder Pitch
            3.3, # Left Shoulder Roll
            5.5, # Left Shoulder Yaw
            3.7, # Left Elbow Pitch
            3.7, # Left Elbow Roll
            2.0, # Left Wrist Pitch
            2.0, # Left Wrist Yaw
            # Right Arm (7 joints)
            5.2, # Right Shoulder Pitch
            3.3, # Right Shoulder Roll
            5.5, # Right Shoulder Yaw
            3.7, # Right Elbow Pitch
            3.7, # Right Elbow Roll
            2.0, # Right Wrist Pitch
            2.0, # Right Wrist Yaw
        ]
        noise_lowerbound = [
            # Left Arm
            -2.6, -0.3, -1.2, -1.2, -1.2, -1.0, -1.0,
            # Right Arm
            -2.6, -3.0, -4.3, -1.2, -1.2, -1.0, -1.0,
        ]
        uniform_scale = 1
        uniform_noise = True
        noise_ratio = 1
        interrupt_action_buffer = None
        start_by_curriculum = True
        replace_action = True
        disturb_rad = 0.2
        disturb_rad_curriculum = True
        disturb_curriculum_method = 2

        noise_update_step = 30
        switch_prob = 0.005
        interrupt_in_cmd = INTERRUPT_IN_CMD
        stand_interrupt_only = False
        noise_curriculum_ratio = 0.5
        disturb_in_last_action = False
        obs_target_interrupt_in_privilege = NOISE_IN_PRIVILEGE
        obs_executed_actions_in_privilege = EXECUTE_IN_PRIVILEGE
        disturb_terminate_assets = []

    class curriculum_thresholds( H12Cfg.curriculum_thresholds):
        class disturb:
            tracking_lin_vel = 0.6

class H12InterruptCfgPPO( H12CfgPPO ):
    class runner( H12CfgPPO.runner ):
        experiment_name = "h1_2_interrupt"
        resume = False
        resume_path = None
        max_iterations = 40000
        save_interval = 2000

    class policy( H12CfgPPO.policy ):
        model_name = "MlpAdaptModel"
        class NetModel:
            class MlpAdaptModel( H12CfgPPO.policy.NetModel.MlpAdaptModel ):
                # Override dimensions to match the new config
                cmd_dim = CMD_DIM + CLOCK_INPUT
                privileged_dim = PRIVILEGED_DIM
                terrain_dim = TERRAIN_DIM

        critic_obs_dim = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM