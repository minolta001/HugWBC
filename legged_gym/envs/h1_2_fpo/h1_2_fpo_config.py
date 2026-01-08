from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgFPO

PROPRIOCEPTION_DIM = 87 # 27 * 3 + 3 + 3
CMD_DIM = 3 + 4 + 1 + 2
TERRAIN_DIM = 221 


# NOTE: What is the dimensionality of privilige information?
PRIVILEGED_DIM = 3 + 1 + 2 + 1 + 6 + 13
CLOCK_INPUT = 2

NUM_ACTIONS = 27

class H12Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT
        num_privileged_obs = None
        num_actions = 27
        observe_body_height = True
        observe_body_pitch = True
        observe_waist_roll = True
        stack_history_obs = True
        include_history_steps = 1   # 5
        obs_interval = 1
        has_privileged_info = True
        observe_gait_commands = True
        observe_body_height = True
        observe_body_pitch = True
        observe_waist_roll = True

        num_envs = 1024
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.05] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint': 0.0,
            'left_hip_pitch_joint': -0.4,
            'left_hip_roll_joint': 0.02,
            'left_knee_joint': 0.8,         #0.8,

            'left_ankle_pitch_joint': -0.4,
            'left_ankle_roll_joint': 0.0,

            'right_hip_yaw_joint': 0.0,
            'right_hip_pitch_joint': -0.4,
            'right_hip_roll_joint': -0.02,
            'right_knee_joint': 0.8,        #0.8,
            'right_ankle_pitch_joint': -0.4,
            'right_ankle_roll_joint': 0.0,

            'torso_joint': 0.0,
            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_pitch_joint': 0.3,
            'left_elbow_roll_joint': 0.0,
            'left_elbow_joint': 0.0,

            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'left_wrist_roll_joint': 0.0,

            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_pitch_joint': 0.3,
            'right_elbow_roll_joint': 0.0,
            'right_elbow_joint': 0.0,

            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
            'right_wrist_roll_joint': 0.0
        }
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle_roll': 40,
                     'ankle_pitch': 40,
                     'torso': 300,
                     'shoulder_pitch': 40,
                     'shoulder_roll': 40,
                     'shoulder_yaw': 18,
                     "elbow_pitch": 18,
                     "elbow_roll": 18,
                     "elbow": 18,
                     "wrist_pitch": 10,
                     "wrist_yaw": 10,
                     "wrist_roll": 18

                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle_pitch': 2,
                     'ankle_roll': 2,
                     'torso': 6,
                     'shoulder_pitch': 2,
                     'shoulder_roll': 2,
                     'shoulder_yaw': 2,
                     "elbow_pitch": 2,
                     "elbow_roll": 2,
                     "elbow": 2,
                     "wrist_pitch": 2,
                     "wrist_yaw": 2,
                     "wrist_roll": 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        torque_limits = {
                     'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle_roll': 40,
                     'ankle_pitch': 40,
                     'torso': 200,
                     'shoulder_pitch': 40,
                     'shoulder_roll': 40,
                     'shoulder_yaw': 18,
                     'elbow_pitch': 18,
                     'elbow_roll': 18,
                     'elbow': 18,
                     'wrist_pitch': 10,
                     'wrist_yaw': 10,
                     'wrist_roll': 18,
                     }
        # action scale: target angle = actionScale * action + defaultAngle
        #action_scale = 0.25
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'  
        measure_foot_scan = True
        terrain_proportions = [1.0]
        border_size = 10 # [m]
        max_init_terrain_level = 3  
        print_all_levels = True
        print_all_types = False

        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        curriculum = True

        measured_points_x = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
        measured_points_y = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6]
        height_radius = 0.07
        scan_radius = 0.05
        # trimesh only:
        slope_treshold = 0.8  # slopes above this threshold will be corrected to vertical surfaces
        global_difficulty = [1]
        flat_std_limit = 0.04 # m
        init_difficulty = 0
        class level_property:
            random_max_height = 0.02
            max_stairs_height = 0.30
            max_obstacles_height = 0.11  # (-h~h)
            max_large_step_height = 0.25
            max_slope_angle = 0.6 # np.tan(45/180*np.pi)
        
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = CMD_DIM
        resampling_time = 10. 
        heading_command = True 
        selected_terrain_type = None
        terrain_kwargs = None
        min_vel = 0.15 
        num_bins_vel_x = 12
        num_bins_vel_yaw = 10 
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.6, 0.6] # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            gait_frequency = [1.5, 3.5]

            foot_swing_height = [0.1, 0.35]

            #body_height = [-0.3, 0.0]
            body_height = [-0.4, 0.4]

            #body_pitch = [0.0, 0.4]
            body_pitch = [0.0, 0.8]

            waist_roll = [-1.0, 1.0]
            heading = [-3.14, 3.14]
            abs_vel = [0.35, 0.6]

            limit_vel_x = [-0.6, 2.0]
            limit_vel_yaw = [-1.0, 1.0]

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        max_collision_force = 2
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        termination_height = True
        upper_curriculum = False
        max_contact_force = 500
        base_height_target = 0.98
        penalize_curriculum = True
        curriculum_init = 0.2
        kappa_gait_probs = 0.05
        gait_force_sigma = 50  
        gait_vel_sigma = 5  
        penalize_curriculum_sigma = 0.95
        reward_curriculum_list = ['action_rate', 'feet_stumble',
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
                                  'standing_vel',
                                  'standing_air',
                                  # 'standing_joint_deviation'
                                  ]
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 3.0
            dof_pos_limits = -10.0
            lin_vel_z = -0.1
            ang_vel_xy = -0.5
            hip_deviation = -2 
            shoulder_deviation = -1
            hip_yaw_deviation = -0
            hip_roll_deviation = -0
            shoulder_yaw_deviation = -0
            shoulder_roll_deviation = -0
            shoulder_pitch_deviation = -0
            elbow_deviation = -0
            joint_power_distribution = -0.5
            no_fly = 0.25
            termination = -200

            dof_vel_limits = -2
            action_rate = -0.01
            feet_contact_forces = -0.2  
            feet_slip = -0.2
            feet_stumble = -0.2
            dof_acc = -2.5e-7 
            torques = -5e-6
            orientation_control = -20.0
            waist_control = -2.0
            base_height = -40
            collision = -0.0
            stand_still = -5
            tracking_contacts_shaped_force = 2
            tracking_contacts_shaped_vel = 4
            feet_clearance_cmd_linear = -30 
            feet_clearance_cmd_polynomial = -0
            hopping_symmetry = -5
            standing = 1.0
            standing_air = -1.0
            standing_joint_deviation = -2.0
            alive = 0.2

    class curriculum_thresholds:
        class upper_motion:
            tracking_lin_vel = 0.8  # closer to 1 is tighter
        class terrains_level:
            tracking_lin_vel = 0.8  # closer to 1 is tighter
            dof_vel_limits = 0.1 
        class terrains_type:  
            tracking_lin_vel = 0.8  # closer to 1 is tighter
        class commands:
            tracking_lin_vel = 0.8
            tracking_ang_vel = 0.4  # too tight 0.7
            tracking_contacts_shaped_force = 0.8
            tracking_contacts_shaped_vel = 0.8

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/urdf/h1_2_handless.urdf'
        name = "h1_2"
        base_name = "pelvis" 
        foot_name = "ankle_roll"
        auxiliary_foot_link = ["left_foot", "right_foot"]
        penalize_contacts_on = ["elbow", "torso", "hip", "knee"]    # remove "knee"
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        override_inertia = False
        override_com = False
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True

        friction_range = [0.1, 2.75]  # [0.5, 1.25]
        #friction_range = [0.1, 1.25]

        push_interval_s = 5
        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]
        motor_strength_range = [0.8, 1.2]
        randomize_control_latency = True
        control_latency_range = [0.0, 0.02]  # s
        randomize_link_props = True
        inertia_ratio = [0.8, 1.2]
        mass_ratio = [0.8, 1.2]
        link_com_offset = 0.01
        randomize_base_mass = True

        added_mass_range = [-3, 9]  # [-1, 3]

        base_com_x_offset_range = [-0.03, 0.03]
        base_com_y_offset_range = [-0.03, 0.03]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        max_push_vel_xy = 0.6
        max_push_ang_vel_xy = 0.6
        push_f_v_scale = 15
        push_f_steps = 30
    
class H12CfgFPO( LeggedRobotCfgFPO ):
    class policy( LeggedRobotCfgFPO.policy ):
        init_noise_std = 1.0
        max_std = 1.2
        min_std = 0.1
        model_name = "DiffusionPolicy"
        class NetModel:
            class DiffusionPolicy:               
                in_dim = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT
                out_dim = NUM_ACTIONS

        #critic_hidden_dims = [512, 256, 128]
        critic_in_dim = PROPRIOCEPTION_DIM + CMD_DIM + PRIVILEGED_DIM +TERRAIN_DIM
        critic_out_dim = 1

        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_activation = None

    class runner( LeggedRobotCfgFPO.runner ):
        run_name = ''
        experiment_name = 'h12_teacher'
        resume = False
        resume_path = "/home/PJLAB/xueyufei/RL_Code/unitree_rl_gym/logs/h1_teacher/Jul24_20-35-38_/model_2100.pt"
        save_interval = 2000

    class algorithm( LeggedRobotCfgFPO.algorithm ):
        entropy_coef = 0.01
        use_wbc_sym_loss = True
        symmetry_loss_coef = 0.5
        sync_update = True
        num_training_steps = 40000

        # NOTE: change learning rate
        #learning_rate = 5.e-4 # Try a smaller value

    



  
