import os
import sys
sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, update_class_from_dict
from isaacgym import gymapi
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from isaacgym import gymapi
from utils.low_state_controller import LowStateCmdHandler
from utils.low_state_handler import JointID
import time
from transforms3d import quaternions
from legged_gym.legged_utils.observation_buffer import ObservationBuffer
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.h1.h1_config import H1Cfg

PROPRIOCEPTION_DIM = 63
INTERRUPT_IN_CMD = True    
NOISE_IN_PRIVILEGE = False 
EXECUTE_IN_PRIVILEGE = False 
CMD_DIM = 3 + 4 + 1 + 2 + INTERRUPT_IN_CMD 
TERRAIN_DIM = 221 
PRIVILEGED_DIM = 3 + 1 + 1 + 6 + 11 + 2 + 9 * NOISE_IN_PRIVILEGE + 19 * EXECUTE_IN_PRIVILEGE 
CLOCK_INPUT = 2
DISTURB_DIM = 8

NUM_PARTIAL_OBS = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT
NUM_OBS = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM

DECIMATION = 4

h1_commands_scale = torch.tensor([2.0, 2.0, 0.25, 1.0, 1.0, 1.0, 0.15, 2.0, 0.5, 0.5, 1], device='cuda:0', requires_grad=False)
h12_action_scale = H1Cfg.control.action_scale

clock_inputs = torch.zeros(1, 2, dtype=torch.float, device="cuda:0", requires_grad=False, )
gait_indices = torch.zeros(1, dtype=torch.float, device="cuda:0", requires_grad=False, )
dt = DECIMATION * 1 / 200
#obs_clip = 100

H1_DOF_MAP = {
    'left_ankle_joint': 4, 'left_elbow_joint': 14, 'left_hip_pitch_joint': 2,
    'left_hip_roll_joint': 1, 'left_hip_yaw_joint': 0, 'left_knee_joint': 3,
    'left_shoulder_pitch_joint': 11, 'left_shoulder_roll_joint': 12,
    'left_shoulder_yaw_joint': 13, 'right_ankle_joint': 9, 'right_elbow_joint': 18,
    'right_hip_pitch_joint': 7, 'right_hip_roll_joint': 6,
    'right_hip_yaw_joint': 5, 'right_knee_joint': 8, 'right_shoulder_pitch_joint': 15,
    'right_shoulder_roll_joint': 16, 'right_shoulder_yaw_joint': 17, 'torso_joint': 10
}

H1_2_DOF_NAMES = [
    'left_hip_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_yaw_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'torso_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_pitch_joint', 'left_elbow_roll_joint', 'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_pitch_joint', 'right_elbow_roll_joint', 'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

def h12_action_handler(handler, actions, default_dof_pos, reset_dof_pos, mode=0):
    '''
        Handle action input before sending it to low level command handler.
        Decide which mode to run the action command.

        mode 0: position-based control
        mode 1: torque-based control
        mode 2: damping control
    '''
    # position-based control
    if mode == 0:
        target_pos = reset_dof_pos + 1.0 * (
            default_dof_pos + actions * h12_action_scale - reset_dof_pos
        )
        handler.target_pos = target_pos
    else:
        return NotImplemented


def dof_pos_convert_h1_2_to_h1(h1_2_dof_reading: np.ndarray) -> np.ndarray:
    """
    Converts DOF position readings from H1-2 format to H1 format with corrected logic.

    Args:
        h1_2_dof_reading: A numpy array of shape (27,) with H1-2 DOF values.

    Returns:
        A numpy array of shape (19,) with H1 DOF values.
    """
    if h1_2_dof_reading.shape[0] != len(H1_2_DOF_NAMES):
        raise ValueError(f"Input must have {len(H1_2_DOF_NAMES)} elements, but got {h1_2_dof_reading.shape[0]}")

    # Initialize the output array for H1.
    h1_dof_reading = np.zeros(len(H1_DOF_MAP))

    # Iterate through the H1-2 joints by their index and name.
    for h1_2_idx, h1_2_name in enumerate(H1_2_DOF_NAMES):
        # By default, the name we look for in the H1 map is the H1-2 name itself.
        key_to_check = h1_2_name
        
        # **Corrected Logic**: Only simplify the name for ankle and elbow joints.
        if "ankle_pitch_joint" in h1_2_name or "elbow_pitch_joint" in h1_2_name:
            key_to_check = h1_2_name.replace("_pitch", "")
        
        # Check if the (potentially modified) joint name exists in the target H1 map.
        if key_to_check in H1_DOF_MAP:
            # Get the target index for the H1 robot.
            h1_idx = H1_DOF_MAP[key_to_check]
            # Place the value from the H1-2 reading into the correct H1 position.
            h1_dof_reading[h1_idx] = h1_2_dof_reading[h1_2_idx]
        else:
            print("DOF pos convert met error!")
            raise KeyError

    return h1_dof_reading

def dof_vel_convert_h1_2_to_h1(h1_2_vel_reading: np.ndarray) -> np.ndarray:
    """
    Converts DOF velocity readings from H1-2 format to H1 format.
    
    The mapping, filtering, and reordering logic is identical to the
    DOF position conversion.

    Args:
        h1_2_vel_reading: A numpy array of shape (27,) with H1-2 DOF velocity values.

    Returns:
        A numpy array of shape (19,) with H1 DOF velocity values.
    """
    if h1_2_vel_reading.shape[0] != len(H1_2_DOF_NAMES):
        raise ValueError(f"Input must have {len(H1_2_DOF_NAMES)} elements, but got {h1_2_vel_reading.shape[0]}")

    # Initialize the output array for H1.
    h1_vel_reading = np.zeros(len(H1_DOF_MAP))

    # Iterate through the H1-2 joints by their index and name.
    for h1_2_idx, h1_2_name in enumerate(H1_2_DOF_NAMES):
        # By default, the name we look for in the H1 map is the H1-2 name itself.
        key_to_check = h1_2_name
        
        # Only simplify the name for ankle and elbow joints.
        if "ankle_pitch_joint" in h1_2_name or "elbow_pitch_joint" in h1_2_name:
            key_to_check = h1_2_name.replace("_pitch", "")
        
        # Check if the (potentially modified) joint name exists in the target H1 map.
        if key_to_check in H1_DOF_MAP:
            # Get the target index for the H1 robot.
            h1_idx = H1_DOF_MAP[key_to_check]
            # Place the value from the H1-2 reading into the correct H1 position.
            h1_vel_reading[h1_idx] = h1_2_vel_reading[h1_2_idx]
        else:
            print("DOF vel convert met error!")
            raise KeyError
            
    return h1_vel_reading


def convert_action_h1_to_h1_2(h1_action: torch.Tensor) -> torch.Tensor:
    """
    Converts an H1 action (19 values) to an H1-2 action (27 values).
    
    The input H1 action order is assumed to follow the H1_DOF_MAP.
    Joints that exist in H1-2 but not in H1 are set to 0.0.
    
    This function handles both 1D (19,) and 2D (N, 19) batched tensors.

    Args:
        h1_action: A torch.Tensor of shape (19,) or (N, 19) with H1 action values.

    Returns:
        A torch.Tensor of shape (27,) or (N, 27) with H1-2 action values.
    """
    is_batched = h1_action.dim() == 2
    
    if is_batched:
        # Batched input, e.g., shape (N, 19)
        if h1_action.shape[1] != len(H1_DOF_MAP):
            raise ValueError(f"Input batch must have shape (N, {len(H1_DOF_MAP)}), but got {h1_action.shape}")
        batch_size = h1_action.shape[0]
        output_shape = (batch_size, len(H1_2_DOF_NAMES))
    else:
        # 1D input, e.g., shape (19,)
        if h1_action.shape[0] != len(H1_DOF_MAP):
            raise ValueError(f"Input action must have {len(H1_DOF_MAP)} elements, but got {h1_action.shape}")
        output_shape = (len(H1_2_DOF_NAMES),)

    # Initialize the output tensor with zeros on the same device/dtype.
    h1_2_action = torch.zeros(output_shape, dtype=torch.float, device="cuda:0")
    
    # Iterate through the H1-2 joints by their index and name.
    for h1_2_idx, h1_2_name in enumerate(H1_2_DOF_NAMES):
        # By default, the name we look for in the H1 map is the H1-2 name itself.
        key_to_check = h1_2_name
        
        # Only simplify the name for ankle and elbow joints.
        if "ankle_pitch_joint" in h1_2_name or "elbow_pitch_joint" in h1_2_name:
            key_to_check = h1_2_name.replace("_pitch", "")
        
        # Check if this joint exists in the H1 map.
        if key_to_check in H1_DOF_MAP:
            # If it exists, get its index in the H1 action array.
            h1_idx = H1_DOF_MAP[key_to_check]
            
            # Get the action value from the input H1 action and place it.
            if is_batched:
                h1_2_action[:, h1_2_idx] = h1_action[:, h1_idx]
            else:
                h1_2_action[h1_2_idx] = h1_action[h1_idx]
        
        # If key_to_check is not in H1_DOF_MAP, that joint is an H1-2-only
        # joint, and its value will remain 0.0, as initialized.
            
    return h1_2_action


def make_observation(handler, actions, commands):
    #NOTE: handler return quaternion in w, x, y, z 

    gravity_vector = np.array([0, 0, -1])

    base_ang_vel = handler.ang_vel * LeggedRobotCfg.normalization.obs_scales.ang_vel     # tensor([[ 0.0148,  0.8147, -0.6067]], device='cuda:0')

    # NOTE: probably wrong. Need to recheck this part
    projected_gravity = quaternions.rotate_vector(
        v=gravity_vector,
        q=quaternions.qinverse(handler.quat)
        
    )    # tensor([[ 1.2025e-02,  2.3940e-04, -9.9993e-01]], device='cuda:0')


    # h1 dof-index dict: {'left_ankle_joint': 4, 'left_elbow_joint': 14, 'left_hip_pitch_joint': 2, 'left_hip_roll_joint': 1, 'left_hip_yaw_joint': 0, 'left_knee_joint': 3, 'left_shoulder_pitch_joint': 11, 'left_shoulder_roll_joint': 12, 'left_shoulder_yaw_joint': 13, 'right_ankle_joint': 9, 'right_elbow_joint': 18, 'right_hip_pitch_joint': 7, 'right_hip_roll_joint': 6, 'right_hip_yaw_joint': 5, 'right_knee_joint': 8, 'right_shoulder_pitch_joint': 15, 'right_shoulder_roll_joint': 16, 'right_shoulder_yaw_joint': 17, 'torso_joint': 10}
    '''
    h1 default dof pos
            'left_hip_yaw_joint' : 0.00,   
           'left_hip_roll_joint' : 0.02,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : -0.00, 
           'right_hip_roll_joint' : -0.02, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,                                     
           'torso_joint' : 0., 
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
    '''
    '''
    h1-2 default dof pos
            'left_hip_yaw_joint': 0.0,
            'left_hip_pitch_joint': -0.4,
            'left_hip_roll_joint': 0.02,
            'left_knee_joint': 0.8,
            'left_ankle_pitch_joint': -0.4,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_pitch_joint': -0.4,
            'right_hip_roll_joint': -0.02,
            'right_knee_joint': 0.8,
            'right_ankle_pitch_joint': -0.4,
            'right_ankle_roll_joint': 0.0,
            'torso_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_pitch_joint': 0.0,
            'left_elbow_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_pitch_joint': 0.0,
            'right_elbow_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0
    '''
    # NOTE: the dof reading from h1-2 should be re-arranged to a dof pose list, following the h1 dof-index dict above

    default_dof_pos = handler.default_pos
    # dof pos 
    tmp_joint_pos = (handler.joint_pos - default_dof_pos) * 1.0
    dof_pos = dof_pos_convert_h1_2_to_h1(h1_2_dof_reading=tmp_joint_pos)
    assert(type(dof_pos) == torch.Tensor)

    # dof vel
    tmp_joint_vel = handler.joint_vel * LeggedRobotCfg.normalization.obs_scales.dof_vel
    dof_vel = dof_vel_convert_h1_2_to_h1(h1_2_vel_reading=tmp_joint_vel) 
    assert(type(dof_vel) == torch.Tensor)

    # action
    assert(actions.shape == torch.Size([1, 19]))

    # commands
    assert(commands.shape == torch.Size([1, 11]))
    commands = commands * h1_commands_scale

    # clock inputs
    frequencies = commands[:, 3]
    phases = commands[:, 4]
    durations = commands[:, 5]
    gait_indices = torch.remainder(gait_indices + dt * frequencies, 1.0)
    foot_indices = [gait_indices + phases, gait_indices] 
    clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
    clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
    assert(type(clock_inputs) == torch.Tensor)
     
    obs = torch.cat((
            base_ang_vel,
            projected_gravity,
            dof_pos,
            dof_vel,
            actions,
            commands,
            clock_inputs,
        ), dim=-1)
    
    


    return KeyError


def deploy(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    resume_path = train_cfg.runner.resume_path
    print(resume_path)
    
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_projectedlength_s = 100000 

    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.randomize_load = False
    env_cfg.domain_rand.randomize_gains = False 
    env_cfg.domain_rand.randomize_link_props = False
    env_cfg.domain_rand.randomize_base_mass = False

    env_cfg.commands.resampling_time = 100
    env_cfg.rewards.penalize_curriculum = False
    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.selected = True
    env_cfg.terrain.selected_terrain_type = "random_uniform"

    env_cfg.terrain.measure_heights = False     # was True
    env_cfg.terrain.terrain_kwargs = {  # Dict of arguments for selected terrain
        "random_uniform":
            {
                "min_height": -0.00,
                "max_height": 0.00,
                "step": 0.005,
                "downsampled_scale": 0.2
            },
    }

    env_cfg.env.has_privileged_info=False

    # prepare     # planeenvironment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)


    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    _, _ = env.reset()

    # initialize action, obs and commands
    last_actions = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device)
    #obs, critic_obs, _, _, _ = env.step(last_action)
    commands = torch.zeros(1, 11, dtype=torch.float, device='cuda:0', requires_grad=False)

    # initialize command handler
    cmd_handler = LowStateCmdHandler(cfg=None)
    cmd_handler.init()
    cmd_handler.start()
    
    # load default and reset dof pos
    default_dof_pos = cmd_handler.default_pos.copy()
    reset_dof_pos = cmd_handler.reset_pos.copy()

    # initialize observation buffer
    full_obs_buf = torch.zeros(1, NUM_OBS, device=env.device, dtype=torch.float)
    env_idxs = torch.tensor([0], device='cuda:0')
    obs_buf_history = ObservationBuffer(num_envs=1, 
                                        num_obs=NUM_PARTIAL_OBS, 
                                        include_history_steps=5,
                                        device=env.device,
                                        zero_pad=False)
    # reset buffer
    obs_buf_history.reset(env_idxs, full_obs_buf[:, :NUM_PARTIAL_OBS])

    try:
        while not cmd_handler.Start:
            time.sleep(0.1)

        print("Start runing policy")
        last_update_time = time.time()

        step_id = 0

        while not cmd_handler.emergency_stop:
            if time.time() - last_update_time < 0.02:
                time.sleep(0.001)
                continue
            last_update_time = time.time()

            # make commands, refer play.py for more details
            commands[:, 0] = 0
            commands[:, 1] = 0
            commands[:, 2] = 0
            commands[:, 3] = 2.0
            commands[:, 4] = 0.5
            commands[:, 5] = 0.5
            commands[:, 6] = 0.2
            commands[:, 7] = -0.0
            commands[:, 8] = 0.0
            commands[:, 9] = 0.0
            commands[:, 10] = 1.0

            # make obs 
            obs = make_observation(handler=cmd_handler, actions=last_actions, )
            assert(obs.shape == torch.Size([1, 76]))   # check the shape of obs
            # clip obs
            obs_clip = LeggedRobotCfg.normalization.clip_observations 
            cliped_obs = torch.clip(obs, -obs_clip, obs_clip)
            assert(cliped_obs == torch.Size([1, 76]))  # check the shape of obs after clipping
            obs_buf_history.insert(cliped_obs)
            
            # get obs buffer, historical 5 step observations
            obs_buf, _ = obs_buf_history.get_obs_tensor_3D()
            assert(obs_buf.shape == torch.Size([1, 5, 76]))

            actions = policy.act_inference(obs, privileged_obs=None)

            # store actions for next round observation making
            last_actions = actions.clone()

            # clip actions for current step
            action_clip = LeggedRobotCfg.normalization.clip_actions
            cliped_actions = torch.clip(actions.clone(), -action_clip, action_clip)

            # TODO: action handler before sending action to cmd_handler?
            h12_actions = convert_action_h1_to_h1_2(cliped_actions)

            h12_action_handler(handler=cmd_handler,
                               actions=h12_actions, 
                               default_dof_pos=default_dof_pos,
                               reset_dof_pos=reset_dof_pos,
                               mode=0)
 
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args = get_args()
    deploy(args)