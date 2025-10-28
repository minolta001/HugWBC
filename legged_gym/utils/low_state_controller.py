import time
import sys
import yaml
import argparse

import numpy as np
import torch
from transforms3d import quaternions

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmd_go
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmd_hg
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

from .low_state_handler import LowStateMsgHandler, JointID

class LowStateCmdHandler(LowStateMsgHandler):
    def __init__(self, cfg=None, freq=1000):
        super().__init__(cfg, freq)

        if cfg != None: # for registered robot
            kp_groups = self.cfg.robot_args.dof_kp
            self.kp = [kp_groups[self.group_from_name(name, kp_groups.keys())] for name in self.dof_names]

            kd_groups = self.cfg.robot_args.dof_kd
            self.kd = [kd_groups[self.group_from_name(name, kd_groups.keys())] for name in self.dof_names]

            self.default_pos = np.array([self.cfg.robot_args.default_dof[name] for name in self.dof_names])
            if "reset_joint_angles" in vars(self.cfg.robot_args).keys():
                self.reset_pos = np.array([self.cfg.robot_args.reset_joint_angles[name] for name in self.dof_names])
                self.target_pos = np.array([self.cfg.robot_args.reset_joint_angles[name] for name in self.dof_names])
            else:
                self.reset_pos = np.array([self.cfg.robot_args.default_dof[name] for name in self.dof_names])
                self.target_pos = np.array([self.cfg.robot_args.default_dof[name] for name in self.dof_names])
            self.full_default_pos = np.zeros(self.num_full_dof)
            for i in range(self.num_dof):
                self.full_default_pos[self.dof_index[i]] = self.default_pos[i]
                self.full_joint_pos[self.dof_index[i]] = self.reset_pos[i]

            if self.robot_name == "go2":
                self.low_cmd = unitree_go_msg_dds__LowCmd_()
            elif self.robot_name == "g1":
                self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        else: # for non-registered robot (e.g. h1-2)
            # manually define kp, kd tables
            kp_groups = {
                'ankle_pitch': 40,
                'ankle_roll': 40,
                'elbow_pitch': 100,
                'elbow_roll': 100,
                'hip_pitch': 200,
                'hip_roll': 200,
                'hip_yaw': 200,
                'knee': 300,
                'shoulder_pitch': 100,
                'shoulder_roll': 100,
                'shoulder_yaw': 100,
                'torso': 300,
                'wrist_pitch': 100,
                'wrist_yaw': 100
                }

            self.kp = [kp_groups[self.group_from_name(name, kp_groups.keys())] for name in self.dof_names]
            
            kd_groups = self.cfg.robot_args.dof_kd

            kd_groups = {
                'ankle_pitch': 2,
                'ankle_roll': 2,
                'elbow_pitch': 2,
                'elbow_roll': 2,
                'hip_pitch': 5,
                'hip_roll': 5,
                'hip_yaw': 5,
                'knee': 6,
                'shoulder_pitch': 2,
                'shoulder_roll': 2,
                'shoulder_yaw': 2,
                'torso': 6,
                'wrist_pitch': 2,
                'wrist_yaw': 2
                }

            self.kd = [kd_groups[self.group_from_name(name, kd_groups.keys())] for name in self.dof_names]


            # manually define default dof positions
            self.default_dof = {
                'left_hip_yaw_joint': 0.0,
                'left_hip_pitch_joint': -0.4,
                'left_hip_roll_joint': 0.0,
                'left_knee_joint': 0.8,
                'left_ankle_pitch_joint': -0.4,
                'left_ankle_roll_joint': 0.0,
                'right_hip_yaw_joint': 0.0,
                'right_hip_pitch_joint': -0.4,
                'right_hip_roll_joint': 0.0,
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
            }

            self.default_pos = np.array([self.default_dof[name] for name in self.dof_names])

            self.reset_pos = np.array([self.default_dof[name] for name in self.dof_names])
            self.target_pos = np.array([self.default_dof[name] for name in self.dof_names])
            self.full_default_pos = np.zeros(self.num_full_dof)

            for i in range(self.num_dof):
                self.full_default_pos[self.dof_index[i]] = self.default_pos[i]
                self.full_joint_pos[self.dof_index[i]] = self.reset_pos[i]

            if self.robot_name == "h1-2":
                self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.emergency_stop = False

        # thread handling
        self.lowCmdWriteThreadPtr = None

        self.crc = CRC()

    # Public methods
    def init(self):
        super().init()

        if self.robot_name == "go2":
            self.lidar_switch_publisher = ChannelPublisher("rt/utlidar/switch", String_)
            self.lidar_switch_publisher.Init()
            self.lidar_switch = std_msgs_msg_dds__String_()
            self.lidar_switch.data = "OFF"
            self.lidar_switch_publisher.Write(self.lidar_switch)

        # create publisher #
        if self.robot_name == "go2":
            self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_go)
            self.lowcmd_publisher.Init()
        if self.robot_name == "g1":
            self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_hg)
            self.lowcmd_publisher.Init()
        if self.robot_name == "h1-2":
            self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_hg)
            self.lowcmd_publisher.Init()
       
        # self.sc = SportClient()  
        # self.sc.SetTimeout(5.0)
        # self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def start(self):

        self.msc.ReleaseMode()

        self.full_initial_pos = self.full_joint_pos.copy()
        self.initial_stage = 0.0

        self.init_low_cmd()

        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.update_interval, target=self.LowCmdWrite, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def recover(self):
        status, result = self.msc.CheckMode()
        while result['name'] != 'normal':
            self.msc.SelectMode("normal")
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def init_low_cmd(self):
        if self.robot_name == "g1":
            Kp = [
                60, 60, 60, 100, 40, 40,      # legs
                60, 60, 60, 100, 40, 40,      # legs
                120, 120, 120,                   # waist
                40, 40, 40, 40,  40, 40, 40,  # arms
                40, 40, 40, 40,  40, 40, 40   # arms
            ]
            Kd = [
                1, 1, 1, 2, 1, 1,     # legs
                1, 1, 1, 2, 1, 1,     # legs
                6, 6, 6,              # waist
                1, 1, 1, 1, 1, 1, 1,  # arms
                1, 1, 1, 1, 1, 1, 1   # arms 
            ]
            self.low_cmd.mode_pr = 0  # 0 for pitch roll, 1 for A B
            self.low_cmd.mode_machine = self.msg.mode_machine
            for i in range(29):
                self.low_cmd.motor_cmd[i].mode = 1  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].q = self.full_initial_pos[i]
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kd = Kd[i]
                self.low_cmd.motor_cmd[i].tau = 0. 

        if self.robot_name == "h1-2":
            Kp = [
                200., 200., 200.,   # left hip
                300., 40., 40.,     # left knee/ankle
                200., 200., 200.,   # right hip
                300., 40., 40.,     # right knee/ankle
                200.,               # torso
                40., 40., 18.,      # left shoulder
                18., 18.,           # left elbow
                10., 10.,           # left wrist (new → use conservative defaults)
                40., 40., 18.,      # right shoulder
                18., 18.,           # right elbow
                10., 10.            # right wrist (new → use conservative defaults)
            ]
            Kd = self.kd
            self.low_cmd.mode_pr = 0  # 0 for pitch roll, 1 for A B
            self.low_cmd.mode_machine = self.msg.mode_machine
            for i in range(self.num_dof):
                self.low_cmd.motor_cmd[i].mode = 1  # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].q = self.full_initial_pos[i]
                self.low_cmd.motor_cmd[i].kp = Kp[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kd = Kd[i]
                self.low_cmd.motor_cmd[i].tau = 0.

        elif self.robot_name == "go2":
            # self.low_cmd.head[0]=0xFE
            # self.low_cmd.head[1]=0xEF
            # self.low_cmd.level_flag = 0xFF
            # self.low_cmd.gpio = 0
            for i in range(12):
                self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
                self.low_cmd.motor_cmd[i].q = self.full_initial_pos[i]
                self.low_cmd.motor_cmd[i].kp = 30
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kd = 1.5
                self.low_cmd.motor_cmd[i].tau = 0. 

    def set_stop_cmd(self):
        if self.robot_name == "go2":
            for i in range(20):
                self.low_cmd.motor_cmd[i].mode = 0
                self.low_cmd.motor_cmd[i].q= 2.146e9
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].dq = 16000.0
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0
        elif self.robot_name == "g1":
            for i in range(29):
                self.low_cmd.motor_cmd[i].mode = 0
                self.low_cmd.motor_cmd[i].q= 0
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0
        elif self.robot_name == "h1-2":
            for i in range(self.num_dof):
                self.low_cmd.motor_cmd[i].mode = 0
                self.low_cmd.motor_cmd[i].q= 0
                self.low_cmd.motor_cmd[i].kp = 0
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kd = 5
                self.low_cmd.motor_cmd[i].tau = 0 


    def set_cmd(self):
        for i in range(self.num_dof):
            self.low_cmd.motor_cmd[self.dof_index[i]].q = self.target_pos[i]
            self.low_cmd.motor_cmd[self.dof_index[i]].dq = 0
            self.low_cmd.motor_cmd[self.dof_index[i]].kp = self.kp[i]
            # NOTE: Why kd times 3?
            #self.low_cmd.motor_cmd[self.dof_index[i]].kd = self.kd[i] * 3
            self.low_cmd.motor_cmd[self.dof_index[i]].kd = self.kd[i]
            self.low_cmd.motor_cmd[self.dof_index[i]].tau = 0

    def LowCmdWrite(self):
        if self.L2 and self.R2:
            self.emrgence_stop()

        if self.emergency_stop:
            self.set_stop_cmd()
        elif self.initial_stage < 1.0:
            target_pos = self.full_initial_pos + (self.full_default_pos - self.full_initial_pos) * self.initial_stage
            for i in range(self.num_full_dof):
                self.low_cmd.motor_cmd[i].q = target_pos[i]
                self.low_cmd.motor_cmd[i].kp = 30
                self.low_cmd.motor_cmd[i].kd = 1.5
            self.initial_stage += 0.001
        else:
            self.set_cmd()

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def emrgence_stop(self):
        self.emergency_stop = True

    def group_from_name(self, joint_name: str, groups) -> str:
        for g in sorted(groups, key=len, reverse=True):
            if joint_name.endswith(g + "_joint") or joint_name.endswith(g):
                return g

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--robot', type=str, default='go2')
    parser.add_argument('-n', '--name', type=str, default='default')
    parser.add_argument('-c', '--cfg', type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(f"../{args.robot}.yaml"))
    if args.cfg is not None:
        cfg = yaml.safe_load(open(f"../{args.robot}/{args.cfg}.yaml"))

    # Run steta publisher
    low_state_handler = LowStateCmdHandler(cfg)
    low_state_handler.init()
    low_state_handler.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if low_state_handler.robot_name == "go2":
            low_state_handler.recover()
        sys.exit()