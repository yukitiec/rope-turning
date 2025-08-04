import gym
import torch
import glob
import cv2
import os
import re
import pybullet as p

import ur5e_rope as ur5e
import sac_withppo as sac
import trainer
import cosine_scheduler as lr_scheduler
import time
from datetime import datetime
import shutil
import yaml

#MPC part
from IPython.display import clear_output
import os
import time

import numpy as np
import casadi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys
#pybullet
import mpc_controller_simple as mpc
import mpc_controller_4springs as mpc_4springs

# Add the parent directory to the path to allow imports from other directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

#ur5e
try:
    from ur5e import ur_custom_casadi as ur_kin  # Commented out - not used
except ImportError:
    print("Warning: Could not import ur_custom_casadi")
    ur_kin = None

#utils
try:
    import utils._create_update_model as update_model
    import utils.reference as ref
    import utils.human_motion as hum_motion
    import utils.kalman_filter_3d as kf
    from utils.kalman_filter_3d import KalmanFilter3D
except ImportError:
    print("Warning: Could not import utils._create_update_model")
    ref = None
    hum_motion = None



def get_day_time():
    """Get the current day time.

    Returns:
        str: Current day time in format 'YYYY-MM-DD HH:MM:SS'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# Define your environment (ensure UR5eRopeEnv is correctly implemented and accessible)
class UR5eRopeEnvWrapper(gym.Env):
    def __init__(self, fps, step_episode, client_id):
        self.env = ur5e.UR5eRopeEnv(
            fps=fps, step_episode=step_episode, client_id=client_id
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_episode_steps = self.env.max_episode_steps
        self.joint_storage = self.env.joint_storage

    @property
    def bool_base(self):
        """Get the current bool_base value from the underlying environment"""
        return self.env.bool_base

    @property  # can be used as an attribute.
    def rosbag(self):
        """Get the current rosbag from the underlying environment"""
        return self.env.rosbag

    @property
    def joints_list(self):
        """Get the current joints_list from the underlying environment"""
        return self.env.joints_list

    @property
    def rope_mid_point(self):
        return self.env.rope_mid_point

    @property
    def rope_mid_point_estimate(self):
        return self.env.rope_mid_point_estimate

    @property
    def rope_length(self):
        return self.env.rope_length

    @property
    def R_r2h(self):
        return self.env.R_r2h

    @property
    def moving_point_center(self):
        return self.env.moving_point_center

    @property
    def moving_point_radius(self):
        return self.env.moving_point_radius

    def step(self, action):
        return self.env.step(action)

    def step_demo(self):  # for demonstration.
        return self.env.step_demo()

    def reset(self, i=0, rope_length=0.5, bool_base=False, bool_eval=False):
        return self.env.reset(i=i, rope_length=rope_length, bool_base=bool_base, bool_eval=bool_eval)

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode="human"):
        if mode == "rgb_array":
            frame = self.env.render(mode)
            return frame
        else:
            self.env.render(mode)

    def close(self):
        self.env.close()


class MakeVideo:
    def __init__(self, fps, imgsize, src, videoname):
        self.fps = fps
        self.imgsize = imgsize
        self.src = src
        self.videoname = videoname
        self.main()

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r"(\d+)", text)]

    def main(self):
        img_array = []
        for filename in sorted(glob.glob(f"{self.src}/*.png"), key=self.natural_keys):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(
            self.videoname, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.imgsize
        )  # 10fps

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


# Initialize PPO algorithm
"""CHECK HERE"""
params_sim = {
    "step_epsiode": 5,  # 20 seconds.
    "num_eval_episodes": 3,  # number of episodes per evaluation
    "root_dir": os.getcwd(),
    "root_config": r"C:\Users\kawaw\python\mpc\casadi_mpc_nyuumon\src\pybullet\config"
}

rootDir = params_sim["root_dir"]
"""End : tocheck"""

#parameters setting.
root_config = params_sim["root_config"]
#config_env = os.path.join(root_config, "env", "mpc_simple.yaml")
config_env = os.path.join(root_config, "env", "mpc_simple_4springs.yaml")
config_policy = os.path.join(root_config, "policy", "mpc.yaml")

try:
    with open(config_env, 'r', encoding='utf-8') as f:
        yaml_env = yaml.load(f, Loader=yaml.FullLoader)
except UnicodeDecodeError:
    # Fallback to system default encoding if UTF-8 fails
    with open(config_env, 'r') as f:
        yaml_env = yaml.load(f, Loader=yaml.FullLoader)

ns = int(yaml_env["params"]["ns"])
print(ns)
if ns==2:
    mpc_controller = mpc.MPCController(config_env, config_policy)
    postfix = f"_2springs_{get_day_time()}"
    saveDir = os.path.join(rootDir, postfix)
    os.makedirs(saveDir, exist_ok=True)
    file_base = os.path.join(saveDir, "ppo_actor_critic.pt")  # Path to the baseline model.
    ## file setting
    video_folder = os.path.join(saveDir, "video")
elif ns==4:
    mpc_controller = mpc_4springs.MPCController(config_env, config_policy)
    postfix = f"_4springs_{get_day_time()}"
    saveDir = os.path.join(rootDir, postfix)
    os.makedirs(saveDir, exist_ok=True)
    file_base = os.path.join(saveDir, "ppo_actor_critic.pt")  # Path to the baseline model.
    ## file setting
    video_folder = os.path.join(saveDir, "video")

#solver setting.
mpc_controller.type_solver = "ipopt" #"ipopt","qrsqp", "osqp", "qpoases"
S = mpc_controller.get_solver(mpc_controller.type_solver)

#simulation control frequency.
_fps = mpc_controller.control_frequency
_replay_speed = 1.0
_replay_speed = int(
    _replay_speed * _fps / 10
)  # replay_speed [frames/s] = _fps/(10frame).rendering is every 10 frames.
_step_epsiode = _fps * params_sim["step_epsiode"]

# Initialize environments
id_render = p.connect(p.DIRECT)
env_render = UR5eRopeEnvWrapper(
    fps=_fps, step_episode=_step_epsiode, client_id=id_render
)

# Initialize Trainer
trainer_instance = trainer.Trainer(
    mpc_controller=mpc_controller,
    S=S,
    env_render=env_render,
    fps=_fps,
    seed=0,
    step_epsiode=_step_epsiode,
    rootDir=rootDir,
    saveDir=saveDir,
)

# Optionally, visualize the trained policy
trainer_instance.visualize()

# make a video
video_path = os.path.join(saveDir, "result.mp4")
imgsize = (640, 480)
mkVideo_left = MakeVideo(
    fps=_replay_speed, imgsize=imgsize, src=video_folder, videoname=video_path
)

# Clean up video folder after video creation
if os.path.exists(video_folder):
    shutil.rmtree(video_folder)
