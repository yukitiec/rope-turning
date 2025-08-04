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
def get_day_time():
	"""Get the current day time.
	
	Returns:
		str: Current day time in format 'YYYY-MM-DD HH:MM:SS'
	"""
	return datetime.now().strftime('%Y%m%d_%H%M%S')

# Define your environment (ensure UR5eRopeEnv is correctly implemented and accessible)
class UR5eRopeEnvWrapper(gym.Env):
    def __init__(self,fps,step_episode,client_id):
        self.env = ur5e.UR5eRopeEnv(fps=fps,step_episode=step_episode,client_id=client_id)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_episode_steps = self.env.max_episode_steps
        self.joint_storage = self.env.joint_storage

    @property
    def bool_base(self):
        """Get the current bool_base value from the underlying environment"""
        return self.env.bool_base

    @property #can be used as an attribute.
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
    
    def step(self, action,bool_base=False,bool_render=False):
        return self.env.step(action,bool_base,bool_render)
    
    def step_demo(self):#for demonstration.
        return self.env.step_demo()

    def reset(self,bool_base=False):
        return self.env.reset(bool_base)

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode='human'):
        if mode=="rgb_array":
            frame = self.env.render(mode)
            return frame
        else:
            self.env.render(mode)

    def close(self):
        self.env.close()

class MakeVideo():
    def __init__(self,fps,imgsize,src,videoname):
        self.fps = fps
        self.imgsize = imgsize
        self.src = src
        self.videoname = videoname
        self.main()

    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]

    def main(self):
        img_array=[]
        for filename in sorted(glob.glob(f"{self.src}/*.png"), key=self.natural_keys):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)

        out = cv2.VideoWriter(self.videoname, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.imgsize)#10fps

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

# Initialize PPO algorithm
"""CHECK HERE"""
#learning rate scheduler
rootDir = r"C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\code\robot_RL_curriculum"
type_data = "ppo" # model ; "ppo", "sac"
bool_residual_lr = True # Redidual learning or not.
bool_soft_penalty=False # add a soft penalty. suppress the model's parameters deviate from the baseline model.

#environment FPS.
_fps = 200 #200 fps
_replay_speed = 1.0
_replay_speed = int(_replay_speed*_fps/20)#replay_speed [frames/s] = _fps/(10frame).rendering is every 10 frames.
_rolloout_length=_fps*20 #20 seconds
_step_epsiode = _rolloout_length
#learning framework
bool_sac = False #True : SAC, False:PPo
type_reward = "mix"
bool_train = True
NUM_EPOCH = int(1.0*10**6)#1e6: 250 spisodes. 500 episodes.4*10**6
EPOCH = int(NUM_EPOCH/_rolloout_length)
_length_warmup=int(0.1*EPOCH)#int(0.1*EPOCH/_rolloout_length)

lr_type = "cosine"#"const":constant, "cosine":Cosine scheduler, "mab" : Multi-Armed Bandit algorithm
lr_init = 3.0e-4#3e-4
scheduler = None
if lr_type=="cosine":
    scheduler = lr_scheduler.CosineScheduler(epochs=EPOCH, lr=lr_init, warmup_length=_length_warmup)
#model type
model_type = "with_rope"#"with_rope" : with rope's middle points' estimator. "normal"
## file setting
postfix = type_data + f"_lrType_{lr_type}" + f"_{get_day_time()}"
saveDir = os.path.join(rootDir, postfix)
os.makedirs(saveDir, exist_ok=True)
video_folder = os.path.join(saveDir,"video")
file_base = os.path.join(saveDir,"ppo_actor_critic.pt") # Path to the baseline model. 
"""End : tocheck"""

# Initialize environments
id_train = p.connect(p.DIRECT)
id_eval = p.connect(p.DIRECT)
id_render = p.connect(p.DIRECT)
env = UR5eRopeEnvWrapper(fps=_fps,step_episode = _rolloout_length,client_id=id_train)
env_test = UR5eRopeEnvWrapper(fps=_fps,step_episode = _rolloout_length,client_id=id_eval)
env_render = UR5eRopeEnvWrapper(fps=_fps,step_episode = _rolloout_length,client_id=id_render)

#Actor-Critic + PPO
algo = sac.PPO(
    state_shape=env.observation_space.shape,
    action_shape=env.action_space.shape,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    seed=0,
    batch_size=256,
    gamma=0.995,
    lr_actor=lr_init,
    lr_critic=lr_init,#3e-4
    rollout_length=_rolloout_length,
    num_updates=10,
    clip_eps=0.2,
    lambd=0.97,
    coef_ent=0.0,
    max_grad_norm=0.5,
    lr_scheduler = scheduler, 
    lr_type=lr_type,
    model_type=model_type,
    file_pt = file_base,
    bool_soft_penalty=bool_soft_penalty#add a soft penalyt
)

#SAC
if bool_sac: 
    algo = sac.SAC_alpha(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        seed=0,
        batch_size=256,
        gamma=0.99, 
        lr_actor=lr_init, 
        lr_critic=lr_init,
        lr_alpha=lr_init,
        replay_size=10**6, 
        start_steps=_rolloout_length, 
        tau=5e-3, 
        reward_scale=1.0,
        lr_scheduler = scheduler, 
        lr_type=lr_type,
        model_type=model_type
    )
"""CHECK HERE"""

# Load the pretrained weights
#pretrained_weights = torch.load(r'C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\10Hz\ppo_actor_critic.pt')
# # Load the pretrained weights into the actor and critic
# Load the pretrained weights into the actor and critic
#algo.actor.load_state_dict(pretrained_weights['actor_state_dict'])
#algo.critic.load_state_dict(pretrained_weights['critic_state_dict'])

# # If optimizer states are also saved and you want to load them:
# algo.optim_actor.load_state_dict(pretrained_weights['optim_actor'])
# algo.optim_critic.load_state_dict(pretrained_weights['optim_critic'])

# Initialize Trainer
trainer = trainer.Trainer(
    env=env,
    env_test=env_test,
    env_render=env_render,
    algo=algo,
    bool_sac=bool_sac,
    fps=_fps,
    seed=0,
    num_steps=NUM_EPOCH,          # Total training steps
    eval_interval=int(10*_rolloout_length),      # Evaluation every 10,000 steps
    num_eval_episodes=3,       # Number of episodes per evaluation
    step_epsiode=_step_epsiode,
    rootDir=rootDir,
    type_reward=type_data,
    lr_type=lr_type,
	saveDir = saveDir
)

#File setting##
if bool_train:
    # Start training
    trainer.train()
    print("Finish training")
    # Optionally, visualize the trained policy
    trainer.visualize()
    
    if bool_sac:#plot alpha transition
        trainer.plot_alpha_transition()
    
    #plot lr transition.
    trainer.plot_lr()

    # Plot training progress
    trainer.plot()

    #make a video
    video_path = os.path.join(saveDir, "result.mp4")
    imgsize = (640,480)
    mkVideo_left = MakeVideo(fps=_replay_speed,imgsize=imgsize,src=video_folder,videoname=video_path)

else:#visualize demonstration
    # Optionally, visualize the trained policy
    trainer.visualize_demo()
    video_folder = r"C:/Users/kawaw/python/pybullet/ur5-bullet/UR5/video_demo"
    video_path = os.path.join(video_folder, "result.mp4")
    imgsize = (640,480)
    mkVideo_left = MakeVideo(imgsize=imgsize,src=video_folder,videoname=video_path)
