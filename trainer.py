import time
import numpy as np
import torch
from datetime import timedelta
import os
import gym
import glob
from base64 import b64encode
from IPython.display import HTML
from gym.wrappers.monitoring import video_recorder
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import mpc_controller_simple as mpc
import mpc_controller_4springs as mpc_4springs

# Add the parent directory to the path to allow imports from other directories
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
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

"""CHECK HERE"""
rootDir = r"C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\code\robot_RL_curriculum\residual_policy\residual_target"
type_reward = "ppo"
lr_type = "constant"
bool_residual_lr = False  # Redidual learning or not.
file_base = os.path.join(rootDir, "ppo_actor_critic.pt")  # Path to the baseline model.
"""CHECK HERE"""


class Trainer:

    def __init__(
        self,
        mpc_controller, #MPC controller
        S, #solver
        env_render,
        fps=200,
        seed=0,
        step_epsiode=4 * 10**3,
        rootDir=rootDir,
        saveDir=rootDir,
    ):
        self.mpc_controller = mpc_controller
        self.S = S
        self.env_render = env_render
        self.fps = fps
        self.saveDir = saveDir
        os.makedirs(self.saveDir, exist_ok=True)
        # Dictionary to store average returns
        self.returns = {"step": [], "return": [], "return_base": [], "return_diff": []}

        self.step_epsiode = step_epsiode


    def visualize(self):
        """Visualize a single episode using the trained policy."""
        # env = gym.make(self.env.unwrapped.spec.id)
        # Ensure the video folder exists
        video_folder = os.path.join(self.saveDir, "video")

        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        ### system setting

        # self.env = gym.wrappers.RecordVideo(self.env, saveDir,episode_trigger=lambda x: True)  # Save videos to the 'videos/' directory
        state = self.env_render.reset(i=1, rope_length=self.mpc_controller.length_rope, bool_base=False, bool_eval=True)
        # State
        #    end_effector_position,  # 3
        #    end_effector_orientation,  # 3 (6)
        #    vel_eef,  # 6 (12)
        #    self.target_position,  # 3 (15)
        #    self.vel_human,  # 3 (18)
        #    net_force,  # 3 (21)
        #    net_torque,  # 3 (24)
        #    state_rope_middle,  # 7. (length,pos,vel) (31)
        #Action by MPC.
        done = False

        #Kalman filter setting.
        #Initial state.
        x_init = self.mpc_controller.make_x_init(state[:3], state[12:15], self.env_render.rope_length)
        offset = 3 * (self.mpc_controller.ns + 1) #position of the mass points.
        x_current = x_init[:,0]
        x0 = self.mpc_controller.make_x0(x_current)

        #Storage for the results.
        Times_ipopt = []
        X = [x_init]
        U = []#robot's linear velocity.

        # Human motion setting.
        p_hum_prev = state[12:15]

        # human kf model
        kf_hum = KalmanFilter3D(
            initial_x=p_hum_prev[0],
            initial_y=p_hum_prev[1],
            initial_z=p_hum_prev[2],
            initial_vx=0,  # initial_vel[0],
            initial_vy=0,  # initial_vel[1],
            initial_vz=0,  # initial_vel[2],
            initial_ax=0,  # initial_acc[0],
            initial_ay=0,  # initial_acc[1],
            initial_az=0,
            process_noise_pos=1,
            process_noise_vel=1,
            process_noise_acc=1,
            measurement_noise=1,
            dt=self.mpc_controller.dt,
            measure_velocity=True,  # measure human's velocity.
        )

        p_mid = state[25:28]
        kf_mid = KalmanFilter3D(
            initial_x=p_mid[0],
            initial_y=p_mid[1],
            initial_z=p_mid[2],
            initial_vx=0,  # initial_vel[0],
            initial_vy=0,  # initial_vel[1],
            initial_vz=0,  # initial_vel[2],
            initial_ax=0,  # initial_acc[0],
            initial_ay=0,  # initial_acc[1],
            initial_az=0,
            process_noise_pos=1,
            process_noise_vel=1,
            process_noise_acc=1,
            measurement_noise=1,
            dt=self.mpc_controller.dt,
            measure_velocity=True,  # measure human's velocity.
        )

        #Storage for the results.
        x_hum_pos_list = []
        x_hum_vel_list = []
        x_hum_acc_list = []
        x_hum_pos_gt = []
        x_hum_vel_gt = []
        x_hum_acc_gt = []
        x_mid_pos_gt = []
        x_mid_vel_gt = []
        x_mid_acc_gt = []
        x_hum_kf_pos_list = []
        x_hum_kf_vel_list = []
        x_hum_kf_acc_list = []
        x_mid_kf_pos_list = []
        x_mid_kf_vel_list = []
        x_mid_kf_acc_list = []
        vel_robot = np.zeros(3)
        q_robot_list = [self.env_render.joints_list.tolist()]#(6,)
        p_robot_list = [state[:6].tolist()]
        a_human = np.zeros(3)

        #Simulation setting.
        counter = 0
        fps = self.fps
        period_save = int(fps / 10)
        observations = np.empty((0, 31))
        # Initialize an empty numpy array
        times = []
        joints_storage = []
        while not done:
            # State
            #    end_effector_position,  # 3
            #    end_effector_orientation,  # 3 (6)
            #    vel_eef,  # 6 (12)
            #    self.target_position,  # 3 (15)
            #    self.vel_human,  # 3 (18)
            #    net_force,  # 3 (21)
            #    net_torque,  # 3 (24)
            #    state_rope_middle,  # 7. (length,pos,vel) (31)
            # Action by MPC.
            start_time = time.perf_counter()
            u_opt, x0,res = self.mpc_controller.solve_optimization(x_current, a_human, x0=x0, S=self.S)
            x0=res["x"]
            Times_ipopt.append(time.perf_counter() - start_time)

            action = u_opt.full().ravel() #convert to numpy array

            state, state_rope_middle, reward, done, _ = self.env_render.step(action)
            observations = np.append(observations, [state], axis=0)
            joints_storage.append(self.env_render.joints_list.tolist())
            times.append(counter / fps)

            X.append(x_current)

            U.append(action.tolist())#.full().ravel().tolist())

            #get the measurement of the human and the middle point.
            pos_human = state[12:15] + np.random.normal(0, 0.03, 3)
            vel_human = state[15:18] + np.random.normal(0, 0.03, 3)
            pos_mid = state[25:28] + np.random.normal(0, 0.03, 3)
            vel_mid = state[28:31] + np.random.normal(0, 0.03, 3)
            # Kalman filter prediction and update
            measure_human = np.concatenate([pos_human, vel_human])
            measure_mid = np.concatenate([pos_mid, vel_mid])
            #KF update
            kf_hum.predict()
            kf_hum.update(measure_human)  # pos_human)
            kf_mid.predict()
            kf_mid.update(measure_mid)
            #Get internal state
            pos_kf = kf_hum.get_position()
            vel_kf = kf_hum.get_velocity()
            acc_kf = kf_hum.get_acceleration()
            pos_kf_mid = kf_mid.get_position()
            vel_kf_mid = kf_mid.get_velocity()
            acc_kf_mid = kf_mid.get_acceleration()
            # estimate human's acceleration.
            x_hum_kf_pos_list.append(pos_kf.tolist())
            x_hum_kf_vel_list.append(vel_kf.tolist())
            x_hum_kf_acc_list.append(acc_kf.tolist())
            x_mid_kf_pos_list.append(pos_kf_mid.tolist())
            x_mid_kf_vel_list.append(vel_kf_mid.tolist())
            x_mid_kf_acc_list.append(acc_kf_mid.tolist())
            #ground truth
            x_hum_pos_gt.append(state[12:15].tolist())
            x_hum_vel_gt.append(state[15:18].tolist())
            x_mid_pos_gt.append(state[25:28].tolist())
            x_mid_vel_gt.append(state[28:31].tolist())

            #update the current state
            if self.mpc_controller.ns == 2: #4 springs.
                offset = 3 * (self.mpc_controller.ns + 1) #position of the mass points.
                #robot
                x_current[:3] = state[:3].copy() #human's position.
                x_current[offset:offset+3] = state[6:9].copy() #eef linear velocity.
                x0[0:3] = state[:3].copy()
                x0[offset:offset+3] = state[offset:offset+3].copy()
                #middle.
                x_current[3:6] = pos_kf_mid.copy()
                x_current[offset+3:offset+6] = np.clip(vel_kf_mid.copy(), -6.0,6.0)
                x0[3:6] = pos_kf_mid.copy()
                x0[offset+3:offset+6] = np.clip(vel_kf_mid.copy(), -6.0,6.0)
                #human
                x_current[6:9] = pos_kf.copy()
                x_current[offset+6:offset+9] = np.clip(vel_kf.copy(), -6.0,6.0)
                x0[6:9] = pos_kf.copy()
                x0[offset+6:offset+9] = np.clip(vel_kf.copy(), -6.0,6.0)
            elif self.mpc_controller.ns == 4: #4 springs.
                offset = 3 * (self.mpc_controller.ns + 1) #position of the mass points.
                #robot
                x_current[:3] = state[:3].copy() #human's position.
                x_current[offset:offset+3] = state[6:9].copy() #eef linear velocity.
                x0[0:3] = state[:3].copy()
                x0[offset:offset+3] = state[offset:offset+3].copy()
                #middle.
                x_current[6:9] = pos_kf_mid.copy()
                x_current[offset+6:offset+9] = np.clip(vel_kf_mid.copy(), -6.0,6.0)
                x0[6:9] = pos_kf_mid.copy()
                x0[offset+6:offset+9] = np.clip(vel_kf_mid.copy(), -6.0,6.0)
                #human
                x_current[12:15] = pos_kf.copy()
                x_current[offset+12:offset+15] = np.clip(vel_kf.copy(), -6.0,6.0)
                x0[12:15] = pos_kf.copy()
                x0[offset+12:offset+15] = np.clip(vel_kf.copy(), -6.0,6.0)
            #acceleration
            a_human = np.clip(acc_kf.copy(),-9.8,9.8)

            if counter % period_save == 1:
                print(f"counter={counter}/{self.step_epsiode} :: average processing speed : {1.0/(np.mean(Times_ipopt)+1e-6):.1f} Hz")
                frame = self.env_render.render(mode="rgb_array")
                if isinstance(frame, np.ndarray):
                    file_img = os.path.join(video_folder, f"{counter:05d}.png")
                    # print("frame.shape=",frame.shape)
                    cv2.imwrite(file_img, frame)
            counter += 1

        joints_storage = np.array(joints_storage)
        self.env_render.close()
        times = np.array(times)
        self.plot_eval(times, observations, joints_storage)
        self.plot_kf(x_hum_pos_list=x_hum_pos_gt, x_hum_vel_list=x_hum_vel_gt, x_hum_kf_pos_list=x_hum_kf_pos_list, x_hum_kf_vel_list=x_hum_kf_vel_list, x_hum_kf_acc_list=x_hum_kf_acc_list, t_eval=times, saveDir=self.saveDir)
        self.plot_kf(x_hum_pos_list=x_mid_pos_gt, x_hum_vel_list=x_mid_vel_gt, x_hum_kf_pos_list=x_mid_kf_pos_list, x_hum_kf_vel_list=x_mid_kf_vel_list, x_hum_kf_acc_list=x_mid_kf_acc_list, t_eval=times, saveDir=self.saveDir,type_plot="rope")
        self.plot_mpc(X=X, U=U, t_eval=times, saveDir=self.saveDir)
        self.plot_time(Times_ipopt=Times_ipopt)

    def plot_time(self, Times_ipopt):
        Times_ipopt = np.array(Times_ipopt)
        print(f"[*] solver: ipopt, elapsed time: {Times_ipopt.mean():1f}s")
        plt.plot(Times_ipopt)
        plt.xlabel("Time")
        plt.ylabel("Elapsed Time")
        plt.savefig(os.path.join(self.saveDir, "mpc_time.png"))
        plt.clf()

    def plot_kf(self, x_hum_pos_list, x_hum_vel_list, x_hum_kf_pos_list, x_hum_kf_vel_list, x_hum_kf_acc_list, t_eval, saveDir,type_plot="human"):
        # HUman state
        x_hum_pos_list = np.array(x_hum_pos_list)
        x_hum_vel_list = np.array(x_hum_vel_list)
        x_hum_kf_pos_list = np.array(x_hum_kf_pos_list)
        x_hum_kf_vel_list = np.array(x_hum_kf_vel_list)
        x_hum_kf_acc_list = np.array(x_hum_kf_acc_list)

        fig, ax = plt.subplots(3, 3, figsize=(20, 20))
        for i in range(3):
            ax[i, 0].plot(t_eval, x_hum_pos_list[:, i], linestyle="--", label="actual")
            ax[i, 0].plot(t_eval, x_hum_kf_pos_list[:, i], label="kf")
            ax[i, 0].legend()
            ax[i, 0].set_title(f"position")
            ax[i, 1].plot(t_eval, x_hum_vel_list[:, i], linestyle="--", label="actual")
            ax[i, 1].plot(t_eval, x_hum_kf_vel_list[:, i], label="kf")
            ax[i, 1].legend()
            ax[i, 1].set_title(f"velocity")
            # ax[i,2].plot(t_eval,x_hum_acc_list[:,i],linestyle="--"label="actual")
            ax[i, 2].plot(t_eval, x_hum_kf_acc_list[:, i], label="kf")
            ax[i, 2].legend()
            ax[i, 2].set_title(f"acceleration")
        plt.savefig(os.path.join(saveDir, f"{type_plot}_kf_acc.png"))
        plt.clf()

    def plot_mpc(self, X, U, t_eval, saveDir):
        # Rope state
        X.pop()
        X = np.array(X).reshape(t_eval.size, self.mpc_controller.nx)
        U = np.array(U).reshape(t_eval.size, self.mpc_controller.nu)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        for k in range(3):
            plt.plot(t_eval, X[:, k], label=f"x_{k}")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.subplot(1, 2, 2)
        for k in range(self.mpc_controller.nu):
            plt.step(t_eval, U[:, k], linestyle="--", label=f"u_{k}")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Control")
        plt.savefig(os.path.join(saveDir, "mpc_result.png"))
        plt.clf()

    def plot_eval(self, times, observations, joints_storage):
        """
        Parameters:
        -----------
        observations :
            end_effector_position, #3
            end_effector_orientation, #4 (7)
            vel_eef, #7 (14)
            self.pos_human_current, #3 (17)
            self.vel_human, #3 (20)
            net_force, #3 (23)
            net_torque, #3 (26)
            state_rope_middle  #7. (length,pos,vel) (33)
        """
        # robot joints.
        fig0, ax0 = plt.subplots(1, 6, figsize=(35, 6))  # (w,h)
        titles = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
        for i in range(6):
            ax0[i].plot(times, joints_storage[:, i], color="k")
            ax0[i].set_title(titles[i], fontsize=20)
            ax0[i].set_xlabel("Time [s]", fontsize=16)
            ax0[i].set_ylabel("Angle [radian]", fontsize=16)
            ax0[i].tick_params(axis="both", labelsize=14)
            # ax0[i].set_ylim(-6.28,6.28)
        # fig0.suptitle("Joint angles",fontsize=20)
        plt.tight_layout()
        file_img = os.path.join(self.saveDir, "joints.png")
        plt.savefig(file_img)
        plt.clf()

        # Convert to DataFrame and save
        data_save = []
        for i in range(joints_storage.shape[0]):
            temp = [times[i]]
            for j in range(6):
                temp.append(joints_storage[i][j])
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv = os.path.join(self.saveDir, "joints.csv")
        df.to_csv(
            file_csv, index=False, header=False
        )  # Set header=True if you need column names

        # robot position and target position
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        titles = ["X ", "Y", "Z"]
        axes = ["X [m]", "Y [m]", "Z [m]"]
        for i in range(3):
            ax[i].plot(times, observations[:, i + 12], color="r", label="Human")
            ax[i].plot(times, observations[:, i], color="b", label="Robot")
            # ax[i].set_title(titles[i],fontsize=18)
            ax[i].set_xlabel("Time [s]", fontsize=16)
            ax[i].set_ylabel(axes[i], fontsize=16)
            ax[i].tick_params(axis="both", labelsize=16)
            if i == 2:
                ax[i].legend(loc="best", fontsize=16)
            plt.tight_layout()
        file_img = os.path.join(self.saveDir, "position_transition.png")
        fig.savefig(file_img)
        plt.clf()
        # fig.show()

        data_save = []
        for i in range(observations.shape[0]):  # for each sequence.
            temp = [times[i]]
            for j in range(3):  # target position
                temp.append(observations[i][j])
            for j in range(3):  # robot end-effector position
                temp.append(observations[i][3 + j])
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv = os.path.join(self.saveDir, "ee_transition.csv")
        df.to_csv(
            file_csv, index=False, header=False
        )  # Set header=True if you need column names

        distances = []
        for i in range(observations.shape[0]):
            d = (
                (observations[i, 0] - observations[i, 12]) ** 2
                + (observations[i, 1] - observations[i, 13]) ** 2
                + (observations[i, 2] - observations[i, 14]) ** 2
            ) ** (0.5)
            distances.append(d)
        distances = np.array(distances)

        fig = plt.figure(figsize=(8, 6))
        plt.plot(times, distances, color="k", linewidth=2)
        plt.xlabel("Time [s]", fontsize=18)
        plt.ylabel("Distance [m]", fontsize=18)
        plt.axhline(y=observations[-1, -7], color="r", linestyle="--", linewidth=2)
        plt.tick_params(labelsize=16)
        # plt.title('distance between end effector and target', fontsize=18)
        plt.tight_layout()
        file_img = os.path.join(self.saveDir, "distance.png")
        plt.savefig(file_img)
        plt.clf()

        data_save = []
        for i in range(distances.shape[0]):  # for each sequence
            temp = []
            temp.append(times[i])
            temp.append(distances[i])
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv = os.path.join(self.saveDir, "distance.csv")
        df.to_csv(
            file_csv, index=False, header=False
        )  # Set header=True if you need column names

        # middle points
        # load ground truth
        # file_path = r"C:\Users\kawaw\python\pybullet\ur5-bullet\UR5\code\robot_RL\middle_points.csv"
        # data = pd.read_csv(filepath_or_buffer=file_path)
        # data = data.values
        fig2, ax2 = plt.subplots(1, 3, figsize=(21, 7))
        titles = ["X", "Y", "Z"]
        axes = ["X [m]", "Y [m]", "Z [m]"]
        for i in range(3):
            # ax2[i].plot(data[:,0],data[:,(i+1)],color="b",label="ideal")#time, (x,y,z)
            ax2[i].plot(times, observations[:, -6 + i], color="k", label="middle")
            ax2[i].plot(times, observations[:, -13 + i], color="r", label="force")
            ax2[i].set_title(titles[i], fontsize=16)
            ax2[i].set_xlabel("Time [sec]", fontsize=16)
            ax2[i].set_ylabel(axes[i], fontsize=16)
            ax2[i].tick_params(axis="both", labelsize=16)
            if i == 2:
                ax2[i].legend(loc="best", fontsize=16)
        # fig2.suptitle("Middle position of the long rope")
        plt.tight_layout()
        file_img = os.path.join(self.saveDir, "middle_point.png")
        fig2.savefig(file_img)
        plt.clf()

        # data_save = []
        # for i in range(data.shape[0]):#for each sequence.
        #     temp = []
        #     for j in range(4):
        #         temp.append(data[i,j])#add ideal data
        #     data_save.append(temp)
        # data_save = np.array(data_save)
        # df = pd.DataFrame(data_save)
        # file_csv =os.path.join(self.saveDir,"ee_ideal.csv")
        # df.to_csv(file_csv, index=False, header=False)  # Set header=True if you need column names

        data_save = []
        for i in range(observations.shape[0]):  # for each sequence.
            temp = [times[i]]
            for j in range(3):
                temp.append(observations[i, -6 + j])  # add ideal data
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv = os.path.join(self.saveDir, "middle.csv")
        df.to_csv(
            file_csv, index=False, header=False
        )  # Set header=True if you need column names

        data_save = []
        for i in range(observations.shape[0]):  # for each sequence.
            temp = [times[i]]
            for j in range(3):
                temp.append(observations[i, -13 + j])  # add ideal data
            data_save.append(temp)
        data_save = np.array(data_save)
        df = pd.DataFrame(data_save)
        file_csv = os.path.join(self.saveDir, "force.csv")
        df.to_csv(
            file_csv, index=False, header=False
        )  # Set header=True if you need column names

    @property
    def time(self):
        """Calculate the elapsed training time."""
        return str(timedelta(seconds=int(time.time() - self.start_time)))
