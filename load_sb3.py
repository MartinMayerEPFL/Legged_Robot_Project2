# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results

LEARNING_ALG = "PPO" #"SAC" or "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '102824115106'
log_dir = interm_dir + '121925205000'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {
    "observation_space_mode": "LR_COURSE_OBS", #DEFAULT, LR_COURSE_OBS
    "task_env": "LR_COURSE_TASK", #  "LR_COURSE_TASK", FWD_LOCOMOTION
    "motor_control_mode": "CARTESIAN_PD", # CARTESIAN_PD, PD
    "terrain" : "SLOPES", # FLAT, SLOPES, ROUGH
}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = True 
env_config['on_rack'] = False  # place robot on ground instead of hanging

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

def unwrap_robot(vec_env):
    """Helper to get underlying QuadrupedGymEnv robot."""
    try:
        return vec_env.venv.envs[0].env.robot, vec_env.venv.envs[0].env
    except Exception:
        return None, None

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# metrics logging (first episode)
robot, base_env = unwrap_robot(env)
dt = base_env._time_step * base_env._action_repeat
total_mass = float(np.sum(robot.GetTotalMassFromURDF()))
g = 9.81
base_pos_start = np.array(robot.GetBasePosition(), dtype=np.float64)
last_base_pos = base_pos_start.copy()
energy_joules = 0.0
stance_time = np.zeros(4, dtype=np.float64)
swing_time = np.zeros(4, dtype=np.float64)
stance_bouts_s = []
swing_bouts_s = []
contact_prev = None
phase_start_s = np.zeros(4, dtype=np.float64)
valid_steps = 0
fell_at_s = None
vx_hist = []
vxy_hist = []
t_hist = []

for i in range(2000):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards

    # log velocities
    base_lin_vel = np.array(robot.GetBaseLinearVelocity(), dtype=np.float64)
    vx_hist.append(base_lin_vel[0])
    vxy_hist.append(np.linalg.norm(base_lin_vel[:2]))
    t_hist.append(valid_steps * dt)

    # energy for CoT
    motor_torques = np.array(robot.GetMotorTorques(), dtype=np.float64)
    motor_velocities = np.array(robot.GetMotorVelocities(), dtype=np.float64)
    energy_joules += float(np.sum(np.abs(motor_torques * motor_velocities))) * dt

    # contact accounting
    feet_contact = np.array(robot.GetContactInfo()[3], dtype=np.int32)
    stance_time += feet_contact * dt
    swing_time += (1 - feet_contact) * dt

    if contact_prev is None:
        contact_prev = feet_contact.copy()
        phase_start_s[:] = 0.0
    else:
        t_s = (valid_steps + 1) * dt
        for leg_id in range(4):
            if feet_contact[leg_id] != contact_prev[leg_id]:
                dur = t_s - phase_start_s[leg_id]
                if contact_prev[leg_id] == 1:
                    stance_bouts_s.append(dur)
                else:
                    swing_bouts_s.append(dur)
                phase_start_s[leg_id] = t_s
                contact_prev[leg_id] = feet_contact[leg_id]

    valid_steps += 1
    last_base_pos = np.array(robot.GetBasePosition(), dtype=np.float64)

    if base_env.is_fallen() and fell_at_s is None:
        fell_at_s = valid_steps * dt

    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        break

# close open bouts
total_time_s = valid_steps * dt
if contact_prev is not None and total_time_s > 0:
    for leg_id in range(4):
        dur = total_time_s - phase_start_s[leg_id]
        if contact_prev[leg_id] == 1:
            stance_bouts_s.append(dur)
        else:
            swing_bouts_s.append(dur)

distance_m = float(np.linalg.norm((last_base_pos - base_pos_start)[:2]))
denom = total_mass * g * max(distance_m, 1e-6)
cot = float(energy_joules / denom)
mean_stance_bout_s = float(np.mean(stance_bouts_s)) if stance_bouts_s else 0.0
mean_swing_bout_s = float(np.mean(swing_bouts_s)) if swing_bouts_s else 0.0
duty_factor_per_leg = stance_time / np.maximum(stance_time + swing_time, 1e-9)
mean_duty_factor = float(np.mean(duty_factor_per_leg)) if total_time_s > 0 else 0.0

print("\n==== Episode metrics (until fall/termination) ====")
print(f"  total time: {total_time_s:.3f} s, distance: {distance_m:.3f} m")
print(f"  mean stance bout: {mean_stance_bout_s:.3f} s | mean swing bout: {mean_swing_bout_s:.3f} s | mean duty factor: {mean_duty_factor:.3f}")
print(f"  energy: {energy_joules:.2f} J | mass: {total_mass:.2f} kg | CoT: {cot:.3f}")
if fell_at_s is not None:
    print(f"  fell at t = {fell_at_s:.3f} s (metrics computed up to fall)")

# Plot vx and |v_xy| vs time
plt.figure()
plt.plot(t_hist, vx_hist, label="vx")
plt.plot(t_hist, vxy_hist, linestyle="--", label="|v_xy|")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.title("Base speed vs time")
plt.legend()
plt.tight_layout()
plt.show()
