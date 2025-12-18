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

""" Run CPG """

import numpy as np

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

#from quadruped import ComputeInverseKinematics

RENDER = True
TIME_STEP = 0.001
TEST_DURATION = 5.0
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)
TRACK_LEG = 0  # leg index to track (0: FR, 1: FL, 2: RR, 3: RL)
GAIT = "BOUND"  # try TROT, WALK, BOUND, or CUSTOM
TEST_STEPS = int(TEST_DURATION / TIME_STEP)
t = np.arange(TEST_STEPS) * TIME_STEP

leg_labels = ["FR", "FL", "RR", "RL"]
joint_labels = ["hip", "thigh", "calf"]

############## Sample Gains
# joint PD gains
kp_joint = np.array([100, 100, 100])
kd_joint = np.array([2, 2, 2])
# Cartesian PD gains
kp_cartesian = np.diag([500] * 3)
kd_cartesian = np.diag([20] * 3)


def run_trial(*, add_cartesian_pd: bool, render: bool, seed: int, log_cpg: bool):
  """Run one CPG rollout and log desired vs actual tracking for the tracked leg."""
  np.random.seed(seed)  # HopfNetwork uses numpy RNG for initial oscillator conditions

  env = QuadrupedGymEnv(
      render=render,
      on_rack=False,
      isRLGymInterface=False,
      time_step=TIME_STEP,
      action_repeat=1,
      motor_control_mode="TORQUE",
      add_noise=False,
  )
  env.reset(seed=seed)

  cpg = HopfNetwork(time_step=TIME_STEP, gait=GAIT)
  omega_swing = cpg._omega_swing
  omega_stance = cpg._omega_stance

  q_des_hist = np.zeros((3, TEST_STEPS))
  q_act_hist = np.zeros((3, TEST_STEPS))

  xs_hist = np.zeros((4, TEST_STEPS)) if log_cpg else None
  zs_hist = np.zeros((4, TEST_STEPS)) if log_cpg else None
  cpg_r_hist = np.zeros((4, TEST_STEPS)) if log_cpg else None
  cpg_theta_hist = np.zeros((4, TEST_STEPS)) if log_cpg else None
  cpg_dr_hist = np.zeros((4, TEST_STEPS)) if log_cpg else None
  cpg_dtheta_hist = np.zeros((4, TEST_STEPS)) if log_cpg else None
  desired_leg_pos_hist = np.zeros((3, TEST_STEPS)) if log_cpg else None
  actual_leg_pos_hist = np.zeros((3, TEST_STEPS)) if log_cpg else None

  for j in range(TEST_STEPS):
    action = np.zeros(12)

    xs, zs = cpg.update()
    if log_cpg:
      xs_hist[:, j] = xs
      zs_hist[:, j] = zs
      cpg_r_hist[:, j] = cpg.get_r()
      cpg_theta_hist[:, j] = cpg.get_theta()
      cpg_dr_hist[:, j] = cpg.get_dr()
      cpg_dtheta_hist[:, j] = cpg.get_dtheta()

    q = env.robot.GetMotorAngles()
    dq = env.robot.GetMotorVelocities()

    tracked_leg_q_des = None
    for leg_id in range(4):
      leg_xyz = np.array([xs[leg_id], sideSign[leg_id] * foot_y, zs[leg_id]])

      leg_q_des = env.robot.ComputeInverseKinematics(leg_id, leg_xyz)
      tau = kp_joint * (leg_q_des - q[3 * leg_id:3 * leg_id + 3]) + kd_joint * (
          0 - dq[3 * leg_id:3 * leg_id + 3]
      )

      if add_cartesian_pd:
        J_leg, pos_leg = env.robot.ComputeJacobianAndPosition(leg_id)
        foot_vel = J_leg @ dq[3 * leg_id:3 * leg_id + 3]
        tau += J_leg.T @ (
            kp_cartesian @ (leg_xyz - pos_leg) + kd_cartesian @ (0 - foot_vel)
        )

      action[3 * leg_id:3 * leg_id + 3] = tau

      if leg_id == TRACK_LEG:
        tracked_leg_q_des = leg_q_des
        if log_cpg:
          desired_leg_pos_hist[:, j] = leg_xyz
          _, pos_leg = env.robot.ComputeJacobianAndPosition(leg_id)
          actual_leg_pos_hist[:, j] = pos_leg

    env.step(action)
    q_after = env.robot.GetMotorAngles()
    q_des_hist[:, j] = tracked_leg_q_des
    q_act_hist[:, j] = q_after[3 * TRACK_LEG:3 * TRACK_LEG + 3]

  env.close()

  return {
      "q_des": q_des_hist,
      "q_act": q_act_hist,
      "omega_swing": omega_swing,
      "omega_stance": omega_stance,
      "xs_hist": xs_hist,
      "zs_hist": zs_hist,
      "cpg_r_hist": cpg_r_hist,
      "cpg_theta_hist": cpg_theta_hist,
      "cpg_dr_hist": cpg_dr_hist,
      "cpg_dtheta_hist": cpg_dtheta_hist,
      "desired_leg_pos_hist": desired_leg_pos_hist,
      "actual_leg_pos_hist": actual_leg_pos_hist,
  }


def rmse(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
  return np.sqrt(np.mean((desired - actual) ** 2, axis=1))


seed = 0
trial_joint_pd = run_trial(add_cartesian_pd=False, render=False, seed=seed, log_cpg=False)
trial_cart_pd = run_trial(add_cartesian_pd=True, render=RENDER, seed=seed, log_cpg=True)

rmse_joint_pd = rmse(trial_joint_pd["q_des"], trial_joint_pd["q_act"])
rmse_cart_pd = rmse(trial_cart_pd["q_des"], trial_cart_pd["q_act"])
print("==== Gains used ====")
print("kp_joint =", kp_joint, "kd_joint =", kd_joint)
print("kp_cartesian diag =", kp_cartesian.diagonal(), "kd_cartesian diag =", kd_cartesian.diagonal())
print("==== Tracking RMSE (rad) for leg", leg_labels[TRACK_LEG], "====")
print("joint PD only   :", rmse_joint_pd)
print("+ Cartesian PD  :", rmse_cart_pd)

##################################################### 
# PLOTS
#####################################################
avg_omega = 0.5 * (trial_cart_pd["omega_swing"] + trial_cart_pd["omega_stance"])
cycles_to_plot = 3
steps_to_plot = min(TEST_STEPS, int(np.ceil(cycles_to_plot * (2 * np.pi / avg_omega) / TIME_STEP)))
t_plot = t[:steps_to_plot]

# Plot CPG states (r, theta, r_dot, theta_dot) over a few cycles
fig, axes = plt.subplots(4, 4, figsize=(10, 10), sharex=True)
state_titles = [r"$r$", r"$\theta$", r"$\dot{r}$", r"$\dot{\theta}$"]
for leg_idx in range(4):
  axes[leg_idx, 0].plot(t_plot, trial_cart_pd["cpg_r_hist"][leg_idx, :steps_to_plot])
  axes[leg_idx, 1].plot(t_plot, trial_cart_pd["cpg_theta_hist"][leg_idx, :steps_to_plot])
  axes[leg_idx, 2].plot(t_plot, trial_cart_pd["cpg_dr_hist"][leg_idx, :steps_to_plot])
  axes[leg_idx, 3].plot(t_plot, trial_cart_pd["cpg_dtheta_hist"][leg_idx, :steps_to_plot])
  axes[leg_idx, 0].set_ylabel(leg_labels[leg_idx])
for col, title in enumerate(state_titles):
  axes[0, col].set_title(title)
for ax in axes[-1, :]:
  ax.set_xlabel("Time [s]")
fig.suptitle(f"CPG states - {GAIT} gait (~{cycles_to_plot} cycles)")
fig.tight_layout(rect=[0, 0, 1, 0.96])

# Plot foot trajectories in leg frame (x and z)
plt.figure()
for i in range(4):
  plt.plot(t, trial_cart_pd["xs_hist"][i, :], label=f"{leg_labels[i]} x")
  plt.plot(t, trial_cart_pd["zs_hist"][i, :], linestyle="--", label=f"{leg_labels[i]} z")
plt.xlabel("Time [s]")
plt.ylabel("Foot position (x and z) [m]")
plt.title("Foot trajectories vs time")
plt.legend()
plt.tight_layout()

# Plot desired vs actual foot position for tracked leg (leg-frame)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, trial_cart_pd["desired_leg_pos_hist"][0, :], label="desired x")
plt.plot(t, trial_cart_pd["actual_leg_pos_hist"][0, :], label="actual x")
plt.ylabel("x [m]")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(t, trial_cart_pd["desired_leg_pos_hist"][2, :], label="desired z")
plt.plot(t, trial_cart_pd["actual_leg_pos_hist"][2, :], label="actual z")
plt.xlabel("Time [s]")
plt.ylabel("z [m]")
plt.legend()
plt.suptitle(
    f"Leg {leg_labels[TRACK_LEG]} foot tracking (+ Cartesian PD)\n"
    f"kp_joint={kp_joint}, kd_joint={kd_joint}, kp_cart={kp_cartesian.diagonal()}, kd_cart={kd_cartesian.diagonal()}"
)
plt.tight_layout(rect=[0, 0, 1, 0.92])

# Plot desired vs actual joint angles, comparing joint PD vs joint+Cartesian PD
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for j_idx in range(3):
  axes[j_idx].plot(t, trial_joint_pd["q_des"][j_idx, :], "k--", linewidth=1.5, label="desired")
  axes[j_idx].plot(t, trial_joint_pd["q_act"][j_idx, :], label="actual (joint PD)")
  axes[j_idx].plot(t, trial_cart_pd["q_act"][j_idx, :], label="actual (+ Cartesian PD)")
  axes[j_idx].set_ylabel(f"{joint_labels[j_idx]} [rad]")
  axes[j_idx].grid(True, alpha=0.3)
axes[-1].set_xlabel("Time [s]")
axes[0].legend(ncol=3, fontsize=9)
fig.suptitle(
    f"Leg {leg_labels[TRACK_LEG]} joint tracking\n"
    f"kp_joint={kp_joint}, kd_joint={kd_joint}; kp_cart diag={kp_cartesian.diagonal()}, kd_cart diag={kd_cartesian.diagonal()}\n"
    f"RMSE joint PD={rmse_joint_pd.round(3)} rad, +Cart PD={rmse_cart_pd.round(3)} rad"
)
fig.tight_layout(rect=[0, 0, 1, 0.92])

plt.show()
