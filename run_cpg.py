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

import time
import numpy as np
import matplotlib as plt

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

#from quadruped import ComputeInverseKinematics

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(5 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
#---MartinStart---
xs = np.zeros(1)
zs = np.zeros(1)
xs_hist = np.zeros((4, TEST_STEPS))
zs_hist = np.zeros((4, TEST_STEPS))
#---MartinEnd---


############## Sample Gains
#----MartinStart-----
# joint PD gains
kp_joint=np.array([100,100,100])
kd_joint=np.array([2,2,2])
# Cartesian PD gains
kp_cartesian = np.diag([500]*3)
kd_cartesian = np.diag([20]*3)
#----MartinEnd-----



for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12)

  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  xs_hist[:, j] = xs
  zs_hist[:, j] = zs

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  #----Martin Start----#
  q = np.zeros(12)
  dq = np.zeros(12)
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()
  #----Martin end----#
  

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = np.zeros(3) # [TODO]
    #----Martin Start----#
    leg_q =  env.robot.ComputeInverseKinematics(i, leg_xyz)
    #----Martin end ----#

    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += np.zeros(3) # [TODO]
    #----Martin Start----#
    tau += kp_joint * (leg_q - q[3*i:3*i+3]) + kd_joint * (0 - dq[3*i:3*i+3]) #pas sur de savoir comment récup cette ligne
    #----Martin End----#

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      # [TODO]
      _, pos_leg_xyz = env.robot.ComputeJacobianAndPosition(i, leg_q)
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO] 
      jacobian, pos_leg = env.robot.ComputeJacobianAndPosition(i)

      # Get current foot velocity in leg frame (Equation 2)
      # [TODO]
      #----Martin Start----#
      foot_vel = jacobian @ env.robot.GetMotorVelocities()[3*i:3*i+3]
      #----Martin End----#

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += np.zeros(3) # [TODO]
      #----Martin Start----#
      tau += np.transpose(jacobian) @ (kp_cartesian @ (pos_leg_xyz - pos_leg) + kd_cartesian @ (0 - foot_vel))
      #----Martin end----#

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # [TODO] save any CPG or robot states
  # xs = np.vstack((xs, cpg.())) 
  # zs = np.vstack((zs, cpg.get_z()))

##################################################### 
# PLOTS
#####################################################
# [TODO] Create your plots

plt.figure()
leg_labels = ["FR", "FL", "RR", "RL"]
for i in range(4):
  plt.plot(t, xs_hist[i, :], label=f"{leg_labels[i]} x")
  plt.plot(t, zs_hist[i, :], linestyle="--", label=f"{leg_labels[i]} z")
plt.xlabel("Time [s]")
plt.ylabel("Foot position (x and z) [m]")
plt.title("Foot trajectories vs time")
plt.legend()
plt.tight_layout()
plt.show()
