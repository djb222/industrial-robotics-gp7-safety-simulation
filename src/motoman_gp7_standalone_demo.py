# motoman_gp7_cylinders_fixed.py
# Author: Dylan B
# Description: Motoman GP7 visualised in Swift using cylindrical link geometry
# ----------------------------------------------------------------------------

import swift
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base import transl
from spatialgeometry import Cuboid, Cylinder, Sphere
from ir_support import CylindricalDHRobotPlot
from math import pi
import time
import numpy as np

# # ---------------------------------------------------------------------------
# # Define Motoman GP7 parameters (Modified DH form)
# # ---------------------------------------------------------------------------
# d1, a1 = 0.350, 0.0          # Base to shoulder height
# a2, a3 = 0.540, 0.080        # Arm link lengths
# d4, d5, d6 = 0.400, 0.0, 0.082  # Wrist offsets

# alpha = [-pi/2, 0.0, -pi/2,  pi/2, -pi/2, 0.0]
# a     = [ a1,    a2,   a3,    0.0,   0.0,  0.0]
# d     = [ d1,    0.0,  0.0,   d4,    d5,   d6]

# --- GP7 (standard DH for plotting only) ---
# using your dimensions: d1=0.350, a2=0.540, a3=0.080, d4=0.400, d6=0.082
links = [
    rtb.RevoluteDH(a=0.000, d=0.350,  alpha=-pi/2),
    rtb.RevoluteDH(a=0.540, d=0.000,  alpha= 0.0),
    rtb.RevoluteDH(a=0.080, d=0.000,  alpha= pi/2),
    rtb.RevoluteDH(a=0.000, d=0.400,  alpha=-pi/2),
    rtb.RevoluteDH(a=0.000, d=0.000,  alpha= pi/2),
    rtb.RevoluteDH(a=0.000, d=0.082,  alpha= pi),   # tool flange
]

# deg = np.pi / 180  # Define degree to radian conversion

# links = [
#     rtb.RevoluteDH(a=0.000, d=0.350,  alpha=-np.pi/2, qlim=[-180*deg, 180*deg]),  # J1
#     rtb.RevoluteDH(a=0.540, d=0.000,  alpha= 0.0,     qlim=[-170*deg,  -5*deg]),  # J2 shoulder (lean forward)
#     rtb.RevoluteDH(a=0.080, d=0.000,  alpha= np.pi/2, qlim=[  10*deg, 170*deg]),  # J3 elbow   (bend up)
#     rtb.RevoluteDH(a=0.000, d=0.400,  alpha=-np.pi/2, qlim=[-180*deg, 180*deg]),
#     rtb.RevoluteDH(a=0.000, d=0.000,  alpha= np.pi/2, qlim=[-180*deg, 180*deg]),
#     rtb.RevoluteDH(a=0.000, d=0.082,  alpha= np.pi,   qlim=[-180*deg, 180*deg]),
# ]


robot = rtb.DHRobot(links, name='Motoman GP7 (std DH)')
# was: SE3.Rx(pi/2)
robot.base = SE3(0, 0, 0.0)    # upright in world Z
robot.q = [0, -pi/2, 0, 0, 0, 0]

# Use Modified DH (important!)
# links = [rtb.RevoluteMDH(d=d[i], a=a[i], alpha=alpha[i]) for i in range(6)]

# Build DH robot
robot = rtb.DHRobot(links, name='Motoman GP7')
#robot.base = SE3(0, 0, 0) * SE3.Rx(pi/2)   # maps world Y → Z (upright)

# Set visible home pose before creating visuals
q_home = [0, -pi/2, 0, 0, 0, 0]
robot.q = q_home

# Create cylindrical link geometry (same as ABB)
cyl = CylindricalDHRobotPlot(robot, cylinder_radius=0.04, multicolor=True)
robot_vis = cyl.create_cylinders()

# ---------------------------------------------------------------------------
# Launch Swift environment
# ---------------------------------------------------------------------------
env = swift.Swift()
env.launch(realtime=True)

# Add ground
env.add(Cuboid(scale=[2.0, 2.0, 0.02], pose=SE3(0, 0, -0.01)))

# Add robot
# env.add(robot)
env.add(robot_vis) 

deg = pi/180
dt = 0.012

# Orange tool on the EE
stick = Cylinder(radius=0.02, length=0.25, pose=SE3(), color=[1, 0.5, 0, 1])
env.add(stick)

def step_q(qnext):
    robot.q = qnext
    cyl.update(qnext)
    # attach stick to TCP (tweak 0.12 to sit nicely)
    stick.T = robot.fkine(qnext) * SE3.Trans(0, 0, 0.12)
    env.step(float(dt))

def ik_pose_keepseed(p, yaw, qseed):
    # Tool points straight down (Rx(pi)); yaw spins around world Z
    T = SE3(p[0], p[1], p[2]) * SE3.Rx(pi) * SE3.Rz(yaw)
    # Enforce position + full orientation so it stays upright
    sol = robot.ikine_LM(T, q0=qseed, mask=[1,1,1,1,1,1])
    return sol.q if sol.success else qseed

def go(q_goal, n=90):
    qs = rtb.jtraj(robot.q, q_goal, n).q
    for qk in qs:
        step_q(qk)

# --- params (same as you had) ---
d1 = 0.350
center = np.array([0.45, 0.05, d1 + 0.07])  # stir center
R = 0.07
rev = 3
steps_per_rev = 90
seg_steps = 5
wrist_amp = 15*pi/180
posture_bias = np.array([0, -pi/2, 0, 0, 0, 0])  # upright-ish
beta = 0.08  # small bias weight

# pre-stir hover (upright tool)
q_pre = ik_pose_keepseed(center + np.array([0,0,0.12]), yaw=0.0, qseed=robot.q)
go(q_pre, n=80)

# descend to stir height (upright tool)
q_stir = ik_pose_keepseed(center, yaw=0.0, qseed=q_pre)
go(q_stir, n=60)

# circle with fixed roll/pitch, yaw follows the tangent
q_prev = robot.q.copy()
for k in range(rev * steps_per_rev):
    th = 2*np.pi * k/steps_per_rev
    px = center[0] + R*np.cos(th)
    py = center[1] + R*np.sin(th)
    yaw = th + pi/2  # tool yaw tangential to the path

    q_tgt = ik_pose_keepseed([px, py, center[2]], yaw, q_prev)
    # optional tiny wrist oscillation for stirring look
    q_tgt[-1] = q_tgt[-1] + wrist_amp*np.sin(2*th)
    # light posture pull to keep the robot upright
    q_tgt = (1-beta)*q_tgt + beta*posture_bias

    for qk in rtb.jtraj(q_prev, q_tgt, seg_steps).q:
        step_q(qk)         # updates cylinders + stick
    q_prev = q_tgt

# retract upright
q_up = ik_pose_keepseed(center + np.array([0,0,0.12]), yaw=0.0, qseed=q_prev)
go(q_up, n=60)


print("✅ Stirring sequence complete")
env.hold()
