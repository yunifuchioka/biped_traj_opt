# test forward kinematics for biped

import numpy as np
from numpy import pi
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
plt.style.use('seaborn')

theta_offset =  np.array([0.0, -pi/2, 0.0, 0.0, 0.0, pi/2])
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
a = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
alpha = np.array([-pi/2, -pi/2, 0.0, 0.0, pi/2, pi/2])

torso_size = np.array([0, 0.3, 0.5])
foot_size = np.array([0.2, 0.1, 0])

i0 = np.array([1, 0, 0])[:, None];
j0 = np.array([0, 1, 0])[:, None];
k0 = np.array([0, 0, 1])[:, None];

def skew(x): # cross product matrix
    x = np.squeeze(x)
    return np.array([ \
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def R(theta, x): # rotation matrix. Assumed that x is a unit vector
    return np.eye(3) + np.sin(theta)*skew(x) + (1-np.cos(theta))*skew(x)@skew(x)

def DH_homog(theta, theta_offset, d, a, alpha):
    return np.vstack(( \
        np.hstack((R(theta+theta_offset,k0)@R(alpha,i0), 
            R(theta+theta_offset,k0)@(a*i0) + d*k0)),
        [0.0, 0.0, 0.0, 1.0]
        ))

def T(theta, theta_offset, d, a, alpha):
    T = np.eye(4)
    for i in range(max(theta.shape)):
        T = T@DH_homog(theta[i], theta_offset[i], d[i], a[i], alpha[i])
    
    return T

def legforkin(theta, theta_offset, d, a, alpha):
    T1 = T(theta[0:1], theta_offset, d, a, alpha)
    T2 = T(theta[0:2], theta_offset, d, a, alpha)
    T3 = T(theta[0:3], theta_offset, d, a, alpha)
    T4 = T(theta[0:4], theta_offset, d, a, alpha)
    T5 = T(theta[0:5], theta_offset, d, a, alpha)
    T6 = T(theta[0:6], theta_offset, d, a, alpha)

    Tc1 = T6 @ np.array([ \
        [0, 1, 0, -foot_size[1]/2],
        [1, 0, 0, foot_size[0]/2],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
        ])

    Tc2 = T6 @ np.array([ \
        [0, 1, 0, foot_size[1]/2],
        [1, 0, 0, foot_size[0]/2],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
        ])

    Tc3 = T6 @ np.array([ \
        [0, 1, 0, -foot_size[1]/2],
        [1, 0, 0, -foot_size[0]/2],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
        ])

    Tc4 = T6 @ np.array([ \
        [0, 1, 0, foot_size[1]/2],
        [1, 0, 0, -foot_size[0]/2],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
        ])

    return { \
        "1": T1,
        "2": T2,
        "3": T3,
        "4": T4,
        "5": T5,
        "6": T6,
        "c1": Tc1,
        "c2": Tc2,
        "c3": Tc3,
        "c4": Tc4
    }

#np.set_printoptions(suppress=True) 
#import ipdb; ipdb.set_trace()

T_base_right = DH_homog(pi/2, 0, -torso_size[2], -torso_size[1]/2, pi)
T_base_left = DH_homog(pi/2, 0, -torso_size[2], torso_size[1]/2, pi)

N = 100
tf = 10.0
t = np.linspace(0, tf, N)
q = np.array([ \
    0.0*t,
    0.0*t,
    pi/8*np.sin(t),
    pi/4 + pi/8*np.sin(t),
    0.0*t,
    0.0*t,

    0.0*t,
    0.0*t,
    pi/8*np.sin(t+pi),
    pi/4 + pi/8*np.sin(t+pi),
    0.0*t,
    0.0*t
    ])

or0, or3, or6, orc1, orc2, orc3, orc4 = \
    None, None, None, None, None, None, None

for i in range(N):

    Tr_raw_dict = legforkin(q[:6, i], theta_offset, d, a, alpha)
    Tl_raw_dict = legforkin(q[6:, i], theta_offset, d, a, alpha)

    or_dict = {}
    for k, v in Tr_raw_dict.items():
        or_dict[k] = (T_base_right@v)[0:3,3:]
    ol_dict = {}
    for k, v in Tl_raw_dict.items():
        ol_dict[k] = (T_base_left@v)[0:3,3:]

    or0_i, or3_i, or6_i, orc1_i, orc2_i, orc3_i, orc4_i = \
        or_dict['1'], or_dict['3'], or_dict['6'], \
        or_dict['c1'], or_dict['c2'], or_dict['c3'], or_dict['c4']
    ol0_i, ol3_i, ol6_i, olc1_i, olc2_i, olc3_i, olc4_i = \
        ol_dict['1'], ol_dict['3'], ol_dict['6'], \
        ol_dict['c1'], ol_dict['c2'], ol_dict['c3'], ol_dict['c4']

    if i == 0:
        or0, or3, or6, orc1, orc2, orc3, orc4 = \
            or0_i, or3_i, or6_i, orc1_i, orc2_i, orc3_i, orc4_i
        ol0, ol3, ol6, olc1, olc2, olc3, olc4 = \
            ol0_i, ol3_i, ol6_i, olc1_i, olc2_i, olc3_i, olc4_i
    else:
        or0 = np.hstack((or0, or0_i))
        or3 = np.hstack((or3, or3_i))
        or6 = np.hstack((or6, or6_i))
        orc1 = np.hstack((orc1, orc1_i))
        orc2 = np.hstack((orc2, orc2_i))
        orc3 = np.hstack((orc3, orc3_i))
        orc4 = np.hstack((orc4, orc4_i))
        ol0 = np.hstack((ol0, ol0_i))
        ol3 = np.hstack((ol3, ol3_i))
        ol6 = np.hstack((ol6, ol6_i))
        olc1 = np.hstack((olc1, olc1_i))
        olc2 = np.hstack((olc2, olc2_i))
        olc3 = np.hstack((olc3, olc3_i))
        olc4 = np.hstack((olc4, olc4_i))

leg_r_x_points = np.array([or0[0,:], or3[0,:], or6[0,:]])
leg_r_y_points = np.array([or0[1,:], or3[1,:], or6[1,:]])
leg_r_z_points = np.array([or0[2,:], or3[2,:], or6[2,:]])
foot_r_x_points = np.array([orc1[0,:], orc2[0,:], orc4[0,:], orc3[0,:], orc1[0,:]])
foot_r_y_points = np.array([orc1[1,:], orc2[1,:], orc4[1,:], orc3[1,:], orc1[1,:]])
foot_r_z_points = np.array([orc1[2,:], orc2[2,:], orc4[2,:], orc3[2,:], orc1[2,:]])

leg_l_x_points = np.array([ol0[0,:], ol3[0,:], ol6[0,:]])
leg_l_y_points = np.array([ol0[1,:], ol3[1,:], ol6[1,:]])
leg_l_z_points = np.array([ol0[2,:], ol3[2,:], ol6[2,:]])
foot_l_x_points = np.array([olc1[0,:], olc2[0,:], olc4[0,:], olc3[0,:], olc1[0,:]])
foot_l_y_points = np.array([olc1[1,:], olc2[1,:], olc4[1,:], olc3[1,:], olc1[1,:]])
foot_l_z_points = np.array([olc1[2,:], olc2[2,:], olc4[2,:], olc3[2,:], olc1[2,:]])

torso_x_points = np.array([0*or0[0,:], or0[0,:], ol0[0,:], 0*or0[0,:]])
torso_y_points = np.array([0*or0[0,:], or0[1,:], ol0[1,:], 0*or0[0,:]])
torso_z_points = np.array([0*or0[0,:], or0[2,:], ol0[2,:], 0*or0[0,:]])

anim_fig = plt.figure(figsize=(12, 12))
axis_lim = 1.5
ax = Axes3D(anim_fig)
lines = [plt.plot([], [])[0] for _ in range(5)]

def animate(i):

    lines[0].set_data(foot_r_x_points[:,i], foot_r_y_points[:,i])
    lines[0].set_3d_properties(foot_r_z_points[:,i])
    lines[1].set_data(foot_l_x_points[:,i], foot_l_y_points[:,i])
    lines[1].set_3d_properties(foot_l_z_points[:,i])

    lines[2].set_data(leg_r_x_points[:,i], leg_r_y_points[:,i])
    lines[2].set_3d_properties(leg_r_z_points[:,i])
    lines[3].set_data(leg_l_x_points[:,i], leg_l_y_points[:,i])
    lines[3].set_3d_properties(leg_l_z_points[:,i])

    lines[4].set_data(torso_x_points[:,i], torso_y_points[:,i])
    lines[4].set_3d_properties(torso_z_points[:,i])

    for line in lines[0:2]:
        line.set_color('g')
        line.set_linewidth(5)
        line.set_marker('o')
        line.set_markeredgewidth(7)

    for line in lines[2:]:
        line.set_color('b')
        line.set_linewidth(5)
        line.set_marker('o')
        line.set_markeredgewidth(7)

    return lines

ax.view_init(azim=45)
ax.set_xlim3d(-1, 1)
ax.set_xlabel('x')
ax.set_ylim3d(-1, 1)
ax.set_ylabel('y')
ax.set_zlim3d(-2, 0)
ax.set_zlabel('z')

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=N, 
    interval=tf*1000/N, repeat=True, blit=True)

plt.show()