# test forward kinematics

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
plt.style.use('seaborn')

i0 = np.array([1, 0, 0])[:, None];
j0 = np.array([0, 1, 0])[:, None];
k0 = np.array([0, 0, 1])[:, None];

def skew(x):
    x = np.squeeze(x)
    return np.array([ \
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def R(theta, x):
    return np.eye(3) + np.sin(theta)*skew(x) + (1-np.cos(theta))*skew(x)@skew(x)

theta_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
a = np.array([0.0, 2.0, 2.0, 0.0, 2.0, 2.0])
alpha = np.array([-np.pi/2, 0.0, np.pi/2,-np.pi/2, 0.0, np.pi/2])
hip_size = np.array([2.0, 1.0])

def forkin(theta, theta_offset, d, a, alpha):
    def DH_homog(theta, theta_offset, d, a, alpha):
        return np.vstack(( \
            np.hstack((R(theta,k0)@R(alpha,i0), R(theta,k0)@(a*i0) + d*k0)),
            [0.0, 0.0, 0.0, 1.0]
            ))

    T = np.eye(4)
    for i in range(max(theta.shape)):
        T = T@DH_homog(theta[i], theta_offset[i], d[i], a[i], alpha[i])
    
    return T

N = 100
tf = 10.0
t = np.linspace(0, tf, N)
theta = np.array([ \
    0.2*np.sin(2*t),
    np.sin(t),
    np.full(t.shape, 1) + np.sin(t),
    0.2*np.sin(2*t+np.pi),
    np.sin(t+np.pi),
    np.full(t.shape, 1) + np.sin(t+np.pi)
    ])

T_base_right = np.vstack(( \
    np.hstack((expm(np.pi/2*skew(j0)), np.array([0, -hip_size[0]/2, -hip_size[1]])[:,None])),
    [0.0, 0.0, 0.0, 1.0]
    ))

T_base_left = np.vstack(( \
    np.hstack((expm(np.pi/2*skew(j0)), np.array([0, hip_size[0]/2, -hip_size[1]])[:,None])),
    [0.0, 0.0, 0.0, 1.0]
    ))

o1, o2, o3, o4, o5, o6 = None, None, None, None, None, None
for i in range(N):
    o1_i = (T_base_right@forkin(theta[0:1, i], theta_offset, d, a, alpha))[0:3,3:]
    o2_i = (T_base_right@forkin(theta[0:2, i], theta_offset, d, a, alpha))[0:3,3:]
    o3_i = (T_base_right@forkin(theta[0:3, i], theta_offset, d, a, alpha))[0:3,3:]
    o4_i = (T_base_left@forkin(theta[3:4, i], theta_offset, d, a, alpha))[0:3,3:]
    o5_i = (T_base_left@forkin(theta[3:5, i], theta_offset, d, a, alpha))[0:3,3:]
    o6_i = (T_base_left@forkin(theta[3:6, i], theta_offset, d, a, alpha))[0:3,3:]

    if i == 0:
        o1, o2, o3, o4, o5, o6 = o1_i, o2_i, o3_i, o4_i, o5_i, o6_i
    else:
        o1 = np.hstack((o1, o1_i))
        o2 = np.hstack((o2, o2_i))
        o3 = np.hstack((o3, o3_i))
        o4 = np.hstack((o4, o4_i))
        o5 = np.hstack((o5, o5_i))
        o6 = np.hstack((o6, o6_i))

x_points = np.array([o3[0,:], o2[0,:], o1[0,:], np.zeros(o1[0,:].shape), 
    o4[0,:], o5[0,:], o6[0,:]])
y_points = np.array([o3[1,:], o2[1,:], o1[1,:], np.zeros(o1[1,:].shape), 
    o4[1,:], o5[1,:], o6[1,:]])
z_points = np.array([o3[2,:], o2[2,:], o1[2,:], np.zeros(o1[2,:].shape), 
    o4[2,:], o5[2,:], o6[2,:]])

anim_fig = plt.figure(figsize=(12, 12))
axis_lim = 1.5
ax = Axes3D(anim_fig)
lines = [plt.plot([], [])[0] for _ in range(1)]

def animate(i):

    lines[0].set_data(x_points[:,i], y_points[:,i])
    lines[0].set_3d_properties(z_points[:,i])

    for line in lines:
        line.set_color('b')
        line.set_linewidth(5)
        line.set_marker('o')
        line.set_markeredgewidth(7)

    return lines

ax.set_xlim3d(-4, 4)
ax.set_xlabel('x')
ax.set_ylim3d(-4, 4)
ax.set_ylabel('y')
ax.set_zlim3d(-4, 4)
ax.set_zlabel('z')

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=N, 
    interval=tf*1000/N, repeat=True, blit=True)

plt.show()