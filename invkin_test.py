
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
plt.style.use('seaborn')

i0 = np.array([1.0, 0.0, 0.0])[:,None]
j0 = np.array([0.0, 1.0, 0.0])[:,None]
k0 = np.array([0.0, 0.0, 1.0])[:,None]

tf = 10.0
N = 100
th_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
a = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.5])
alpha = np.array([-np.pi/2, 0.0, np.pi/2,-np.pi/2, 0.0, np.pi/2])
hip_size = np.array([0.3, 0.3])

def derive_funcs():
    s = ca.SX.sym('s', 3)
    th = ca.SX.sym('th')
    th_offset = ca.SX.sym('th_offset')
    d = ca.SX.sym('d')
    a = ca.SX.sym('a')
    alpha = ca.SX.sym('alpha')

    skew_sym = ca.SX(np.array([ \
            [0, -s[2], s[1]],
            [s[2], 0, -s[0]],
            [-s[1], s[0], 0]
        ]))
    R_sym = ca.SX.eye(3) + ca.sin(th)*skew_sym + (1-ca.cos(th))*skew_sym@skew_sym
    R = ca.Function('R', [th, s], [R_sym])

    T_sym = ca.vertcat( \
        ca.horzcat(R(th,k0)@R(alpha,i0), R(th,k0)@(a*i0) + d*k0),
        np.array([0.0, 0.0, 0.0, 1.0])[None,:]
        )
    T = ca.Function('T', [th, th_offset, d, a, alpha], [T_sym])

    return R, T

R, T = derive_funcs()

T_base_right = np.vstack(( \
    np.hstack((R(np.pi/2, j0), np.array([0, -hip_size[0]/2, -hip_size[1]])[:,None])),
    [0.0, 0.0, 0.0, 1.0]
    ))

T_base_left = np.vstack(( \
    np.hstack((R(np.pi/2, j0), np.array([0, hip_size[0]/2, -hip_size[1]])[:,None])),
    [0.0, 0.0, 0.0, 1.0]
    ))

opti = ca.Opti()

o_des = opti.parameter(6, 2*N+1)

Q = opti.variable(6, 2*N+1)

# objective
J = 0
for i in range(2*N+1):
    T_right_foot = T_base_right
    T_left_foot = T_base_left
    for j in range(0,3):
        T_right_foot = T_right_foot@T(Q[j,i], th_offset[j], d[j], a[j], alpha[j])
    for j in range(3,6):
        T_left_foot = T_left_foot@T(Q[j,i], th_offset[j], d[j], a[j], alpha[j])
    o = ca.vertcat(T_right_foot[0:3,3], T_left_foot[0:3,3])
    J += ca.dot(o-o_des[:,i], o-o_des[:,i])

    if i != 0:
        J += 10 * ca.dot(o_des[:,i]-o_des[:,i-1], o_des[:,i]-o_des[:,i-1])
opti.minimize(J)

opti.subject_to(opti.bounded( \
    np.full(Q[2,:].shape, 0), 
    Q[2,:], 
    np.full(Q[2,:].shape, np.pi)
    ))

opti.subject_to(opti.bounded( \
    np.full(Q[5,:].shape, 0), 
    Q[5,:], 
    np.full(Q[5,:].shape, np.pi)
))

t = np.linspace(0, tf, 2*N+1)
o_des_num = np.array([ \
    0.2*np.sin(t),
    #np.full(t.shape, -hip_size[0]/2 - 0.1),
    np.zeros(t.shape),
    np.full(t.shape, -1),
    0.2*np.sin(t+np.pi),
    np.full(t.shape, hip_size[0]/2),
    np.full(t.shape, -1)
    ])

opti.set_value(o_des, o_des_num)

p_opts = {}
s_opts = {'print_level': 5}
opti.solver('ipopt', p_opts, s_opts)
sol = opti.solve()

sol_Q = np.array(sol.value(Q))

# visualize forkin ######################################################################

o1, o2, o3, o4, o5, o6 = None, None, None, None, None, None

for i in range(2*N+1):
    T1 = T_base_right@T(sol_Q[0,i], th_offset[0], d[0], a[0], alpha[0])
    T2 = T1@T(sol_Q[1,i], th_offset[1], d[1], a[1], alpha[1])
    T3 = T2@T(sol_Q[2,i], th_offset[2], d[2], a[2], alpha[2])
    T4 = T_base_left@T(sol_Q[3,i], th_offset[3], d[3], a[3], alpha[3])
    T5 = T4@T(sol_Q[4,i], th_offset[4], d[4], a[4], alpha[4])
    T6 = T5@T(sol_Q[5,i], th_offset[5], d[5], a[5], alpha[5])

    o1_i = T1[0:3,3:]
    o2_i = T2[0:3,3:]
    o3_i = T3[0:3,3:]
    o4_i = T4[0:3,3:]
    o5_i = T5[0:3,3:]
    o6_i = T6[0:3,3:]

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

ax.set_xlim3d(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylim3d(-1.5, 1.5)
ax.set_ylabel('y')
ax.set_zlim3d(-1, 0.5)
ax.set_zlabel('z')

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=N, 
    interval=tf*1000/N, repeat=True, blit=True)

plt.show()