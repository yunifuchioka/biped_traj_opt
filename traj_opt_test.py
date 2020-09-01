"""
Stand-alone trajectory optimization test for the LIP-ZMP tracking problem described in
Kajita et al 2003 "Biped walking pattern generation by using preview control of zero-moment
point"
"""

import numpy as np
import casadi as ca
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

tf = 20.0 # final time
N = 200 # number of finite elements
t_traj = np.linspace(0, tf, 2*N+1) # collocated time

# com height
zc = 1.45
# x, y dimensions of feet
foot_size = np.array([0.25, 0.15])
# y-width of torso, z-height of torso (from hip joint to COM)
torso_size = np.array([0.3, 0.5])
# length of thigh and calves respectively
leg_length = np.array([0.5, 0.5])

nx = 6 # dimension of state vector
nu = 2 # dimension of control vector
ny = 2 # dimension of output vector
g = 9.81 # gravity
com_accel_max = 10.0 # maximum COM acceleration
Q = np.array([500, 500]) # output error cost weight
Qf = np.array([50, 50]) #final output error cost weight
R = np.array([0.1, 0.1]) #input regulation cost weight

# footstep plan generation #############################################################

num_steps = 30 # total number of footsteps during trajectory
# forward step length, width between footsteps, maximum height during step
stride = np.array([0.2, 0.3, 0.2])

# matrix defining footstep plan
footstep_plan = np.array([ \
    np.linspace(0, tf, num_steps), # time
    #np.linspace(0.0, num_steps*stride[0], num_steps), # x
    np.hstack((np.zeros(2),
        np.linspace(0.0, (num_steps-4)*stride[0], num_steps-4),
        np.full(2, (num_steps-4)*stride[0]),
        )),
    np.array([-stride[1]/2 + stride[1]*(i%2) for i in range(num_steps)]) #y
    ])

def generate_zmp_spline(foostep_plan, num_points):
    def _zmp_func(t):
        t_idx = np.where(footstep_plan[0]<=t)[0][-1]
        return np.array([ \
            footstep_plan[1, t_idx],
            footstep_plan[2, t_idx]])

    t = np.linspace(0, tf, num_points)
    zmp = np.array([_zmp_func(i) for i in t]).T
    return CubicSpline(t, zmp, axis=1)

# desired zmp trajectory function (which is a cubic spline interpolant)
y_des_t = generate_zmp_spline(foostep_plan=footstep_plan, num_points=num_steps*5)

# NLP definition ########################################################################

lip_opti = ca.Opti()

# decision variables
X = lip_opti.variable(nx, 2*N+1)
U = lip_opti.variable(nu, 2*N+1)

# parameters
self_x_init = lip_opti.parameter(nx)
self_y_des = lip_opti.parameter(ny, 2*N+1)

# dynamics
# symbolic variables used to derive equations of motion
x = ca.SX.sym('x', nx) #state
u = ca.SX.sym('u', nu) #control
# eq 12 in Kajita et al 2003, expanded to include both x and y components
A = ca.SX(np.array([ \
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]))
B = ca.SX(np.array([ \
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    ]))
C = ca.SX(np.array([ \
    [1.0, 0.0, 0.0, 0.0, -zc/g, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, -zc/g]
    ]))
f = ca.Function('f', [x,u], [A@x+B@u])
y = ca.Function('y', [x], [C@x])

# objective
# simpson quadrature coefficients, to be used to compute integrals
simp = np.empty((1,2*N+1))
simp[0,::2] = 2
simp[0,1::2] = 4
simp[0,0], simp[0,-1]  = 1, 1

# cost = finite horizon LQR cost computed with simpson quadrature
# here, regulation is performed on the COM accel and not input
J = 0.0
for k in range(2):
    J += R[k] * ca.dot(simp, X[4+k,:]*X[4+k,:])

for j in range(ny):
    for i in range(2*N+1):
        J += Q[j] * simp[0][i] \
            * (y(X[:,i])[j]-self_y_des[j,i])*(y(X[:,i])[j]-self_y_des[j,i])
    J += Qf[j] * (y(X[:,-1])[j]-self_y_des[j,-1])*(y(X[:,-1])[j]-self_y_des[j,-1])

lip_opti.minimize(J)

# COM accel constraint
lip_opti.subject_to(lip_opti.bounded( \
    np.full(X[4:,:].shape, -com_accel_max), 
    X[4:,:], 
    np.full(X[4:,:].shape, com_accel_max)
    ))

# initial condition constraint
lip_opti.subject_to(X[:,0] == self_x_init)

for i in range(2*N+1):
    if i%2 != 0:
        # for each finite element:
        x_left, x_mid, x_right = X[:,i-1], X[:,i], X[:,i+1]
        u_left, u_mid, u_right = U[:,i-1], U[:,i], U[:,i+1]
        f_left, f_mid, f_right = f(x_left,u_left), f(x_mid,u_mid), f(x_right,u_right)

        # interpolation constraints
        lip_opti.subject_to( \
            # equation (6.11) in Kelly 2017
            x_mid == (x_left+x_right)/2.0 + tf/N*(f_left-f_right)/8.0)

        # collocation constraints
        lip_opti.subject_to( \
            # equation (6.12) in Kelly 2017
            tf/N*(f_left+4*f_mid+f_right)/6.0 == x_right-x_left)

# below is done at runtime ###############################################################

x_init = np.array([y_des_t(0)[0], y_des_t(0)[1], 0, 0, 0, 0])
y_des = y_des_t(t_traj)

lip_opti.set_value(self_x_init, x_init)
lip_opti.set_value(self_y_des, y_des)

# guess COM trajectory as desired ZMP trajectory
X_guess = np.vstack(( \
    y_des_t(t_traj),
    y_des_t(t_traj,1),
    y_des_t(t_traj,2)
    ))

lip_opti.set_initial(X, X_guess)
lip_opti.set_initial(U, np.zeros(U.shape))

p_opts = {}
s_opts = {'print_level': 5}
lip_opti.solver('ipopt', p_opts, s_opts)
sol = lip_opti.solve()

sol_x = np.array(sol.value(X))
sol_y = np.array(y(sol_x))

#com and zmp trajectory in cartesian space at each collocation point
com_traj = np.vstack((sol_x[:2,:], np.full(t_traj.shape, zc)))
zmp_traj = np.vstack((sol_y, np.zeros(t_traj.shape)))

'''
# plot zmp and com as function of time
t_fine = np.linspace(0, tf, 1000)
zmp_fine = y_des_t(t_fine)
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(t_fine, y_des_t(t_fine)[0]);
ax[0].plot(t_traj, zmp_traj[0,:], 'o');
ax[0].plot(t_traj, com_traj[0,:], 'o');
ax[0].set_title('Desired ZMP_x vs time')
ax[1].plot(t_fine, y_des_t(t_fine)[1]);
ax[1].plot(t_traj, np.squeeze(zmp_traj[1,:]), 'o');
ax[1].plot(t_traj, com_traj[1,:], 'o');
ax[1].set_title('Desired ZMP_y vs time')

# plot 2D footstep plan, ZMP, and COM trajectory
plt.figure()
plt.plot(zmp_fine[0], zmp_fine[1])
plt.plot(zmp_traj[0,:], zmp_traj[1,:], 'o')
plt.plot(com_traj[0,:], com_traj[1,:], 'o')
ax = plt.gca()
for i in range(footstep_plan.shape[1]):
    rect_x_points = np.array([ \
        footstep_plan[1,i]-foot_size[0]/2,
        footstep_plan[1,i]+foot_size[0]/2,
        footstep_plan[1,i]+foot_size[0]/2,
        footstep_plan[1,i]-foot_size[0]/2,
        footstep_plan[1,i]-foot_size[0]/2
        ])
    rect_y_points = np.array([ \
        footstep_plan[2,i]-foot_size[1]/2,
        footstep_plan[2,i]-foot_size[1]/2,
        footstep_plan[2,i]+foot_size[1]/2,
        footstep_plan[2,i]+foot_size[1]/2,
        footstep_plan[2,i]-foot_size[1]/2
        ])
    plt.plot(rect_x_points, rect_y_points, 'r')
ax.axis('equal')
ax.set_title('Footstep plan')

plt.show()
'''

# foot trajectory generation ###########################################################

right_traj_points, left_traj_points = None, None

# populate foot trajectory points for the corresponding stance phase
for t in t_traj:
    t_idx = np.where(footstep_plan[0]<=t)[0][-1]
    if right_traj_points is None or left_traj_points is None:
        right_traj_points = np.array([t, footstep_plan[1, t_idx], 0])[:,None]
        left_traj_points = np.array([t, footstep_plan[1, 0], 0])[:,None]
    else:
        if t_idx % 2 == 0:
            right_traj_points = np.hstack((right_traj_points,
                np.array([t, footstep_plan[1, t_idx], 0])[:,None]))
        else:
            left_traj_points = np.hstack((left_traj_points,
                np.array([t, footstep_plan[1, t_idx], 0])[:,None]))

# add a point in the middle of swing phase to raise the foot
for i, t in enumerate(footstep_plan[0,:]):
    if i != 0:
        prev_t = footstep_plan[0,i-1]
        t_mid = 0.5*(prev_t + t)
        if i % 2 == 0:
            right_traj_points = np.hstack((right_traj_points,
                np.array([t_mid, footstep_plan[1, i-1], stride[2]])[:,None]))
        else:
            left_traj_points = np.hstack((left_traj_points,
                np.array([t_mid, footstep_plan[1, i-1], stride[2]])[:,None]))

# generate cubic spline trajectory from interpolation points
right_traj_points = right_traj_points[:, right_traj_points[0].argsort()]
left_traj_points = left_traj_points[:, left_traj_points[0].argsort()]
right_traj_spline = CubicSpline(right_traj_points[0,:], right_traj_points[1:,:], axis=1)
left_traj_spline = CubicSpline(left_traj_points[0,:], left_traj_points[1:,:], axis=1)

# right and left foot trajectory in cartesian space at each collocation point
right_traj = np.vstack((right_traj_spline(t_traj)[0,:], 
    np.full(t_traj.shape, -torso_size[0]/2),
    right_traj_spline(t_traj)[1,:]))
left_traj = np.vstack((left_traj_spline(t_traj)[0,:],
    np.full(t_traj.shape, torso_size[0]/2),
    left_traj_spline(t_traj)[1,:]))

feet_traj_rel = np.vstack((right_traj-com_traj, left_traj-com_traj))

'''
# plot foot trajectory
t_fine = np.linspace(0, tf, 1000)
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(right_traj_points[0,:],right_traj_points[1,:], 'o')
ax[0].plot(t_fine, right_traj_spline(t_fine)[0])
ax[0].plot(left_traj_points[0,:],left_traj_points[1,:], 'o')
ax[0].plot(t_fine, left_traj_spline(t_fine)[0])
ax[1].plot(right_traj_points[0,:],right_traj_points[2,:], 'o')
ax[1].plot(t_fine, right_traj_spline(t_fine)[1])
ax[1].plot(left_traj_points[0,:],left_traj_points[2,:], 'o')
ax[1].plot(t_fine, left_traj_spline(t_fine)[1])
plt.show()
'''

# inverse kinematics #####################################################################

# basis vectors
i0 = np.array([1.0, 0.0, 0.0])[:,None]
j0 = np.array([0.0, 1.0, 0.0])[:,None]
k0 = np.array([0.0, 0.0, 1.0])[:,None]

# Denavit Hartenberg parameters for the leg
th_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
a = np.array([0.0, leg_length[0], leg_length[1], 0.0, leg_length[0], leg_length[1]])
alpha = np.array([-np.pi/2, 0.0, np.pi/2,-np.pi/2, 0.0, np.pi/2])

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
    np.hstack((R(np.pi/2, j0), np.array([0, -torso_size[0]/2, -torso_size[1]])[:,None])),
    [0.0, 0.0, 0.0, 1.0]
    ))

T_base_left = np.vstack(( \
    np.hstack((R(np.pi/2, j0), np.array([0, torso_size[0]/2, -torso_size[1]])[:,None])),
    [0.0, 0.0, 0.0, 1.0]
    ))

invkin = ca.Opti()

o_des = invkin.parameter(6, 2*N+1) # desired foot trajectory
Q = invkin.variable(6, 2*N+1) # joint angle trajectory

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
invkin.minimize(J)

# knee joint constraint
invkin.subject_to(invkin.bounded( \
    np.full(Q[2,:].shape, 0), 
    Q[2,:], 
    np.full(Q[2,:].shape, np.pi)
    ))
invkin.subject_to(invkin.bounded( \
    np.full(Q[5,:].shape, 0), 
    Q[5,:], 
    np.full(Q[5,:].shape, np.pi)
))

invkin.set_value(o_des, feet_traj_rel)

p_opts = {}
s_opts = {'print_level': 5}
invkin.solver('ipopt', p_opts, s_opts)
invkin_sol = invkin.solve()

sol_Q = np.array(invkin_sol.value(Q))

# forkin on invkin solution to get joint locations #######################################

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

joint_x_points = np.vstack((o3[0,:], o2[0,:], o1[0,:], np.zeros(t_traj.shape), 
    o4[0,:], o5[0,:], o6[0,:]))
joint_y_points = np.vstack((o3[1,:], o2[1,:], o1[1,:], np.zeros(t_traj.shape), 
    o4[1,:], o5[1,:], o6[1,:]))
joint_z_points = np.vstack((o3[2,:], o2[2,:], o1[2,:], np.zeros(t_traj.shape), 
    o4[2,:], o5[2,:], o6[2,:]))

joint_x_points += com_traj[0,:]
joint_y_points += com_traj[1,:]
joint_z_points += com_traj[2,:]

# plot ###################################################################################

num_draw_footstep = 2
num_draw_traj = 20

# animate 3D LIP
anim_fig = plt.figure(figsize=(12, 12))
axis_lim = 1.5
ax = Axes3D(anim_fig)
lines = [plt.plot([], [])[0] for _ in range(6 + 2*num_draw_footstep+1)]

def animate(i):
    # draw biped
    lines[-1].set_data(joint_x_points[:,i], joint_y_points[:,i])
    lines[-1].set_3d_properties(joint_z_points[:,i])
    lines[-2].set_data(np.array([joint_x_points[2,i],joint_x_points[4,i]]),
        np.array([joint_y_points[2,i],joint_y_points[4,i]]))
    lines[-2].set_3d_properties(np.array([joint_z_points[2,i],joint_z_points[4,i]]))

    # draw LIP
    lines[0].set_data(np.array([zmp_traj[0,i], com_traj[0,i]]), 
        np.array([zmp_traj[1,i], com_traj[1,i]]))
    lines[0].set_3d_properties(np.array([zmp_traj[2,i], com_traj[2,i]]))

    if i < num_draw_traj:
        traj_start = 0
    else:
        traj_start = i-num_draw_traj

    # draw com trajectory
    lines[1].set_data(com_traj[0,traj_start:i], com_traj[1,traj_start:i])
    lines[1].set_3d_properties(com_traj[2,traj_start:i])
    # draw feet trajectory
    lines[2].set_data(right_traj[0,traj_start:i], right_traj[1,traj_start:i])
    lines[2].set_3d_properties(right_traj[2,traj_start:i])
    lines[3].set_data(left_traj[0,traj_start:i], left_traj[1,traj_start:i])
    lines[3].set_3d_properties(left_traj[2,traj_start:i])

    foot_idx = np.histogram(t_traj[i], bins=footstep_plan[0])[0].argmax()
    if foot_idx <= num_draw_footstep:
        footstep_draw = np.vstack(( \
            footstep_plan[1,:2*num_draw_footstep+1],
            footstep_plan[2,:2*num_draw_footstep+1]))
    elif foot_idx+num_draw_footstep+1 >= num_steps:
        footstep_draw = np.vstack(( \
            footstep_plan[1,num_steps-1-2*num_draw_footstep:],
            footstep_plan[2,num_steps-1-2*num_draw_footstep:]))
    else:
        footstep_draw = np.vstack(( \
            footstep_plan[1,foot_idx-num_draw_footstep:foot_idx+num_draw_footstep+1],
            footstep_plan[2,foot_idx-num_draw_footstep:foot_idx+num_draw_footstep+1]))

    for j in range(2*num_draw_footstep+1):
        rect_x_points = np.array([ \
            footstep_draw[0,j]-foot_size[0]/2,
            footstep_draw[0,j]+foot_size[0]/2,
            footstep_draw[0,j]+foot_size[0]/2,
            footstep_draw[0,j]-foot_size[0]/2,
            footstep_draw[0,j]-foot_size[0]/2
            ])
        rect_y_points = np.array([ \
            footstep_draw[1,j]-foot_size[1]/2,
            footstep_draw[1,j]-foot_size[1]/2,
            footstep_draw[1,j]+foot_size[1]/2,
            footstep_draw[1,j]+foot_size[1]/2,
            footstep_draw[1,j]-foot_size[1]/2
            ])
        rect_z_points = np.zeros(5)
        lines[4+j].set_data(rect_x_points, rect_y_points)
        lines[4+j].set_3d_properties(rect_z_points)

    ax.view_init(azim=i/2)
    ax.set_xlim3d([com_traj[0,i]-axis_lim/2, com_traj[0,i]+axis_lim/2])
    ax.set_ylim3d([-axis_lim/2, axis_lim/2])
    ax.set_zlim3d([0.0, axis_lim])

    return lines


for line in lines[-2:]:
    line.set_color('b')
    line.set_linewidth(5)
    line.set_marker('o')
    line.set_markeredgewidth(7)
lines[0].set_color('#ff7f0e')
lines[0].set_linestyle('--')
lines[0].set_linewidth(5)
lines[0].set_marker('o')
lines[0].set_markeredgewidth(7)
lines[1].set_color('#2ca02c')
lines[2].set_color('#9467bd')
lines[3].set_color('#9467bd')
for j in range(2*num_draw_footstep+1):
    lines[4+j].set_color('r')

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=2*N+1, 
    interval=tf*1000/(2*N+1), repeat=True, blit=False)

'''
Writer = animation.writers['ffmpeg']
writer = Writer(fps=int((2*N+1)/tf), metadata=dict(artist='Me'), bitrate=1000)
anim.save('LIP_correct_timing' + '.mp4', writer=writer)
'''
plt.show()