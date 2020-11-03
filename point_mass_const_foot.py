import numpy as np
from numpy import pi
import casadi as ca

from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
plt.style.use('seaborn')

nq = 12 # dimension of joint configuration vector
nr = 3 # dimension of centroid configuration vector
nF = 3*8 # dimension of force vector

leg_len = np.array([0.5, 0.5]) # thigh and calf lengh respectively
torso_size = np.array([0, 0.3, 0.5]) # x,y,z length of torso
foot_size = np.array([0.2, 0.1, 0]) # x,y,z length of feet

# Denavit Hartenberg parameters for a leg
theta_offset =  np.array([0.0, -pi/2, 0.0, 0.0, 0.0, pi/2])
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
a = np.array([0.0, 0.0, leg_len[0], leg_len[1], 0.0, 0.0])
alpha = np.array([-pi/2, -pi/2, 0.0, 0.0, pi/2, pi/2])

m = 10 # mass of torso
g = np.array([0, 0, -9.81]) # gravitational acceleration
mu = 0.7 #friciton coefficient

N = 100 # number of hermite-simpson finite elements. Total # of points = 2*N+1
tf = 20.0 # final time of interval
t = np.linspace(0,tf, 2*N+1) # discretized time

# initial joint angle
q_init = np.array([ \
    0, 0, -0.05, 0.1, -0.05, 0,
    0, 0, -0.05, 0.1, -0.05, 0
    ])

# joint angle limits
q_bounds = np.array([ \
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([1E-6, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([1E-6, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi])
    ])

# objective cost weights
Qr = np.array([1000, 1000, 1000])
Qrd = np.array([1, 1, 1])
Qqd = np.full((nq*2), 1)
RF = np.full((nF), 0.001)

# desired torso pose trajectory
r_des = np.array([ \
    np.full((2*N+1), 0),
    #0.1* np.sin(2*t),
    #np.full((2*N+1),0.01),
    0.15* np.cos(2*t),
    #np.full((2*N+1),1.4)
    1.2 + 0.29*np.sin(2*t)
    ])

# generate torso pose and velocity trajectory
spline = CubicSpline(t, r_des, axis=1)
xr_des = np.vstack((spline(t), spline(t,1)))

# desired foot location
o_feet_des = np.hstack(( \
    np.array([foot_size[0]/2, -torso_size[1]/2-foot_size[1]/2, 0])[:,None],
    np.array([foot_size[0]/2, -torso_size[1]/2+foot_size[1]/2, 0])[:,None],
    np.array([-foot_size[0]/2, -torso_size[1]/2-foot_size[1]/2, 0])[:,None],
    np.array([-foot_size[0]/2, -torso_size[1]/2+foot_size[1]/2, 0])[:,None],
    np.array([foot_size[0]/2, torso_size[1]/2-foot_size[1]/2, 0])[:,None],
    np.array([foot_size[0]/2, torso_size[1]/2+foot_size[1]/2, 0])[:,None],
    np.array([-foot_size[0]/2, torso_size[1]/2-foot_size[1]/2, 0])[:,None],
    np.array([-foot_size[0]/2, torso_size[1]/2+foot_size[1]/2, 0])[:,None]
    ))

# function derivations

# cartesian basis vectors
i0 = np.array([1.0, 0.0, 0.0])[:,None]
j0 = np.array([0.0, 1.0, 0.0])[:,None]
k0 = np.array([0.0, 0.0, 1.0])[:,None]

# derive functions for rotation matrices and homogeneous transformation matrices
def derive_coord_trans():
    s = ca.SX.sym('s', 3)
    th = ca.SX.sym('th')
    th_offset = ca.SX.sym('th_offset')
    d = ca.SX.sym('d')
    a = ca.SX.sym('a')
    alpha = ca.SX.sym('alpha')

    # cross product matrix
    skew_sym = ca.SX(np.array([ \
            [0, -s[2], s[1]],
            [s[2], 0, -s[0]],
            [-s[1], s[0], 0]
        ]))

    # rotation matrix using Rodrigues' rotation formula
    R_sym = ca.SX.eye(3) + ca.sin(th)*skew_sym + (1-ca.cos(th))*skew_sym@skew_sym
    R = ca.Function('R', [th, s], [R_sym])

    # homogeneous transformation matrix
    T_sym = ca.vertcat( \
        ca.horzcat(R(th+th_offset,k0)@R(alpha,i0), R(th+th_offset,k0)@(a*i0) + d*k0),
        np.array([0.0, 0.0, 0.0, 1.0])[None,:]
        )
    T = ca.Function('T', [th, th_offset, d, a, alpha], [T_sym])

    return R, T

# derive forward kinematics for biped
def derive_forkin():
    q = ca.SX.sym('q', nq)

    T_base_right = T(pi/2, 0, -torso_size[2], -torso_size[1]/2, pi)
    T_base_left = T(pi/2, 0, -torso_size[2], torso_size[1]/2, pi)
    T6_c1 = np.array([ \
        [0, 1, 0, -foot_size[1]/2],
        [1, 0, 0, foot_size[0]/2],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
        ])
    T6_c2 = np.array([ \
        [0, 1, 0, foot_size[1]/2],
        [1, 0, 0, foot_size[0]/2],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
        ])
    T6_c3 = np.array([ \
        [0, 1, 0, -foot_size[1]/2],
        [1, 0, 0, -foot_size[0]/2],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
        ])
    T6_c4 = np.array([ \
        [0, 1, 0, foot_size[1]/2],
        [1, 0, 0, -foot_size[0]/2],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
        ])

    T_r1 = T_base_right @ T(q[0], theta_offset[0], d[0], a[0], alpha[0])
    T_r2 = T_r1 @ T(q[1], theta_offset[1], d[1], a[1], alpha[1])
    T_r3 = T_r2 @ T(q[2], theta_offset[2], d[2], a[2], alpha[2])
    T_r4 = T_r3 @ T(q[3], theta_offset[3], d[3], a[3], alpha[3])
    T_r5 = T_r4 @ T(q[4], theta_offset[4], d[4], a[4], alpha[4])
    T_r6 = T_r5 @ T(q[5], theta_offset[5], d[5], a[5], alpha[5])
    T_rc1 = T_r6 @ T6_c1
    T_rc2 = T_r6 @ T6_c2
    T_rc3 = T_r6 @ T6_c3
    T_rc4 = T_r6 @ T6_c4

    T_l1 = T_base_left @ T(q[6], theta_offset[0], d[0], a[0], alpha[0])
    T_l2 = T_l1 @ T(q[7], theta_offset[1], d[1], a[1], alpha[1])
    T_l3 = T_l2 @ T(q[8], theta_offset[2], d[2], a[2], alpha[2])
    T_l4 = T_l3 @ T(q[9], theta_offset[3], d[3], a[3], alpha[3])
    T_l5 = T_l4 @ T(q[10], theta_offset[4], d[4], a[4], alpha[4])
    T_l6 = T_l5 @ T(q[11], theta_offset[5], d[5], a[5], alpha[5])
    T_lc1 = T_l6 @ T6_c1
    T_lc2 = T_l6 @ T6_c2
    T_lc3 = T_l6 @ T6_c3
    T_lc4 = T_l6 @ T6_c4

    o_feet_sym = ca.horzcat(T_rc1[0:3,3], T_rc2[0:3,3], T_rc3[0:3,3], T_rc4[0:3,3],
        T_lc1[0:3,3], T_lc2[0:3,3], T_lc3[0:3,3], T_lc4[0:3,3])
    forkin_feet = ca.Function('forkin_feet', [q], [o_feet_sym])

    o_leg_sym = ca.horzcat(T_r1[0:3,3], T_r3[0:3,3], T_r6[0:3,3],
        T_l1[0:3,3], T_l3[0:3,3], T_l6[0:3,3])
    forkin_leg = ca.Function('forkin_leg', [q], [o_leg_sym])

    return forkin_feet, forkin_leg

# derive function for returning torso frame
def derive_T_r():
    r = ca.SX.sym('r', nr)
    T_r_sym = np.array([ \
        [1, 0, 0, r[0]],
        [0, 1, 0, r[1]],
        [0, 0, 1, r[2]],
        [0, 0, 0, 1]
        ])

    T_r = ca.Function('T_r', [r], [T_r_sym])
    return T_r

# derive dynamic equation of motion
def derive_dynamics():
    xr = ca.SX.sym('xr', nr*2)
    uF = ca.SX.sym('uF', nF)

    # TODO: multiply F by T of feet to account for slanted terrains
    F_net = ca.sum2(uF.reshape((3,8)))
    rddot = F_net/m + g

    f_sym = ca.vertcat(xr[nr:], rddot)
    f = ca.Function('f', [xr,uF], [f_sym])

    return f

_, T = derive_coord_trans()
forkin_feet, forkin_leg = derive_forkin()
T_r = derive_T_r()
f = derive_dynamics()

# trajectory optimization
opti = ca.Opti()

# decision variables
XR = opti.variable(nr*2, 2*N+1) # configuration + velocity of COM position
XQ = opti.variable(nq*2, 2*N+1) # configuration + velocity of joint angles
UF = opti.variable(nF, 2*N+1) # contact point force

# cost
# simpson quadrature coefficients, to be used to compute integrals
simp = np.empty((1,2*N+1))
simp[0,::2] = 2
simp[0,1::2] = 4
simp[0,0], simp[0,-1]  = 1, 1

J = 0.0
for i in range(2*N+1):
    for r_idx in range(nr):
        J += Qr[r_idx]*simp[0][i]*(XR[r_idx,i]-xr_des[r_idx,i])**2
        J += Qrd[r_idx]*simp[0][i]*(XR[nr+r_idx,i]-xr_des[nr+r_idx,i])**2

    for q_idx in range(nq):
        J += Qqd[q_idx]*simp[0][i]*(XQ[nq+q_idx,i])**2

    for F_idx in range(nF):
        J += RF[F_idx]*simp[0][i]*(UF[F_idx,i])**2

opti.minimize(J)

# initial condition constraint
opti.subject_to(XQ[:,0] ==  np.hstack((q_init, np.zeros(q_init.shape))))

for i in range(2*N+1):
    if i%2 != 0:
        # for each finite element:
        xr_left, xr_mid, xr_right = XR[:,i-1], XR[:,i], XR[:,i+1]
        xq_left, xq_mid, xq_right = XQ[:,i-1], XQ[:,i], XQ[:,i+1]
        uF_left, uF_mid, uF_right = UF[:,i-1], UF[:,i], UF[:,i+1]
        f_left, f_mid, f_right = \
            f(xr_left, uF_left), f(xr_mid, uF_mid), f(xr_right, uF_right)

        # interpolation constraints
        opti.subject_to( \
            # equation (6.11) in Kelly 2017
            xr_mid == (xr_left+xr_right)/2.0 + tf/N*(f_left-f_right)/8.0)

        # collocation constraints
        opti.subject_to( \
            # equation (6.12) in Kelly 2017
            tf/N*(f_left+4*f_mid+f_right)/6.0 == xr_right-xr_left)

    # backwards euler velocity constraint on joint angles
    if i != 0:
        opti.subject_to( \
            # q_i - q_{i-1} = qdot_i * dt
            XQ[:nq,i]-XQ[:nq,i-1] == XQ[nq:,i]*tf/(2*N))

    # forward kinematics constraint
    o_feet_rel_i = forkin_feet(XQ[:nq,i])
    o_feet_i = XR[:nr,i] + o_feet_rel_i
    opti.subject_to(o_feet_i == o_feet_des)

    # zero angular momentum constraint
    torque = ca.MX.zeros(3,8)
    for j in range(8):
        torque[:,j] = ca.cross(o_feet_rel_i[:,j], UF[3*j:3*j+3,i])
    opti.subject_to(ca.sum2(torque) == np.zeros((3,1)))

    #friction cone constraint
    Fz = UF[:,i].reshape((3,8))[-1,:]
    opti.subject_to(Fz >= np.zeros(Fz.shape))
    for j in range(8):
        opti.subject_to(UF[3*j,i] <= mu*Fz[j])
        opti.subject_to(-UF[3*j,i] <= mu*Fz[j])
        opti.subject_to(UF[3*j+1,i] <= mu*Fz[j])
        opti.subject_to(-UF[3*j+1,i] <= mu*Fz[j])

    #joint constraints
    opti.subject_to(opti.bounded(q_bounds[:,0], XQ[:nq,i], q_bounds[:,1]))

# initial guess for decision variables
XR_guess = np.repeat(np.array([0, 0, 1.5, 0, 0, 0])[None,:],2*N+1,axis=0).T
XQ_guess = np.repeat(np.hstack((q_init, np.zeros(q_init.shape)))[None,:],2*N+1,axis=0).T
opti.set_initial(XR, XR_guess)
opti.set_initial(XQ, XQ_guess)

# solve NLP
p_opts = {}
s_opts = {'print_level': 5}
opti.solver('ipopt', p_opts, s_opts)
sol =   opti.solve()

# extract NLP output as numpy array
XR_sol = np.array(sol.value(XR))[:nr,:]
XQ_sol = np.array(sol.value(XQ))[:nq,:]
UF_sol = np.array(sol.value(UF))

# animate
axis_lim = 1.5
F_len = 0.02

for i in range(2*N+1):
    r_i = np.array(XR_sol[:nr,i])
    q_i = np.array(XQ_sol[:nq,i])
    o_feet_i = np.array(r_i + forkin_feet(XQ_sol[:,i]))
    o_leg_i = np.array(r_i + forkin_leg(XQ_sol[:,i]))
    F_i = np.array(UF_sol[:,i])

    o_i = {}
    o_i['r']  = r_i[:,None]
    o_i['r1'] = o_leg_i[:,0][:,None]
    o_i['r3'] = o_leg_i[:,1][:,None]
    o_i['r6'] = o_leg_i[:,2][:,None]
    o_i['l1'] = o_leg_i[:,3][:,None]
    o_i['l3'] = o_leg_i[:,4][:,None]
    o_i['l6'] = o_leg_i[:,5][:,None]
    o_i['rc1'] = o_feet_i[:,0][:,None]
    o_i['rc2'] = o_feet_i[:,1][:,None]
    o_i['rc3'] = o_feet_i[:,2][:,None]
    o_i['rc4'] = o_feet_i[:,3][:,None]
    o_i['lc1'] = o_feet_i[:,4][:,None]
    o_i['lc2'] = o_feet_i[:,5][:,None]
    o_i['lc3'] = o_feet_i[:,6][:,None]
    o_i['lc4'] = o_feet_i[:,7][:,None]

    F_vec_i = {}
    F_vec_i['rc1'] = o_i['rc1'] + F_len * F_i[0:3][:,None]
    F_vec_i['rc2'] = o_i['rc2'] + F_len * F_i[3:6][:,None]
    F_vec_i['rc3'] = o_i['rc3'] + F_len * F_i[6:9][:,None]
    F_vec_i['rc4'] = o_i['rc4'] + F_len * F_i[9:12][:,None]
    F_vec_i['lc1'] = o_i['lc1'] + F_len * F_i[12:15][:,None]
    F_vec_i['lc2'] = o_i['lc2'] + F_len * F_i[15:18][:,None]
    F_vec_i['lc3'] = o_i['lc3'] + F_len * F_i[18:21][:,None]
    F_vec_i['lc4'] = o_i['lc4'] + F_len * F_i[21:24][:,None]

    if i==0:
        o = o_i
        F_vec = F_vec_i
    else:
        for k, v in o_i.items():
            o[k] = np.hstack((o[k], v))
        for k, v in F_vec_i.items():
            F_vec[k] = np.hstack((F_vec[k], v))

leg_r_coord = np.zeros((3, 3, 2*N+1)) #(cartesian space)*(# datapoints)*(# time points)
leg_l_coord = np.zeros((3, 3, 2*N+1))
foot_r_coord = np.zeros((3, 5, 2*N+1))
foot_l_coord = np.zeros((3, 5, 2*N+1))
torso_coord = np.zeros((3, 4, 2*N+1))
F_coord = np.zeros((8, 3, 2, 2*N+1)) #(# forces)*(cartesian space)*(# datapoints)*(# time points) 
for xyz in range(3):
    leg_r_coord[xyz,:,:] = np.array([o['r1'][xyz,:], o['r3'][xyz,:], o['r6'][xyz,:]])
    leg_l_coord[xyz,:,:] = np.array([o['l1'][xyz,:], o['l3'][xyz,:], o['l6'][xyz,:]])
    foot_r_coord[xyz,:,:] = np.array([o['rc1'][xyz,:], o['rc2'][xyz,:], 
        o['rc4'][xyz,:], o['rc3'][xyz,:], o['rc1'][xyz,:]])
    foot_l_coord[xyz,:,:] = np.array([o['lc1'][xyz,:], o['lc2'][xyz,:], 
        o['lc4'][xyz,:], o['lc3'][xyz,:], o['lc1'][xyz,:]])
    torso_coord[xyz,:,:] = np.array([o['r'][xyz,:], o['r1'][xyz,:], 
        o['l1'][xyz,:], o['r'][xyz,:]])
    for j, key in enumerate(['rc1', 'rc2', 'rc3', 'rc4', 'lc1', 'lc2', 'lc3', 'lc4']):
        F_coord[j,xyz,:,:] = np.array([o[key][xyz,:], F_vec[key][xyz,:]])

anim_fig = plt.figure(figsize=(12, 12))
ax = Axes3D(anim_fig)
lines = [plt.plot([], [])[0] for _ in range(6+8)]

def animate(i):

    lines[0].set_data(foot_r_coord[0,:,i], foot_r_coord[1,:,i])
    lines[0].set_3d_properties(foot_r_coord[2,:,i])
    lines[1].set_data(foot_l_coord[0,:,i], foot_l_coord[1,:,i])
    lines[1].set_3d_properties(foot_l_coord[2,:,i])

    lines[2].set_data(leg_r_coord[0,:,i], leg_r_coord[1,:,i])
    lines[2].set_3d_properties(leg_r_coord[2,:,i])
    lines[3].set_data(leg_l_coord[0,:,i], leg_l_coord[1,:,i])
    lines[3].set_3d_properties(leg_l_coord[2,:,i])

    lines[4].set_data(torso_coord[0,:,i], torso_coord[1,:,i])
    lines[4].set_3d_properties(torso_coord[2,:,i])

    lines[5].set_data(r_des[0,i], r_des[1,i])
    lines[5].set_3d_properties(r_des[2,i])

    for j in range(8):
        lines[6+j].set_data(F_coord[j,0,:,i], F_coord[j,1,:,i])
        lines[6+j].set_3d_properties(F_coord[j,2,:,i])

    ax.view_init(azim=i/2)

    return lines

#ax.view_init(azim=45)
ax.set_xlim3d([-1, 1])
ax.set_ylim3d([-1, 1])
ax.set_zlim3d([0, 2])

for line in lines[0:2]:
    line.set_color('g')
    line.set_linewidth(5)
    line.set_marker('o')
    line.set_markeredgewidth(7)

for line in lines[2:5]:
    line.set_color('b')
    line.set_linewidth(5)
    line.set_marker('o')
    line.set_markeredgewidth(7)

lines[5].set_color('r')
lines[5].set_marker('o')
lines[5].set_markeredgewidth(7)

for line in lines[6:]:
    line.set_color('r')
    line.set_linewidth(3)

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=2*N+1, 
    interval=tf*1000/(2*N+1), repeat=True, blit=False)

'''
# uncomment to write to file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=int((2*N+1)/tf), metadata=dict(artist='Me'), bitrate=1000)
anim.save('point_mass_yz_circular2' + '.mp4', writer=writer)
'''

plt.show()