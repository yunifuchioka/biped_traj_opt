import numpy as np
from numpy import pi
import casadi as ca

from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
plt.style.use('seaborn')

TOL = 1E-12 # value under which a number is considered to be zero

ndim = 2 # dimension of cartesian space
nr = 2 # dimension of centroid configuration vector
nth = 1 # dimension of rotation configuration vector
nj = 4 # number of contact points

foot_size = np.array([0.3, 0])
torso_size = np.array([0.3, 0.5])

m = 50 # mass of torso
I = 5 # moment of inertia of torso
g = np.array([0, -9.81]) # gravitational acceleration
mu = 0.8 #friciton coefficient

tf = 10 # final time of interval
N = int(tf*4) # number of hermite-simpson finite elements. Total # of points = 2*N+1
t = np.linspace(0,tf, 2*N+1) # discretized time

Qr = np.array([1, 1000])
Qrd = np.array([1, 1])
Qth = np.array([1])
Qthd = np.array([1])
Qc = np.full((nj*ndim), 1000)
Qcd = np.full((nj*ndim), 10)
RF = np.full((nj*ndim), 0.0001)

from scipy.signal import square
r_traj = np.array([ \
    #np.repeat(0, 2*N+1),
    #0.5*-np.sin(2*t),
    0.5*-square(2*t),
    np.repeat(1, 2*N+1),
    #1.2 + 0.2*np.sin(2*t)
    ])
r_spline = CubicSpline(t, r_traj, axis=1)
xr_des = np.vstack((r_spline(t), r_spline(t,1)))

th_traj = np.repeat(0, 2*N+1)
#th_traj = pi/2 * np.sin(t)
th_spline = CubicSpline(t, th_traj, axis=1)
xth_des = np.vstack((th_spline(t), th_spline(t,1)))

#c_des = np.tile([0.5, 0, 0.2, 0, -0.2, 0, -0.5, 0], (2*N+1,1)).T
#'''
c_des = np.array([ \
    np.repeat(0.5, 2*N+1),
    #np.repeat(0, 2*N+1),
    np.maximum(0, 0.2*np.sin(2*t)),
    np.repeat(0.2, 2*N+1),
    #np.repeat(0, 2*N+1),
    np.maximum(0, 0.2*np.sin(2*t)),
    np.repeat(-0.2, 2*N+1),
    #np.repeat(0, 2*N+1),
    np.maximum(0, -0.2*np.sin(2*t)),
    np.repeat(-0.5, 2*N+1),
    #np.repeat(0, 2*N+1)
    np.maximum(0, -0.2*np.sin(2*t)),
    ])
#'''

F_des = np.tile(-m*g/nj, (2*N+1, nj)).T
# normal distance from ground surface for each time
c_normal_dist = np.vstack((c_des[1,:], c_des[3,:],
    c_des[5,:], c_des[7,:]))
# set Fz=0 when c_des not in contact with ground
F_des[1::2,:][np.abs(c_normal_dist)>=TOL] = 0

# derive dynamic equation of motion
def derive_dynamics():
    xr = ca.SX.sym('xr', nr*2)
    xth = ca.SX.sym('xth', nth*2)
    uF = ca.SX.sym('uF', nj*ndim)
    xc = ca.SX.sym('xc', nj*ndim)

    F_net = ca.sum2(uF.reshape((ndim,nj)))
    rddot = F_net/m + g

    #torque_net = ca.SX.zeros(ndim)
    torque_net = ca.SX.zeros(1)
    for j in range(nj):
        #torque_net += ca.cross(xc[j*ndim:j*ndim+ndim], uF[j*ndim:j*ndim+ndim])
        torque_net += xc[j*ndim]*uF[j*ndim+1] - xc[j*ndim+1]*uF[j*ndim]

    fr_sym = ca.vertcat(xr[nr:], rddot)
    fr = ca.Function('fr', [xr,uF], [fr_sym])

    fth_sym = ca.vertcat(xth[nth:]/I, torque_net)
    fth = ca.Function('fth', [xth,xc,uF], [fth_sym])

    return fr, fth
 
fr, fth = derive_dynamics()

opti = ca.Opti()

XR = opti.variable(nr*2, 2*N+1) # configuration + velocity of COM position
XTH = opti.variable(nth*2, 2*N+1)
XC = opti.variable(nj*ndim, 2*N+1) # configuration of contact points
UF = opti.variable(nj*ndim, 2*N+1) # contact point force

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

    for th_idx in range(nth):
        J += Qth[th_idx]*simp[0][i]*(XTH[th_idx,i]-xth_des[th_idx,i])**2
        J += Qthd[th_idx]*simp[0][i]*(XTH[nth+th_idx,i]-xth_des[nth+th_idx,i])**2

    for c_idx in range(nj*ndim):
        J += Qc[c_idx]*simp[0][i]*(XC[c_idx,i]-c_des[c_idx,i])**2
        if i != 0:
            J += Qcd[c_idx]*simp[0][i]*(c_des[c_idx,i]-c_des[c_idx,i-1])**2

    for F_idx in range(nj*ndim):
        J += RF[F_idx]*simp[0][i]*(UF[F_idx,i])**2

opti.minimize(J)

# initial condition constraint
opti.subject_to(XR[:,0] ==  xr_des[:,0])
opti.subject_to(XTH[:,0] ==  xth_des[:,0])
opti.subject_to(XC[:,0] ==  c_des[:,0])

for i in range(2*N+1):
    # point mass dynamics constraints imposed through Hermite-Simpson
    if i%2 != 0:
        # for each finite element:
        xr_left, xr_mid, xr_right = XR[:,i-1], XR[:,i], XR[:,i+1]
        xth_left, xth_mid, xth_right = XTH[:,i-1], XTH[:,i], XTH[:,i+1]
        uF_left, uF_mid, uF_right = UF[:,i-1], UF[:,i], UF[:,i+1]
        xc_left, xc_mid, xc_right = XC[:,i-1], XC[:,i], XC[:,i+1]
        fr_left, fr_mid, fr_right = \
            fr(xr_left, uF_left), fr(xr_mid, uF_mid), fr(xr_right, uF_right)
        fth_left, fth_mid, fth_right = \
            fth(xth_left, xc_left, uF_left), fth(xth_mid, xc_mid, uF_mid), \
            fth(xth_right, xc_mid, uF_right)

        # interpolation constraints
        opti.subject_to( \
            # equation (6.11) in Kelly 2017
            xr_mid == (xr_left+xr_right)/2.0 + tf/N*(fr_left-fr_right)/8.0)
        opti.subject_to( \
            # equation (6.11) in Kelly 2017
            xth_mid == (xth_left+xth_right)/2.0 + tf/N*(fth_left-fth_right)/8.0)

        # collocation constraints
        opti.subject_to( \
            # equation (6.12) in Kelly 2017
            tf/N*(fr_left+4*fr_mid+fr_right)/6.0 == xr_right-xr_left)
        opti.subject_to( \
            # equation (6.12) in Kelly 2017
            tf/N*(fth_left+4*fth_mid+fth_right)/6.0 == xth_right-xth_left)

    # all other constraints imposed at every timestep

    r_i = XR[:nr,i] # COM position at current timestep
    F_i = UF[:,i].reshape((ndim,nj)) # contact forces at current timestep, 3xnj matrix
    c_i = XC[:nj*ndim,i].reshape((ndim,nj)) # global contact pos at current timestep, 3xnj matrix
    c_i_prev = XC[:nj*ndim,i-1].reshape((ndim,nj)) # global contact pos at prev timestep, 3xnj matrix
    c_rel_i = c_i - r_i # contact pos rel to COM at current timestep, 3xnj matrix

    # foot size and orientation constraint
    # TODO: ignore terrain rotations in z axis
    opti.subject_to(c_i[:,0] - c_i[:,1] == foot_size)
    opti.subject_to(c_i[:,2] - c_i[:,3] == foot_size)

    Fs_i = F_i
    cs_i = c_i
    cds_i = c_i - c_i_prev

    # friciton cone constraints in surface coordinates
    opti.subject_to(Fs_i[-1,:] >= np.zeros(Fs_i[-1,:].shape))
    opti.subject_to(opti.bounded(-mu*Fs_i[-1,:], Fs_i[0,:], mu*Fs_i[-1,:]))

    # contact constraints in surface coordinates
    opti.subject_to(cs_i[-1,:] >= np.zeros(cs_i[-1,:].shape))
    opti.subject_to(opti.bounded(-TOL, cs_i[-1,:] * Fs_i[-1,:], TOL))
    if i != 0:
        opti.subject_to(opti.bounded(-TOL, cds_i[0,:] * Fs_i[-1, :], TOL))

opti.set_initial(XR, xr_des)
opti.set_initial(XC, c_des)
opti.set_initial(UF, F_des)

# solve NLP
p_opts = {}
s_opts = {'print_level': 5}
opti.solver('ipopt', p_opts, s_opts)
sol =   opti.solve()

XR_sol = np.array(sol.value(XR))[:nr,:]
XTH_sol = np.array(sol.value(XTH))[:nth,:]
XC_sol = np.array(sol.value(XC))
UF_sol = np.array(sol.value(UF))

F_len = 1/np.max(UF_sol)

for i in range(2*N+1):
    r_i = np.array(XR_sol[:nr,i])
    th_i = np.array(XTH_sol[:nth,i])
    p_feet_i = XC_sol[:,i].reshape((ndim,nj), order='F')
    F_i = np.array(UF_sol[:,i])

    p_i = {}
    p_i['r']  = r_i[:,None]
    p_i['rc1'] = p_feet_i[:,0][:,None]
    p_i['rc2'] = p_feet_i[:,1][:,None]
    p_i['lc1'] = p_feet_i[:,2][:,None]
    p_i['lc2'] = p_feet_i[:,3][:,None]

    rot_i = np.array([[np.cos(th_i[0]), -np.sin(th_i[0])],
        [np.sin(th_i[0]), np.cos(th_i[0])]])

    # torso points
    p_i['rt1'] = p_i['r'] + rot_i @ np.array([torso_size[0]/2, torso_size[1]/2])[:,None]
    p_i['rt2'] = p_i['r'] + rot_i @ np.array([-torso_size[0]/2, torso_size[1]/2])[:,None]
    p_i['rt3'] = p_i['r'] + rot_i @ np.array([-torso_size[0]/2, -torso_size[1]/2])[:,None]
    p_i['rt4'] = p_i['r'] + rot_i @ np.array([torso_size[0]/2, -torso_size[1]/2])[:,None]

    F_vec_i = {}
    F_vec_i['rc1'] = p_feet_i[:,0][:,None] + F_len * F_i[0:2][:,None]
    F_vec_i['rc2'] = p_feet_i[:,1][:,None] + F_len * F_i[2:4][:,None]
    F_vec_i['lc1'] = p_feet_i[:,2][:,None] + F_len * F_i[4:6][:,None]
    F_vec_i['lc2'] = p_feet_i[:,3][:,None] + F_len * F_i[6:8][:,None]

    if i==0:
        p = p_i
        F_vec = F_vec_i
    else:
        for k, v in p_i.items():
            p[k] = np.hstack((p[k], v))
        for k, v in F_vec_i.items():
            F_vec[k] = np.hstack((F_vec[k], v))

torso_coord = np.zeros((ndim, 5, 2*N+1))  #(cartesian space)*(# datapoints)*(# time points)
F_coord = np.zeros((nj, ndim, 2, 2*N+1)) #(# forces)*(cartesian space)*(# datapoints)*(# time points) 
for xyz in range(ndim):
    torso_coord[xyz,:,:] = np.array([p['rt1'][xyz,:], p['rt2'][xyz,:], 
        p['rt3'][xyz,:], p['rt4'][xyz,:],p['rt1'][xyz,:]])
    for j, key in enumerate(['rc1', 'rc2', 'lc1', 'lc2']):
        F_coord[j,xyz,:,:] = np.array([p[key][xyz,:], F_vec[key][xyz,:]])

anim_fig = plt.figure(figsize=(12, 12))
plt.xlim(-1.5, 1.5)
plt.ylim(-1, 2)
lines = [plt.plot([], [])[0] for _ in range(2+2*nj)]

def animate(i):
    lines[0].set_data(XR_sol[0,i], XR_sol[1,i])
    for j in range(nj):
        lines[1+j].set_data(XC_sol[ndim*j,i], XC_sol[ndim*j+1,i])
        lines[1+nj+j].set_data(F_coord[j,0,:,i], F_coord[j,1,:,i])
    lines[1+2*nj].set_data(torso_coord[0,:,i], torso_coord[1,:,i])

lines[0].set_color('b')
lines[0].set_marker('o')
lines[0].set_markeredgewidth(7)

for line in lines[1:1+nj]:
    line.set_color('g')
    line.set_marker('o')
    line.set_markeredgewidth(7)

for line in lines[1+nj:1+2*nj]:
    line.set_color('r')
    line.set_linewidth(3)

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=2*N+1, 
    interval=tf*1000/(2*N+1), repeat=True, blit=False)

plt.show()