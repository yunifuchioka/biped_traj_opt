import numpy as np
from numpy import pi
import casadi as ca

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
plt.style.use('seaborn')

TOL = 1E-12 # value under which a number is considered to be zero

nr = 3 # dimension of centroid configuration vector
nj = 4 # number of contact points
nj3 = 3*nj # dimension of vectors associated with contact points

leg_len = np.array([0.5, 0.5]) # thigh and calf lengh respectively
torso_size = np.array([0.2, 0.2, 0.5]) # x,y,z length of torso
foot_size = np.array([0.2, 0, 0]) # x,y,z length of feet

m = 50 # mass of torso
I_vals = np.array([5, 5, 5])
I = np.diag(I_vals)
I_inv = np.diag(1/I_vals)
g = np.array([0, 0, -9.81]) # gravitational acceleration
mu = 0.8 #friciton coefficient

tf = 7 # final time of interval
N = int(tf*4) # number of hermite-simpson finite elements. Total # of points = 2*N+1
t = np.linspace(0,tf, 2*N+1) # discretized time
dt = tf/(2*N)

# objective cost weights
Qr = np.array([1000, 1000, 1000])
Qrd = np.array([1, 1, 1])
Qth = np.array([50, 50, 50])
Qw = np.full(3, 10)
Qc = np.full((nj3), 1000)
Qcd = np.full((nj3), 10)
RF = np.full((nj3), 0.0001)

Kp = np.linalg.solve(np.array([[2,1,1],[1,2,1],[1,1,2]]), 2*Qth) #3 element vector
Gp = sum(Kp)*np.eye(3)-np.diag(Kp) #3x3 matrix

def derive_rotMat():
    s = ca.SX.sym('s', 3)
    th = ca.SX.sym('th')

    # cross product matrix
    skew_sym = ca.SX(np.array([ \
            [0, -s[2], s[1]],
            [s[2], 0, -s[0]],
            [-s[1], s[0], 0]
        ]))
    skew = ca.Function('skew', [s], [skew_sym])

    # rotation matrix using Rodrigues' rotation formula
    rotMat_sym = ca.SX.eye(3) + ca.sin(th)*skew_sym + (1-ca.cos(th))*skew_sym@skew_sym
    rotMat = ca.Function('rotMat', [th, s], [rotMat_sym])

    return skew, rotMat

skew, rotMat = derive_rotMat()

Rs_r = np.eye(3)
#Rs_r = Rotation.from_rotvec(37*pi/180 * np.array([0, 1, 0])).as_matrix()
ps_r = np.zeros((3,1))

Rs_l = Rs_r
#Rs_l = Rotation.from_rotvec(10*pi/180 * np.array([0, 1, 0])).as_matrix()
ps_l = ps_r
#ps_l = np.array([0, 0, 0.3])[:,None]

r_traj = np.array([ \
    #np.repeat(0, 2*N+1),
    0.2*np.sin(2*t),
    np.repeat(0, 2*N+1),
    #0.15*np.cos(2*t),
    np.repeat(1.45, 2*N+1)
    #1.2 + 0.2*np.cos(2*t)
    ])

TH_des = np.eye(3)

c_des = np.array([ \
    np.repeat(foot_size[0]/2, 2*N+1),
    np.repeat(-torso_size[1]/2, 2*N+1),
    np.repeat(0, 2*N+1),
    np.repeat(-foot_size[0]/2, 2*N+1),
    np.repeat(-torso_size[1]/2, 2*N+1),
    np.repeat(0, 2*N+1),
    np.repeat(foot_size[0]/2, 2*N+1),
    np.repeat(torso_size[1]/2, 2*N+1),
    np.repeat(0, 2*N+1),
    np.repeat(-foot_size[0]/2, 2*N+1),
    np.repeat(torso_size[1]/2, 2*N+1),
    np.repeat(0, 2*N+1),
    ])

# torso pose and velocity trajectory
spline = CubicSpline(t, r_traj, axis=1)
xr_des = np.vstack((spline(t), spline(t,1)))

# guess contact force trajectory
# initialize as Fx=Fy=0, Fz=(torso mass)/(# contact points) 
F_des = np.tile(-m*g/4, (2*N+1, nj)).T
# desired contact point distances from ground surface origin (in global frame)
c_des_rel = c_des - np.vstack((ps_r, ps_r, ps_l, ps_l))
# normal distance from ground surface for each time
c_normal_dist = np.vstack((Rs_r.T[2,:] @ c_des_rel[0:3,:], Rs_r.T[2,:] @ c_des_rel[3:6,:],
    Rs_l.T[2,:] @ c_des_rel[6:9,:], Rs_l.T[2,:] @ c_des_rel[9:12,:]))
# set Fz=0 when c_des not in contact with ground
F_des[2::3,:][np.abs(c_normal_dist)>=TOL] = 0

# trajectory optimization #############################################################
opti = ca.Opti()

# decision variables
XR = opti.variable(nr*2, 2*N+1) # configuration + velocity of COM position
XTH = opti.variable(3, 3*(2*N+1)) # torso angle matrix
XW = opti.variable(3, 2*N+1) # torso angle matrix
XC = opti.variable(nj3, 2*N+1) # configuration of contact points
UF = opti.variable(nj3, 2*N+1) # contact point force

J = 0.0
for i in range(2*N+1):
    for r_idx in range(nr):
        J += Qr[r_idx]*(XR[r_idx,i]-xr_des[r_idx,i])**2
        J += Qrd[r_idx]*(XR[nr+r_idx,i]-xr_des[nr+r_idx,i])**2

    J += 0.5*ca.trace(Gp - Gp @ TH_des.T @ XTH[:,3*i:3*i+3])
    for th_idx in range(3):
        J += Qw[th_idx]*(XW[th_idx,i])**2

    for c_idx in range(nj3):
        J += Qc[c_idx]*(XC[c_idx,i]-c_des[c_idx,i])**2
        if i < 2*N:
            J += Qcd[c_idx]*(c_des[c_idx,i+1]-c_des[c_idx,i])**2

    for F_idx in range(nj3):
        J += RF[F_idx]*(UF[F_idx,i])**2

opti.minimize(J)

# initial condition constraint
opti.subject_to(XR[:,0] ==  xr_des[:,0])
#opti.subject_to(XTH[:,:3] ==  np.eye(3))
opti.subject_to(XTH[:,:3] ==  rotMat(0.4, [1,0,0]))
#opti.subject_to(XW[:,0] ==  np.zeros((3,1)))
opti.subject_to(XC[:,0] ==  c_des[:,0])

for i in range(2*N+1):
    r_i = XR[:nr,i] # COM position at current timestep
    rd_i = XR[nr:,i] # COM velocity at current timestep
    TH_i = XTH[:,3*i:3*i+3]
    w_i = XW[:,i]
    F_i = UF[:,i].reshape((3,nj)) # contact forces at current timestep, 3xnj matrix
    c_i = XC[:,i].reshape((3,nj)) # global contact pos at current timestep, 3xnj matrix
    c_rel_i = c_i - r_i # contact pos rel to COM at current timestep, 3xnj matrix

    F_net_i = ca.sum2(F_i)
    tau_net_i = ca.MX.zeros(3)
    for j in range(nj):
        tau_net_i += ca.cross(c_rel_i[:,j], F_i[:,j])

    if i < 2*N:
        r_i_next = XR[:nr,i+1]
        rd_i_next = XR[nr:,i+1]
        TH_i_next = XTH[:,3*(i+1):3*(i+1)+3]
        w_i_next = XW[:,i+1]
        c_i_next = XC[:,i+1].reshape((3,nj))

        # COM position dynamics
        opti.subject_to(r_i_next - r_i == (rd_i_next + rd_i)*dt/2)
        opti.subject_to(rd_i_next - rd_i == (F_net_i/m + g)*dt)

        # rotation dynamics
        opti.subject_to(TH_i_next == TH_i @ rotMat(dt, w_i))
        opti.subject_to( \
            w_i_next -  w_i == I_inv@( TH_i.T@tau_net_i - skew(w_i)@I@w_i )*dt)

    # foot size and orientation constraint
    opti.subject_to(c_i == c_des[:,i].reshape((3,nj), order='F'))
    '''
    # TODO: ignore terrain rotations in z axis
    opti.subject_to(c_i[:,0] - c_i[:,1] == Rs_r @ foot_size)
    opti.subject_to(c_i[:,2] - c_i[:,3] == Rs_l @ foot_size)
    '''

    # foot positions and reaction forces and surface coordinates
    idx_rl = int(nj/2) # index representing switch between right and left feet
    # ground reaction forces in ground surface frame
    Fs_i = ca.horzcat(Rs_r.T @ F_i[:,:idx_rl], Rs_l.T @ F_i[:,idx_rl:])
    # foot locations in ground surface frame
    cs_i = ca.horzcat(Rs_r.T @ (c_i[:,:idx_rl] - ps_r), Rs_l.T @ (c_i[:,idx_rl:] - ps_l))
    if i < 2*N:
        # foot velocities in ground surface frame
        cds_i = ca.horzcat(Rs_r.T @ (c_i_next[:,:idx_rl] - c_i[:,:idx_rl]),
            Rs_l.T @ (c_i_next[:,idx_rl:] - c_i[:,idx_rl:]))

    # friciton cone constraints in surface coordinates
    opti.subject_to(Fs_i[2,:] >= np.zeros(Fs_i[2,:].shape))
    opti.subject_to(opti.bounded(-mu*Fs_i[2,:], Fs_i[0,:], mu*Fs_i[2,:]))
    opti.subject_to(opti.bounded(-mu*Fs_i[2,:], Fs_i[1,:], mu*Fs_i[2,:]))

    # contact constraints in surface coordinates
    opti.subject_to(cs_i[2,:] >= np.zeros(cs_i[2,:].shape))
    opti.subject_to(opti.bounded(-TOL, cs_i[2,:] * Fs_i[2,:], TOL))
    if i < 2*N:
        opti.subject_to(opti.bounded(-TOL, cds_i[0,:] * Fs_i[2, :], TOL))
        opti.subject_to(opti.bounded(-TOL, cds_i[1,:] * Fs_i[2, :], TOL))

    # joint constraint planar approximation
    '''
    opti.subject_to(c_rel_i[0,0]-foot_size[0]/2-c_rel_i[2,0] <= 1.5)
    opti.subject_to(c_rel_i[0,2]-foot_size[0]/2-c_rel_i[2,2] <= 1.5)
    opti.subject_to(-c_rel_i[0,0]+foot_size[0]/2-c_rel_i[2,0] <= 1.5)
    opti.subject_to(-c_rel_i[0,2]+foot_size[0]/2-c_rel_i[2,2] <= 1.5)
    '''

# initial guess for decision variables
XR_guess = xr_des
XC_guess = c_des
UF_guess = F_des
#XC_guess = np.repeat(c_des.reshape((1,nj3),order='F'),2*N+1,axis=0).T
opti.set_initial(XR, XR_guess)
opti.set_initial(XC, XC_guess)
opti.set_initial(UF, UF_guess)

# solve NLP
p_opts = {}
s_opts = {'print_level': 5}
opti.solver('ipopt', p_opts, s_opts)
sol =   opti.solve()

# extract NLP output as numpy array
XR_sol = np.array(sol.value(XR))[:nr,:]
XTH_sol = np.array(sol.value(XTH))
XC_sol = np.array(sol.value(XC))
UF_sol = np.array(sol.value(UF))

# animate ##############################################################################
axis_lim = 1.5
F_len = 1/np.max(UF_sol)

for i in range(2*N+1):
    r_i = np.array(XR_sol[:nr,i])
    p_feet_i = XC_sol[:,i].reshape((3,nj), order='F')
    #p_leg_i = np.array(r_i + forkin_leg(XQ_sol[:,i]))
    F_i = np.array(UF_sol[:,i])
    rot_i = XTH_sol[:,3*i:3*i+3]

    p_i = {}
    p_i['r']  = r_i[:,None]

    p_i['rt1'] = p_i['r'] + rot_i @ np.array([torso_size[0]/2, torso_size[1]/2, torso_size[2]/2])[:,None]
    p_i['rt2'] = p_i['r'] + rot_i @ np.array([torso_size[0]/2, -torso_size[1]/2, torso_size[2]/2])[:,None]
    p_i['rt3'] = p_i['r'] + rot_i @ np.array([torso_size[0]/2, -torso_size[1]/2, -torso_size[2]/2])[:,None]
    p_i['rt4'] = p_i['r'] + rot_i @ np.array([torso_size[0]/2, torso_size[1]/2, -torso_size[2]/2])[:,None]
    p_i['rt5'] = p_i['r'] + rot_i @ np.array([-torso_size[0]/2, torso_size[1]/2, torso_size[2]/2])[:,None]
    p_i['rt6'] = p_i['r'] + rot_i @ np.array([-torso_size[0]/2, -torso_size[1]/2, torso_size[2]/2])[:,None]
    p_i['rt7'] = p_i['r'] + rot_i @ np.array([-torso_size[0]/2, -torso_size[1]/2, -torso_size[2]/2])[:,None]
    p_i['rt8'] = p_i['r'] + rot_i @ np.array([-torso_size[0]/2, torso_size[1]/2, -torso_size[2]/2])[:,None]

    p_i['rc1'] = p_feet_i[:,0][:,None]
    p_i['rc2'] = p_feet_i[:,1][:,None]
    p_i['lc1'] = p_feet_i[:,2][:,None]
    p_i['lc2'] = p_feet_i[:,3][:,None]

    F_vec_i = {}
    F_vec_i['rc1'] = p_i['rc1'] + F_len * F_i[0:3][:,None]
    F_vec_i['rc2'] = p_i['rc2'] + F_len * F_i[3:6][:,None]
    F_vec_i['lc1'] = p_i['lc1'] + F_len * F_i[6:9][:,None]
    F_vec_i['lc2'] = p_i['lc2'] + F_len * F_i[9:12][:,None]

    if i==0:
        p = p_i
        F_vec = F_vec_i
    else:
        for k, v in p_i.items():
            p[k] = np.hstack((p[k], v))
        for k, v in F_vec_i.items():
            F_vec[k] = np.hstack((F_vec[k], v))

foot_r_coord = np.zeros((3, 2, 2*N+1))
foot_l_coord = np.zeros((3, 2, 2*N+1))
#torso_coord = np.zeros((3, 4, 2*N+1))
#torso_coord = np.zeros((3, 8, 2*N+1))
F_coord = np.zeros((nj, 3, 2, 2*N+1)) #(# forces)*(cartesian space)*(# datapoints)*(# time points) 
for xyz in range(3):
    foot_r_coord[xyz,:,:] = np.array([p['rc1'][xyz,:], p['rc2'][xyz,:]])
    foot_l_coord[xyz,:,:] = np.array([p['lc1'][xyz,:], p['lc2'][xyz,:]])
    #torso_coord[xyz,:,:] = np.array([p['r'][xyz,:], p['r1'][xyz,:], 
    #    p['l1'][xyz,:], p['r'][xyz,:]])
    '''
    torso_coord[xyz,:,:] = np.array([p['rt1'][xyz,:], p['rt2'][xyz,:], 
        p['rt3'][xyz,:], p['rt4'][xyz,:],p['rt5'][xyz,:], p['rt6'][xyz,:],
        p['rt7'][xyz,:], p['rt8'][xyz,:]])
    '''
    for j, key in enumerate(['rc1', 'rc2', 'lc1', 'lc2']):
        F_coord[j,xyz,:,:] = np.array([p[key][xyz,:], F_vec[key][xyz,:]])

anim_fig = plt.figure(figsize=(12, 12))
#anim_fig = plt.figure(figsize=plt.figaspect(0.5)*2)
ax = Axes3D(anim_fig)
lines = [plt.plot([], [])[0] for _ in range(3+nj)]

def animate(i):
    if ax.collections:
        ax.collections.pop()

    lines[0].set_data(xr_des[0,i], xr_des[1,i])
    lines[0].set_3d_properties(xr_des[2,i])

    lines[1].set_data(foot_r_coord[0,:,i], foot_r_coord[1,:,i])
    lines[1].set_3d_properties(foot_r_coord[2,:,i])
    lines[2].set_data(foot_l_coord[0,:,i], foot_l_coord[1,:,i])
    lines[2].set_3d_properties(foot_l_coord[2,:,i])
    
    '''
    lines[nj+3].set_data(torso_coord[0,:,i], torso_coord[1,:,i])
    lines[nj+3].set_3d_properties(torso_coord[2,:,i])
    '''
    
    for j in range(nj):
        lines[3+j].set_data(F_coord[j,0,:,i], F_coord[j,1,:,i])
        lines[3+j].set_3d_properties(F_coord[j,2,:,i])

    vert = np.array([p['rt1'][:,i], p['rt2'][:,i], p['rt3'][:,i], p['rt4'][:,i],
        p['rt5'][:,i], p['rt6'][:,i],p['rt7'][:,i], p['rt8'][:,i]])
    verts = [ [vert[0],vert[1],vert[2],vert[3]], [vert[4],vert[5],vert[6],vert[7]],
        [vert[1],vert[2],vert[6],vert[5]],[vert[0],vert[3],vert[7],vert[4]],
        [vert[0],vert[1],vert[5],vert[4]], [vert[3],vert[2],vert[6],vert[7]]]

    ax.add_collection3d(Poly3DCollection(verts, 
        facecolors='cyan', linewidths=1, edgecolors='k', alpha=.1))

    #ax.view_init(azim=i)
    #ax.set_xlim3d([torso_coord[0,1,i]-1, torso_coord[0,1,i]+1])
    #ax.set_ylim3d([torso_coord[1,1,i]-1, torso_coord[1,1,i]+1])

    return lines

ax.view_init(azim=45)
ax.set_xlim3d([-1, 1])
ax.set_ylim3d([-1, 1])
ax.set_zlim3d([0, 2])
#ax.view_init(elev=0, azim=90)
#fig_scale = 1
#fig_shift = np.array([0, 0, 0])
#ax.set_xlim3d([fig_shift[0]-fig_scale, fig_shift[0]+fig_scale])
#ax.set_ylim3d([fig_shift[1]-fig_scale, fig_shift[1]+fig_scale])
#ax.set_zlim3d([fig_shift[2]-fig_scale, fig_shift[2]+fig_scale])

lines[0].set_color('c')
lines[0].set_marker('o')
lines[0].set_markeredgewidth(7)

for line in lines[1:3]:
    line.set_color('g')
    line.set_linewidth(5)
    line.set_marker('o')
    line.set_markeredgewidth(7)

for line in lines[3:]:
    line.set_color('r')
    line.set_linewidth(3)

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=2*N+1, 
    interval=tf*1000/(2*N+1), repeat=True, blit=False)

'''
# uncomment to write to file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=int((2*N+1)/tf), metadata=dict(artist='Me'), bitrate=1000)
anim.save('potato_test_3D_angleControlOff' + '.mp4', writer=writer)
'''

plt.show()