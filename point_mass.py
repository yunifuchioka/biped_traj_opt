import numpy as np
from numpy import pi
import casadi as ca

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
plt.style.use('seaborn')

nr = 3 # dimension of centroid configuration vector
nj = 4 # number of contact points
nj3 = 3*nj # dimension of vectors associated with contact points

leg_len = np.array([0.5, 0.5]) # thigh and calf lengh respectively
torso_size = np.array([0, 0.3, 0.5]) # x,y,z length of torso
foot_size = np.array([0.2, 0, 0]) # x,y,z length of feet

m = 50 # mass of torso
g = np.array([0, 0, -9.81]) # gravitational acceleration
mu = 0.8 #friciton coefficient

Rs = np.eye(3) # surface frame
ps = np.zeros((3,1)) # surface origin

tf = 20.0 # final time of interval
N = int(tf*4) # number of hermite-simpson finite elements. Total # of points = 2*N+1
t = np.linspace(0,tf, 2*N+1) # discretized time

# objective cost weights
Qr = np.array([1, 1, 1000])
Qrd = np.array([1, 1, 1])
Qc = np.full((nj3), 1000)
Qcd = np.full((nj3), 10)
RF = np.full((nj3), 0.0001)

# footstep planner parameters
stride = np.array([0.3, torso_size[1], 0.1]) # forward step length, leg width, max foot height
stride_std = np.array([0.0, 0.0, 0.0]) # random std dev of step x and y locations
T_step = 0.6 # step period
duty = 0.5 # duty cycle - proportion of gait where foot is on the ground

# desired trajectory planning #############################################################
num_steps = int(tf/T_step) # number of footsteps to take
# matrix that specifies footsteps timing and locations
footstep_plan = np.array([ \
    np.linspace(0, tf, num_steps), # time
    np.hstack((np.zeros(2), # x positions
        np.linspace(0.0, (num_steps-4)*stride[0], num_steps-4),
        np.full(2, (num_steps-4)*stride[0]),
        )),
    np.array([-stride[1]/2 + stride[1]*(i%2) for i in range(num_steps)]) # y positions
    ])

# add random noise
footstep_plan[1,2:-1] += np.random.normal(loc=0, scale=stride_std[0],size=max(footstep_plan[1].shape)-3)
footstep_plan[2,2:-1] += np.random.normal(loc=0, scale=stride_std[1],size=max(footstep_plan[2].shape)-3)

# interpolations points for feet: rows are time, x, y, z
right_traj_points = footstep_plan[:,footstep_plan[2,:]<=0]
left_traj_points = footstep_plan[:,footstep_plan[2,:]>=0]
right_traj_points = np.concatenate(( \
    right_traj_points,
    np.zeros(right_traj_points.shape[1])[None,:]
    ),axis=0) 
left_traj_points = np.concatenate(( \
    left_traj_points,
    np.zeros(left_traj_points.shape[1])[None,:]
    ),axis=0)

# add endpoints if they are missing
if right_traj_points[0,0] != 0:
    right_traj_points = np.concatenate(( \
        np.insert(right_traj_points[1:,0], 0, 0)[:,None],
        right_traj_points
        ),axis=1)
if left_traj_points[0,0] != 0:
    left_traj_points = np.concatenate(( \
        np.insert(left_traj_points[1:,0], 0, 0)[:,None],
        left_traj_points
        ),axis=1)
if right_traj_points[0,-1] != tf:
    right_traj_points = np.concatenate(( \
        right_traj_points,
        np.insert(right_traj_points[1:,-1], 0, tf)[:,None],
        ),axis=1)
if left_traj_points[0,-1] != tf:
    left_traj_points = np.concatenate(( \
        left_traj_points,
        np.insert(left_traj_points[1:,-1], 0, tf)[:,None],
        ),axis=1)

# specify when foot liftoff should occur
right_liftoff_time = right_traj_points[0,:-1]*(1-duty)+right_traj_points[0,1:]*duty
left_liftoff_time = left_traj_points[0,:-1]*(1-duty)+left_traj_points[0,1:]*duty

# specify interpolation points corresponding to foot liftoff
right_liftoff_points = np.concatenate(( \
        right_liftoff_time[None,:],
        right_traj_points[1:,:-1]
        ),axis=0)
left_liftoff_points = np.concatenate(( \
        left_liftoff_time[None,:],
        left_traj_points[1:,:-1]
        ),axis=0)

# specify interpolation points corresponding to swing foot midpoint to specify foot height
right_swing_points = np.concatenate(( \
        (right_liftoff_time[1:-1] + right_traj_points[0,2:-1])[None,:]/2,
        (right_traj_points[1:3,2:-1] + right_traj_points[1:3,1:-2])/2,
        np.full(max(right_liftoff_time.shape)-2, stride[2])[None,:]
        ),axis=0)
left_swing_points = np.concatenate(( \
        (left_liftoff_time[1:-1] + left_traj_points[0,2:-1])[None,:]/2,
        (left_traj_points[1:3,2:-1] + left_traj_points[1:3,1:-2])/2,
        np.full(max(left_liftoff_time.shape)-2, stride[2])[None,:]
        ),axis=0)

# construct foot trajectory interpolator function
right_traj_points = np.concatenate(( \
    right_traj_points, right_liftoff_points, right_swing_points), axis=1)
left_traj_points = np.concatenate(( \
    left_traj_points, left_liftoff_points, left_swing_points), axis=1)
right_traj_points = right_traj_points[:, right_traj_points[0].argsort()]
left_traj_points = left_traj_points[:, left_traj_points[0].argsort()]
right_interp = interp1d(right_traj_points[0], right_traj_points[1:])
left_interp = interp1d(left_traj_points[0], left_traj_points[1:])

# desired foot trajectory
foot_traj = np.concatenate(( \
    right_interp(t),
    left_interp(t),
    ),axis=0)

# desired torso pose trajectory
r_traj = np.array([ \
    np.mean(np.vstack((foot_traj[0],foot_traj[3])), axis=0),
    np.mean(np.vstack((foot_traj[1],foot_traj[4])), axis=0),
    np.full((2*N+1),1.4)
    ])

# desired foot location
c_des = np.array([ \
    foot_traj[0] +foot_size[0]/2,
    foot_traj[1],
    foot_traj[2],
    foot_traj[0] -foot_size[0]/2,
    foot_traj[1],
    foot_traj[2],
    foot_traj[3] +foot_size[0]/2,
    foot_traj[4],
    foot_traj[5],
    foot_traj[3] -foot_size[0]/2,
    foot_traj[4],
    foot_traj[5]
    ])

# torso pose and velocity trajectory
spline = CubicSpline(t, r_traj, axis=1)
xr_des = np.vstack((spline(t), spline(t,1)))

# guess contact force trajectory
F_des = np.tile(-m*g/4, (2*N+1, 4)).T # Fx=Fy=0, Fz=(torso mass)/(# contact points) 
F_des[2::3,:][c_des[2::3,:]!=0] = 0 # set Fz=0 when c_des not in contact with ground

# derive dynamic equation of motion
def derive_dynamics():
    xr = ca.SX.sym('xr', nr*2)
    uF = ca.SX.sym('uF', nj3)

    F_net = ca.sum2(uF.reshape((3,nj)))
    rddot = F_net/m + g

    f_sym = ca.vertcat(xr[nr:], rddot)
    f = ca.Function('f', [xr,uF], [f_sym])

    return f
 
f = derive_dynamics()

# trajectory optimization #############################################################
opti = ca.Opti()

# decision variables
XR = opti.variable(nr*2, 2*N+1) # configuration + velocity of COM position
XC = opti.variable(nj3, 2*N+1) # configuration of contact points
UF = opti.variable(nj3, 2*N+1) # contact point force

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

    for c_idx in range(nj3):
        J += Qc[c_idx]*simp[0][i]*(XC[c_idx,i]-c_des[c_idx,i])**2
        if i != 0:
            J += Qcd[c_idx]*simp[0][i]*(c_des[c_idx,i]-c_des[c_idx,i-1])**2

    for F_idx in range(nj3):
        J += RF[F_idx]*simp[0][i]*(UF[F_idx,i])**2

opti.minimize(J)

# initial condition constraint
opti.subject_to(XR[:,0] ==  xr_des[:,0])
opti.subject_to(XC[:,0] ==  c_des[:,0])

for i in range(2*N+1):
    # point mass dynamics constraints imposed through Hermite-Simpson
    if i%2 != 0:
        # for each finite element:
        xr_left, xr_mid, xr_right = XR[:,i-1], XR[:,i], XR[:,i+1]
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

    # all other constraints imposed at every timestep
    
    r_i = XR[:nr,i] # COM position at current timestep
    F_i = UF[:,i].reshape((3,nj)) # contact forces at current timestep, 3xnj matrix
    c_i = XC[:nj3,i].reshape((3,nj)) # global contact pos at current timestep, 3xnj matrix
    c_i_prev = XC[:nj3,i-1].reshape((3,nj)) # global contact pos at prev timestep, 3xnj matrix
    c_rel_i = c_i - r_i # contact pos rel to COM at current timestep, 3xnj matrix

    # zero angular momentum constraint
    torque_i = ca.MX.zeros(3,nj)
    for j in range(nj):
        torque_i[:,j] = ca.cross(c_rel_i[:,j], F_i[:,j])
    opti.subject_to(ca.sum2(torque_i) == np.zeros((3,1)))

    # foot size and orientation constraint
    opti.subject_to(c_i[:,0] - c_i[:,1] == foot_size)
    opti.subject_to(c_i[:,2] - c_i[:,3] == foot_size)

    # foot positions and reaction forces and surface coordinates
    Fs_i = Rs.T @ F_i
    cs_i = Rs.T @ (c_i - ps)
    cds_i = Rs.T @ (c_i - c_i_prev)

    # friciton cone constraints in surface coordinates
    opti.subject_to(Fs_i[2,:] >= np.zeros(Fs_i[2,:].shape))
    opti.subject_to(opti.bounded(-mu*Fs_i[2,:], Fs_i[0,:], mu*Fs_i[2,:]))
    opti.subject_to(opti.bounded(-mu*Fs_i[2,:], Fs_i[1,:], mu*Fs_i[2,:]))

    # contact constraints in surface coordinates
    opti.subject_to(cs_i[2,:] >= np.zeros(cs_i[2,:].shape))
    opti.subject_to(cs_i[2,:] * Fs_i[2,:] == np.zeros(Fs_i[2,:].shape))
    if i != 0:
        opti.subject_to(cds_i[0,:] * Fs_i[2, :] == np.zeros(Fs_i[2, :].shape))
        opti.subject_to(cds_i[1,:] * Fs_i[2, :] == np.zeros(Fs_i[2, :].shape))

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
XC_sol = np.array(sol.value(XC))
UF_sol = np.array(sol.value(UF))

# animate
axis_lim = 1.5
F_len = 1/np.max(UF_sol)

for i in range(2*N+1):
    r_i = np.array(XR_sol[:nr,i])
    p_feet_i = XC_sol[:,i].reshape((3,nj), order='F')
    F_i = np.array(UF_sol[:,i])

    p_i = {}
    p_i['r']  = r_i[:,None]
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
torso_coord = np.zeros((3, 1, 2*N+1))
F_coord = np.zeros((nj, 3, 2, 2*N+1)) #(# forces)*(cartesian space)*(# datapoints)*(# time points) 
for xyz in range(3):
    foot_r_coord[xyz,:,:] = np.array([p['rc1'][xyz,:], p['rc2'][xyz,:]])
    foot_l_coord[xyz,:,:] = np.array([p['lc1'][xyz,:], p['lc2'][xyz,:]])
    torso_coord[xyz,:,:] = np.array([p['r'][xyz,:]])
    for j, key in enumerate(['rc1', 'rc2', 'lc1', 'lc2']):
        F_coord[j,xyz,:,:] = np.array([p[key][xyz,:], F_vec[key][xyz,:]])

anim_fig = plt.figure(figsize=(12, 12))
ax = Axes3D(anim_fig)
lines = [plt.plot([], [])[0] for _ in range(4+2*nj)]

def animate(i):
    for j in range(nj):
        lines[j].set_data(c_des[3*j,i], c_des[3*j+1,i])
        lines[j].set_3d_properties(c_des[3*j+2,i])

    lines[nj].set_data(xr_des[0,i], xr_des[1,i])
    lines[nj].set_3d_properties(xr_des[2,i])

    lines[nj+1].set_data(foot_r_coord[0,:,i], foot_r_coord[1,:,i])
    lines[nj+1].set_3d_properties(foot_r_coord[2,:,i])
    lines[nj+2].set_data(foot_l_coord[0,:,i], foot_l_coord[1,:,i])
    lines[nj+2].set_3d_properties(foot_l_coord[2,:,i])

    lines[nj+3].set_data(torso_coord[0,:,i], torso_coord[1,:,i])
    lines[nj+3].set_3d_properties(torso_coord[2,:,i])

    for j in range(nj):
        lines[nj+4+j].set_data(F_coord[j,0,:,i], F_coord[j,1,:,i])
        lines[nj+4+j].set_3d_properties(F_coord[j,2,:,i])

    ax.view_init(azim=i/2)
    ax.set_xlim3d([torso_coord[0,:,i]-1, torso_coord[0,:,i]+1])
    ax.set_ylim3d([torso_coord[1,:,i]-1, torso_coord[1,:,i]+1])


    return lines

#ax.view_init(azim=45)
#ax.set_xlim3d([-1, 1])
#ax.set_ylim3d([-1, 1])
ax.set_zlim3d([0, 2])

for line in lines[:nj]:
    line.set_color('c')
    line.set_marker('o')
    line.set_markeredgewidth(7)

lines[nj].set_color('c')
lines[nj].set_marker('o')
lines[nj].set_markeredgewidth(7)

for line in lines[nj+1:nj+3]:
    line.set_color('g')
    line.set_linewidth(5)
    line.set_marker('o')
    line.set_markeredgewidth(7)

lines[nj+3].set_color('b')
lines[nj+3].set_linewidth(5)
lines[nj+3].set_marker('o')
lines[nj+3].set_markeredgewidth(7)

for line in lines[nj+4:]:
    line.set_color('r')
    line.set_linewidth(3)

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=2*N+1, 
    interval=tf*1000/(2*N+1), repeat=True, blit=False)

'''
# uncomment to write to file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=int((2*N+1)/tf), metadata=dict(artist='Me'), bitrate=1000)
anim.save('point_mass_run_random2' + '.mp4', writer=writer)
'''

plt.show()