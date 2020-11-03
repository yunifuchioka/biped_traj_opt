# test point mass dynamics in 2D setting

import numpy as np
from numpy import pi
import casadi as ca

import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.style.use('seaborn')

nr = 2 # dimension of centroid configuration vector
nF = 2*2 # dimensino of force vector

c1 = np.array([-1, 0])
c2 = np.array([1, 0])
r_des = np.array([0.5, 1.5])
r_init = np.array([0, 0.5])

tf = 20.0 # final time
N = 200 # number of finite elements
t_traj = np.linspace(0, tf, 2*N+1) # collocated time

m = 20
g = np.array([0, -9.81])

Qr = np.array([1000, 1000])
Qrd = np.array([100, 100])
RF = np.array([0.01, 0.01, 0.01, 0.01])

def derive_dynamics():
    xr = ca.SX.sym('xr', nr*2)
    uF = ca.SX.sym('uF', nF)
    
    #F_reshape = ca.reshape(uF, (2,2))
    F_net = ca.reshape(uF, (2,2)) @ np.full((2,1),1)
    rddot = F_net/m + g

    f_sym = ca.vertcat(xr[nr:], rddot)
    f = ca.Function('f', [xr,uF], [f_sym])

    return f

f = derive_dynamics()

# trajectory optimization
opti = ca.Opti()

XR = opti.variable(nr*2, 2*N+1)
UF = opti.variable(nF, 2*N+1)

# cost
# simpson quadrature coefficients, to be used to compute integrals
simp = np.empty((1,2*N+1))
simp[0,::2] = 2
simp[0,1::2] = 4
simp[0,0], simp[0,-1]  = 1, 1

J = 0.0
for i in range(2*N+1):
    for r_idx in range(nr):
        J += Qr[r_idx]*simp[0][i]*(XR[r_idx,i]-r_des[r_idx])*(XR[r_idx,i]-r_des[r_idx])
        J += Qrd[r_idx]*simp[0][i]*(XR[nr+r_idx,i])**2

    for F_idx in range(nF):
        J += RF[F_idx]*simp[0][i]*(UF[F_idx,i])**2

opti.minimize(J)

# initial condition constraint
opti.subject_to(XR[:,0] ==  np.hstack((r_init, np.zeros(r_init.shape))))

for i in range(2*N+1):
    if i%2 != 0:
        # for each finite element:
        xr_left, xr_mid, xr_right = XR[:,i-1], XR[:,i], XR[:,i+1]
        uF_left, uF_mid, uF_right = UF[:,i-1], UF[:,i], UF[:,i+1]
        
        f_left, f_mid, f_right = \
            f(xr_left, uF_left), f(xr_mid, uF_mid), \
            f(xr_right, uF_right)

        # interpolation constraints
        opti.subject_to( \
            # equation (6.11) in Kelly 2017
            xr_mid == (xr_left+xr_right)/2.0 + tf/N*(f_left-f_right)/8.0)

        # collocation constraints
        opti.subject_to( \
            # equation (6.12) in Kelly 2017
            tf/N*(f_left+4*f_mid+f_right)/6.0 == xr_right-xr_left)

    # zero angular momentum
    torque = (c1[0] - XR[0,i])*UF[1,i] - (c1[1] - XR[1,i])*UF[0,i]
    torque += (c2[0]- XR[0,i])*UF[3,i] - (c2[1]- XR[1,i])*UF[2,i]
    opti.subject_to(torque == np.zeros(torque.shape))

p_opts = {}
s_opts = {'print_level': 5}
opti.solver('ipopt', p_opts, s_opts)
sol = opti.solve()

r = np.array(sol.value(XR))
F = np.array(sol.value(UF))

for i in range(2*N+1):
    o_r_i = r[:nr,i][:,None]
    o_c1_i = c1[:,None]
    o_c2_i = c2[:,None]

    F_c1_i = F[:2,i][:,None]
    F_c2_i = F[2:,i][:,None]

    if i == 0:
        o_r = o_r_i
        o_c1 = o_c1_i
        o_c2 = o_c2_i
        F_c1 = F_c1_i
        F_c2 = F_c2_i
    else:
        o_r = np.hstack((o_r, o_r_i))
        o_c1 = np.hstack((o_c1, o_c1_i))
        o_c2 = np.hstack((o_c2, o_c2_i))
        F_c1 = np.hstack((F_c1, F_c1_i))
        F_c2 = np.hstack((F_c2, F_c2_i))

r_x_points = np.array([r[0,:]])
r_y_points = np.array([r[1,:]])

F_len = 0.01
F_x_points = np.zeros((2,2,2*N+1))
F_y_points = np.zeros((2,2,2*N+1))
F_x_points[0,:,:] = np.array([o_c1[0,:], o_c1[0,:]+F_len*F_c1[0,:]])
F_y_points[0,:,:] = np.array([o_c1[1,:], o_c1[1,:]+F_len*F_c1[1,:]])
F_x_points[1,:,:] = np.array([o_c2[0,:], o_c2[0,:]+F_len*F_c2[0,:]])
F_y_points[1,:,:] = np.array([o_c2[1,:], o_c2[1,:]+F_len*F_c2[1,:]])

anim_fig = plt.figure(figsize=(12, 12))
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-0.5, 2.5))
lines = [plt.plot([], [])[0] for _ in range(4)]

def animate(i):
    lines[0].set_data(r_x_points[:,i], r_y_points[:,i])
    lines[1].set_data(r_des[0], r_des[1])

    lines[2].set_data(F_x_points[0,:,i], F_y_points[0,:,i])
    lines[3].set_data(F_x_points[1,:,i], F_y_points[1,:,i])


    lines[0].set_color('b')
    lines[1].set_color('r')
    for line in lines[:2]:
        line.set_linewidth(5)
        line.set_marker('o')
        line.set_markeredgewidth(7)

    for line in lines[2:]:
        line.set_color('g')
        line.set_linewidth(5)
        line.set_marker('.')
        line.set_markeredgewidth(7)

    return lines

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=2*N+1, 
    interval=tf*1000/(2*N+1), repeat=True, blit=True)

plt.show()