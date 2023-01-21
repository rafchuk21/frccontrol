import examples.double_jointed_arm as dja
import numpy as np

arm = dja.DoubleJointedArm(.02)
state = np.array([[0.41595911, -0.41897834, -5.33397834, 16.02058829]]).T
(M,C,G) = arm.get_dynamics_matrices(state)

print(np.linalg.inv(M))
print(C-arm.constants.K4)
print(G)


uff = arm.feed_forward(state)
f = arm.dynamics_real

arm.relinearize(state, uff)
q_pos = 0.01745*10
q_vel = 0.08726*10

arm.design_lqr([q_pos, q_pos, q_vel, q_vel], [12.0, 12.0])

K = arm.K

ref = np.array([[ 0.54936878, -0.58802826,  1.06361767, -1.13846523]]).T

def runge_kutta(f, x, u, dt):
    """Fourth order Runge-Kutta integration.

    Keyword arguments:
    f -- vector function to integrate
    x -- vector of states
    u -- vector of inputs (constant for dt)
    dt -- time for which to integrate
    """
    half_dt = dt * 0.5
    k1 = f(x, u)
    print("\t" + np.array2string(k1.T))
    k2 = f(x + half_dt * k1, u)
    print("\t" + np.array2string(k2.T))
    k3 = f(x + half_dt * k2, u)
    print("\t" + np.array2string(k3.T))
    k4 = f(x + dt * k3, u)
    print("\t" + np.array2string(k4.T))
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

u = K @ (ref - state) + arm.feed_forward(ref)

print(state.T)
state2 = runge_kutta(f, state, np.clip(u, -12, 12), .02)
print(state2.T)