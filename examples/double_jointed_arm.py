#!/usr/bin/env python3

"""frccontrol example for a double-jointed arm."""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import StateSpace
import matplotlib.animation as animation

import frccontrol as fct

if "--noninteractive" in sys.argv:
    mpl.use("svg")

class DoubleJointedArm(fct.System):
    """An frccontrol system representing a double-jointed arm."""

    def __init__(self, dt, start_state = np.zeros((6,1))):
        """Double-jointed arm subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        """
        state_labels = [("Angle 1", "rad"), ("Angle 2", "rad"), \
            ("Angular velocity 1", "rad/s"), ("Angular velocity 2", "rad/s"), \
            ("Input Error 1", "Nm"), ("Input Error 2", "Nm")]
        u_labels = [("Voltage 1", "V"), ("Voltage 2", "V")]
        self.set_plot_labels(state_labels, u_labels)
        self.geometry = DoubleJointedArmGeometry()

        fct.System.__init__(
            self,
            np.array([[-12.0]]),
            np.array([[12.0]]),
            dt,
            start_state,
            np.zeros((2, 1)),
            self.dynamics
        )

    # pragma pylint: disable=signature-differs
    def create_model(self, states, inputs):
        """Relinearize model around given state.

        Keyword arguments:
        states -- state vector around which to linearize model (if applicable)
        inputs -- input vector around which to linearize model (if applicable)

        Returns:
        StateSpace instance containing continuous state-space model
        """
        nstates = states.shape[0]
        ninputs = inputs.shape[0]
        self.set_dynamics(states)
        A = fct.numerical_jacobian_x(nstates, nstates, self.f, states, inputs)
        B = fct.numerical_jacobian_u(nstates, ninputs, self.f, states, inputs)
        C = np.concatenate((np.eye(2), np.zeros((2, nstates - 2))), 1)
        D = np.zeros((2, ninputs))

        return StateSpace(A, B, C, D)

    def design_controller_observer(self):
        self.relinearize(self.x_hat, self.feed_forward(self.x_hat))

        q_pos = 0.01745
        q_vel = 0.08726
        
        self.design_lqr([q_pos, q_pos, q_vel, q_vel], [12.0, 12.0])
        self.design_two_state_feedforward()

        q_pos = 0.01745
        q_vel = 0.1745329
        est = 10
        r_pos = 0.05
        self.design_kalman_filter([q_pos, q_pos, q_vel, q_vel, est, est], [r_pos, r_pos])

    def design_lqr(self, Q_elems, R_elems):
        """Design a discrete time linear-quadratic regulator for the system.

        Keyword arguments:
        Q_elems -- a vector of the maximum allowed excursions of the states from
                   the reference.
        R_elems -- a vector of the maximum allowed excursions of the control
                   inputs from no actuation.
        """
        Q = np.diag(1.0 / np.square(Q_elems))
        R = np.diag(1.0 / np.square(R_elems))
        Ar = self.sysc.A[:4,:4]
        Br = self.sysc.B[:4,:]
        Cr = self.sysc.C[:,:4]
        sysd_reduced = StateSpace(Ar, Br, Cr, self.sysc.D).to_discrete(self.dt)
        self.K = fct.lqr(sysd_reduced, Q, R)
    
    def update_plant(self):
        """Advance the model by one timestep."""
        self.x = fct.runge_kutta(self.dynamics_real, self.x, self.u, self.dt)
        self.y = self.sysd.C @ self.x + self.sysd.D @ self.u + np.array([np.random.normal(0, .01, 2)]).T

    def update_controller(self, next_r = None):
        """Advance the controller by one timestep.

        Keyword arguments:
        next_r -- next controller reference (default: current reference)
        """
        u = self.K @ np.array([(self.r[0:4] - self.x_hat[0:4,0])]).T
        uff = self.feed_forward(np.array([next_r]).T)

        self.r = next_r
        self.u = np.clip(u + uff - np.linalg.inv(self.K3) @ self.x_hat[4:], self.u_min, self.u_max)

    def set_dynamics(self, states):
        # Mass of segments
        m1 = 9.34 * .4536
        m2 = 9.77 * .4536

        # Distance from pivot to CG for each segment
        r1 = 21.64 * .0254
        r2 = 26.70 * .0254

        # Moment of inertia about CG for each segment
        I1 = 2957.05 * .0254*.0254 * .4536
        I2 = 2824.70 * .0254*.0254 * .4536

        # Gearing of each segment
        G1 = 140.
        G2 = 90.

        # Number of motors in each gearbox
        N1 = 1
        N2 = 2

        # Gravity
        g = 9.81

        stall_torque = 3.36
        free_speed = 5880.0 * 2.0*np.pi/60.0
        stall_current = 166

        Rm = 12.0/stall_current

        Kv = free_speed / 12.0
        Kt = stall_torque / stall_current

        # K3*Voltage - K4*velocity = motor torque
        self.K3 = np.array([[N1*G1, 0], [0, N2*G2]])*Kt/Rm
        self.K4 = np.array([[G1*G1*N1, 0], [0, G2*G2*N2]])*Kt/Kv/Rm

        [theta1, theta2, omega1, omega2] = states[:4].flat
        c2 = np.cos(theta2)

        hM = self.geometry.l1*r2*c2
        self.M = m1*np.array([[r1*r1, 0], [0, 0]]) + m2*np.array([[self.geometry.l1*self.geometry.l1 + r2*r2 + 2*hM, r2*r2 + hM], [r2*r2 + hM, r2*r2]]) + \
            I1*np.array([[1, 0], [0, 0]]) + I2*np.array([[1, 1], [1, 1]])

        hC = -m2*self.geometry.l1*r2*np.sin(theta2)
        self.C = np.array([[hC*omega2, hC*omega1 + hC*omega2], [-hC*omega1, 0]])

        self.G = g*np.cos(theta1) * np.array([[m1*r1 + m2*self.geometry.l1, 0]]).T + \
            g*np.cos(theta1+theta2) * np.array([[m2*r2, m2*r2]]).T

    def dynamics(self, states, inputs):
        omega_vec = states[2:4]

        basic_torque = self.K3 @ inputs
        if states.shape[0] == 6:
            basic_torque = basic_torque + states[4:]
        back_emf_loss = self.K4 @ omega_vec

        torque = basic_torque - back_emf_loss
        alpha_vec = np.linalg.inv(self.M) @ (torque - self.C @ omega_vec - self.G)
        state_dot = np.concatenate((omega_vec, alpha_vec))
        if states.shape[0] == 6:
            state_dot = np.concatenate((state_dot, np.zeros((2,1))))
        return state_dot
    
    def dynamics_real(self, states, inputs):
        omega_vec = states[2:4]

        basic_torque = self.K3 @ inputs
        back_emf_loss = self.K4 @ omega_vec
        disturbance_torque = np.zeros((2,1))

        G = self.G

        #disturbance_torque = disturbance_torque + np.array([[150, -90]]).T
        #basic_torque = basic_torque * .5
        G = G * 2

        torque = basic_torque - back_emf_loss + disturbance_torque
        alpha_vec = np.linalg.inv(self.M) @ (torque - self.C @ omega_vec - G)
        state_dot = np.concatenate((omega_vec, alpha_vec))
        if states.shape[0] == 6:
            state_dot = np.concatenate((state_dot, np.zeros((2,1))))
        return state_dot

    def feed_forward(self, state, accels = np.zeros((2,1))):
        omegas = state[2:4]
        return np.linalg.inv(self.K3) @ (self.M @ accels  + self.C @ omegas  + self.G + self.K4 @ omegas)

class DoubleJointedArmGeometry(object):
    def __init__(self):
        # Length of segments
        self.l1 = 46.25 * .0254
        self.l2 = 41.80 * .0254
    
    def fwd_kinematics(self, state):
        [theta1, theta2] = state[:2]
        joint2 = np.array([self.l1*np.cos(theta1), self.l1*np.sin(theta1)])
        end_eff = joint2 + np.array([self.l2*np.cos(theta1 + theta2), self.l2*np.sin(theta1 + theta2)])
        return (joint2, end_eff)
    
    def inv_kinematics(self, pos, invert = False):
        """Inverse kinematics for a target position pos (x,y). Invert controls elbow direction."""
        [x,y] = pos.flat
        theta2 = np.arccos((x*x + y*y - (self.l1*self.l1 + self.l2*self.l2)) / \
            (2*self.l1*self.l2))

        if invert:
            theta2 = -theta2
        
        theta1 = np.arctan2(y, x) - np.arctan2(self.l2*np.sin(theta2), self.l1 + self.l2*np.cos(theta2))
        return np.array([[theta1, theta2]]).T

def main():
    """Entry point."""

    dt = 0.005

    geometry = DoubleJointedArmGeometry()

    state1 = geometry.inv_kinematics(np.array([1.5, -1]), False)
    state2 = geometry.inv_kinematics(np.array([1.5, 1]), True)
    state3 = geometry.inv_kinematics(np.array([-1.8, 1]), False)

    state1 = np.concatenate((state1, np.zeros((2,1))))
    state2 = np.concatenate((state2, np.zeros((2,1))))
    state3 = np.concatenate((state3, np.zeros((2,1))))

    traj1 = fct.trajectory.interpolate_states(0, 3, state1, state2)
    traj2 = fct.trajectory.interpolate_states(4, 8, state2, state3)

    traj = traj1.append(traj2)

    tvec = np.arange(0, 10 + dt, dt)

    start_state = np.concatenate((state1, np.zeros((2,1))))
    double_jointed_arm = DoubleJointedArm(dt, start_state)

    # Generate references for simulation
    refs = []
    for i, _ in enumerate(tvec):
        r = np.append(traj.sample(tvec[i])[:4], np.zeros(2))
        #print(r)
        refs.append(r)

    x_rec, ref_rec, u_rec, _ = double_jointed_arm.generate_time_responses(refs)
    double_jointed_arm.plot_time_responses(tvec, x_rec, ref_rec, u_rec)
    indices = np.arange(0, len(tvec), 10)
    if "--noninteractive" in sys.argv:
        plt.savefig("double_jointed_arm_response.svg")
    else:
        animate_arm(double_jointed_arm, tvec[indices], x_rec[:, indices], ref_rec[:, indices])
        #plt.show()

def get_col(m, i):
    return np.array([m[:,i]]).T

def animate_arm(arm, t, states, target_states):
    def get_arm_joints(state):
        """Get the xy positions of all three robot joints (?) - base joint (at 0,0), elbow, end effector"""
        (joint_pos, eff_pos) = arm.geometry.fwd_kinematics(state)
        x = np.array([0, joint_pos[0,0], eff_pos[0,0]])
        y = np.array([0, joint_pos[1,0], eff_pos[1,0]])
        return (x,y)

    dt = t[4] - t[3]
    print(dt)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis('square')
    ax.grid(True)
    total_len = arm.geometry.l1 + arm.geometry.l2
    ax.set_xlim(-total_len, total_len)
    ax.set_ylim(-total_len, total_len)
    (xs, ys) = get_arm_joints(get_col(states, 0))
    target_line, arm_line = ax.plot(xs, ys, 'b--o', xs, ys, 'r-o')
    ax.legend([arm_line, target_line], ["Current State", "Target State"], loc='lower left')

    def init():
        (xs, ys) = get_arm_joints(get_col(states, 0))
        target_line.set_data(xs, ys)
        arm_line.set_data(xs, ys)
        ax.set_xlim(-total_len, total_len)
        ax.set_ylim(-total_len, total_len)
        return target_line, arm_line

    def animate(i):
        (xs, ys) = get_arm_joints(get_col(target_states, i))
        target_line.set_data(xs, ys)
        (xs, ys) = get_arm_joints(get_col(states, i))
        arm_line.set_data(xs, ys)
        ax.set_xlim(-total_len, total_len)
        ax.set_ylim(-total_len, total_len)
        return target_line, arm_line
    
    nframes = len(t)
    anim = animation.FuncAnimation(fig, animate, init_func = init, frames = nframes, interval = int(dt*1000), blit=False, repeat=True)
    #plt.show()
    anim.save('frccontrol_sim_double_gravity.gif', writer='imagemagick')

if __name__ == "__main__":
    main()
