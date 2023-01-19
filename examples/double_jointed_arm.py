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
    """An frccontrol system representing a double-jointed arm.
    The state is:
        [angle 1, angle 2, omega 1, omega 2, input error 1, input error 2]
    
    Since a double-jointed arm is a nonlinear system, we provide it with a
    function to predict its dynamics at a given state. This is done in the
    dynamics() function, which takes a current state and input voltages and
    returns the predicted rate of change of the state, as:
        [omega 1, omega 2, alpha 1, alpha 2, 0, 0]
    
    By taking the jacobian of the dynamics() function, the controller can
    simulate the system as being locally linear, following an equation of the
    form:
        X-dot = A*X + B*U
    where X is the current state, X-dot is the rate of change of the state, and
    U is the input voltages. Note that these A and B are for the continuous
    model of the system - since we are modelling it discretely, we also need to
    discretize the matrices.

    The system also has an observer that reads the encoders, cleans up the
    data, and generally tries to get the most accurate possible estimate of
    the state. This is done using an Extended Kalman Filter, which in essence
    reads the encoder values and incorporates them into its state estimate
    according to how much it trusts the encoders, then uses its state estimate
    and the model of the system to predict the next state of the system, then
    compares that to the encoder readings in the next loop, and so on.
    
    Finally, a real-world arm controller will likely not have a perfect model
    of the arm it's controlling. To integrate that into the simulation, there
    is also a dynamics_real() function, which represents the true dynamics of
    the system. The controller doesn't know what dynamics_real() is, nor can it
    read the true state directly - it can only do so through encoders, which
    will have some error and noise in them. The final two parts of the state,
    the input errors, are where the state observer will try to estimate how
    off its model is from the true model. In this implementation, the values
    input errors are measured in Newton-meters. You can try adding various
    disturbances into dynamics_real(), which the controller won't know directly
    but maybe be able to compensate for.

    The main controller loop is impemented in the System superclass and follows
    the following steps:
        1. update the plant --  update the true system's position and the
                                encoder readings using dynamics_real().
        2. predict the new state -- Use the nonlinear model to predict what this
                                    next state will be.
        3. correct the observer --  Use the encoder readings to update the
                                    observer state estimate, based on its
                                    trust values for the encoders.
        4. update the controller -- Relinearize the system model and determine
                                    the next voltages to apply.
    
    These functions were implemented in the System superclass, but several had
    to be overridden to implement nonlinear control. This class also does not
    use the System Kff method to find feed forward, and instead determines the
    feed forward from the nonlinear model.
    
    """

    def __init__(self, dt, start_state = np.zeros((6,1))):
        """Double-jointed arm subsystem.

        Keyword arguments:
        dt -- time between model/controller updates
        start_state -- initial state of the system
        """
        # Prepare plot labels
        state_labels = [("Angle 1", "rad"), ("Angle 2", "rad"), \
            ("Angular velocity 1", "rad/s"), ("Angular velocity 2", "rad/s"), \
            ("Input Error 1", "Nm"), ("Input Error 2", "Nm")]
        u_labels = [("Voltage 1", "V"), ("Voltage 2", "V")]
        self.set_plot_labels(state_labels, u_labels)

        # Store the arm constants
        self.constants = DoubleJointedArmConstants()

        # Initialize the superclass System
        fct.System.__init__(
            self,
            np.array([[-12.0]]),
            np.array([[12.0]]),
            dt,
            start_state,
            np.zeros((2, 1)),
            self.dynamics       # this is the nonlinear model
        )
    
    """
    -------------------
    MAIN LOOP OVERRIDES
    -------------------
    """

    def update_plant(self):
        """Advance the model by one timestep.
        
        Updates the true state (which the controller can't see) and the
        observed state (which the controller can see).
        """
        self.x = fct.runge_kutta(self.dynamics_real, self.x, self.u, self.dt)
        # Observed state: has some added noise
        self.y = self.sysd.C @ self.x + self.sysd.D @ self.u + \
            np.array([np.random.normal(0, .01, 2)]).T

    # correct_observer() step inherited without changes

    def update_controller(self, next_r = None):
        """Update the controller given a new reference.

        Relinearizes the system and 

        Keyword arguments:
        next_r -- next controller reference (default: current reference)
        """
        # Relinearize and run LQR
        self.design_controller_observer()

        # Voltage output proportional to error:
        u = self.K @ np.array([(self.r[0:4] - self.x_hat[0:4,0])]).T
        # Voltage output from feedforward:
        uff = self.feed_forward(np.array([next_r]).T)
        # Voltage output from input error estimation. Note that the estimate
        # is in Newton-meters, so has to be converted to volts:
        uerr = np.linalg.inv(self.constants.K3) @ self.x_hat[4:]

        self.r = next_r
        self.u = np.clip(u + uff - uerr, self.u_min, self.u_max)

    # predict_observer() step inherited with no changes.

    """
    ---------------
    RELINEARIZATION
    ---------------
    """

    def design_controller_observer(self):
        """Runs LQR and updates the kalman filter parameters after
        relinearizing the system.
        """
        self.relinearize(self.x_hat, self.feed_forward(self.x_hat))

        q_pos = 0.01745
        q_vel = 0.08726
        
        self.design_lqr([q_pos, q_pos, q_vel, q_vel], [12.0, 12.0])

        q_pos = 0.01745
        q_vel = 0.1745329
        est = 10
        r_pos = 0.05
        self.design_kalman_filter([q_pos, q_pos, q_vel, q_vel, est, est], [r_pos, r_pos])

    # relinearize() inherited without changes.

    def create_model(self, states, inputs):
        """Relinearize model around given state.
        This returns a continuous state-space model - must be discretized to
        the controller loop time before use!

        Keyword arguments:
        states -- state vector around which to linearize model (if applicable)
        inputs -- input vector around which to linearize model (if applicable)

        Returns:
        StateSpace instance containing continuous state-space model
        """
        nstates = states.shape[0]
        ninputs = inputs.shape[0]
        A = fct.numerical_jacobian_x(nstates, nstates, self.f, states, inputs)
        B = fct.numerical_jacobian_u(nstates, ninputs, self.f, states, inputs)
        C = np.concatenate((np.eye(2), np.zeros((2, nstates - 2))), 1)
        D = np.zeros((2, ninputs))

        return StateSpace(A, B, C, D)

    def design_lqr(self, Q_elems, R_elems):
        """Design a discrete time linear-quadratic regulator for the system.

        This extracts the 4-state linear model (ignoring input error estimates)
        from the last linearized continuous state-space model, then discretizes
        it to find the opimal K using LQR.

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
    
    """
    --------
    DYNAMICS
    --------
    """

    def get_dynamics_matrices(self, states):
        """Gets the dynamics matrices for the given state.
        
        See derivation at:
        https://www.chiefdelphi.com/t/whitepaper-two-jointed-arm-dynamics/423060

        Keyword arguments:
        states -- current system state
        """
        
        [theta1, theta2, omega1, omega2] = states[:4].flat
        c2 = np.cos(theta2)

        const = self.constants

        l1 = const.l1
        r1 = const.r1
        r2 = const.r2
        m1 = const.m1
        m2 = const.m2
        I1 = const.I1
        I2 = const.I2
        g = const.g

        hM = l1*r2*c2
        M = m1*np.array([[r1*r1, 0], [0, 0]]) + m2*np.array([[l1*l1 + r2*r2 + 2*hM, r2*r2 + hM], [r2*r2 + hM, r2*r2]]) + \
            I1*np.array([[1, 0], [0, 0]]) + I2*np.array([[1, 1], [1, 1]])

        hC = -m2*l1*r2*np.sin(theta2)
        C = np.array([[hC*omega2, hC*omega1 + hC*omega2], [-hC*omega1, 0]])

        G = g*np.cos(theta1) * np.array([[m1*r1 + m2*self.constants.l1, 0]]).T + \
            g*np.cos(theta1+theta2) * np.array([[m2*r2, m2*r2]]).T
        
        return (M, C, G)

    def dynamics(self, states, inputs):
        """Models the rate of change of the system's state given the current
        state and the voltage inputs.
        
        Note that this is what the controller thinks the system will do, not
        necessarily what the system will actually do. This nuance allows for
        us to see how the controller responds to differences between the model
        and reality.
        
        Keyword arguments:
        states -- current state
        inputs -- voltage inputs
        """
        (M, C, G) = self.get_dynamics_matrices(states)
        omega_vec = states[2:4]

        basic_torque = self.constants.K3 @ inputs
        if states.shape[0] == 6: # add the input error estimate, if present
            basic_torque = basic_torque + states[4:]
        back_emf_loss = self.constants.K4 @ omega_vec

        torque = basic_torque - back_emf_loss
        alpha_vec = np.linalg.inv(M) @ (torque - C @ omega_vec - G)
        state_dot = np.concatenate((omega_vec, alpha_vec))
        if states.shape[0] == 6:
            state_dot = np.concatenate((state_dot, np.zeros((2,1))))
        return state_dot
    
    def dynamics_real(self, states, inputs):
        """Gets the rate of change of the system's state given the current
        state and the voltage inputs.
        
        Note that this is what the system will actually do, not what the
        controller thinks the system will do. This nuance allows for us to see
        how the controller responds to differences between the model and
        reality.
        
        Keyword arguments:
        states -- current state
        inputs -- voltage inputs
        """
        (M, C, G) = self.get_dynamics_matrices(states)

        omega_vec = states[2:4]

        basic_torque = self.constants.K3 @ inputs
        back_emf_loss = self.constants.K4 @ omega_vec
        disturbance_torque = np.zeros((2,1))

        # Some example disturbances:
        #disturbance_torque = disturbance_torque + np.array([[150, -90]]).T
        #basic_torque = basic_torque * .5
        #G = G * 2

        torque = basic_torque - back_emf_loss + disturbance_torque
        alpha_vec = np.linalg.inv(M) @ (torque - C @ omega_vec - G)
        state_dot = np.concatenate((omega_vec, alpha_vec))
        if states.shape[0] == 6:
            state_dot = np.concatenate((state_dot, np.zeros((2,1))))
        return state_dot

    def feed_forward(self, states, accels = np.zeros((2,1))):
        (M, C, G) = self.get_dynamics_matrices(states)
        omegas = states[2:4]
        return np.linalg.inv(self.constants.K3) @ (M @ accels  + C @ omegas \
            + G + self.constants.K4 @ omegas)

class DoubleJointedArmConstants(object):
    def __init__(self):
        # Length of segments
        self.l1 = 46.25 * .0254
        self.l2 = 41.80 * .0254

        # Mass of segments
        self.m1 = 9.34 * .4536
        self.m2 = 9.77 * .4536

        # Distance from pivot to CG for each segment
        self.r1 = 21.64 * .0254
        self.r2 = 26.70 * .0254

        # Moment of inertia about CG for each segment
        self.I1 = 2957.05 * .0254*.0254 * .4536
        self.I2 = 2824.70 * .0254*.0254 * .4536

        # Gearing of each segment
        self.G1 = 140.
        self.G2 = 90.

        # Number of motors in each gearbox
        self.N1 = 1
        self.N2 = 2

        # Gravity
        self.g = 9.81

        self.stall_torque = 3.36
        self.free_speed = 5880.0 * 2.0*np.pi/60.0
        self.stall_current = 166

        self.Rm = 12.0/self.stall_current

        self.Kv = self.free_speed / 12.0
        self.Kt = self.stall_torque / self.stall_current

        # K3*Voltage - K4*velocity = motor torque
        self.K3 = np.array([[self.N1*self.G1, 0], [0, self.N2*self.G2]])*self.Kt/self.Rm
        self.K4 = np.array([[self.G1*self.G1*self.N1, 0], [0, self.G2*self.G2*self.N2]])*self.Kt/self.Kv/self.Rm
    
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

    constants = DoubleJointedArmConstants()

    state1 = constants.inv_kinematics(np.array([1.5, -1]), False)
    state2 = constants.inv_kinematics(np.array([1.5, 1]), True)
    state3 = constants.inv_kinematics(np.array([-1.8, 1]), False)

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
        (joint_pos, eff_pos) = arm.constants.fwd_kinematics(state)
        x = np.array([0, joint_pos[0,0], eff_pos[0,0]])
        y = np.array([0, joint_pos[1,0], eff_pos[1,0]])
        return (x,y)

    dt = t[4] - t[3]
    print(dt)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.axis('square')
    ax.grid(True)
    total_len = arm.constants.l1 + arm.constants.l2
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
    plt.show()
    #anim.save('frccontrol_sim.gif', writer='imagemagick')

if __name__ == "__main__":
    main()
