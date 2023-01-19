from __future__ import annotations
import numpy as np

class Trajectory(object):

    def __init__(self, times: np.matrix, states: np.matrix) -> Trajectory:
        """Initialize a Trajectory.
        
        Arguments:
            times: Column vector of trajectory timestamps.
            states: Matrix of corresponding states, where each state is a column
        """
        self.times = times
        self.states = states
        self.start_time = times[0]
        self.end_time = times[-1]
    
    def clip_time(self, time):
        """Limit Trajectory timestamp between start_time and end_time."""
        return np.clip(time, self.start_time, self.end_time)

    def sample(self, time):
        """ Sample the trajectory for the given time.
            Linearly interpolates between trajectory samples.
            If time is outside of trajectory, gives the start/end state.
        
        Arguments:
            time: time to sample
        """
        time = self.clip_time(time)
        prev_idx = np.where(self.times <= time)[0][-1]
        next_idx = np.where(self.times >= time)[0][0]

        if prev_idx == next_idx:
            return np.array([self.states[:, prev_idx]]).T
        
        prev_val = np.array([self.states[:,prev_idx]]).T
        next_val = np.array([self.states[:,next_idx]]).T
        prev_time = self.times[prev_idx]
        next_time = self.times[next_idx]

        return (next_val - prev_val)/(next_time - prev_time)*(time-prev_time) + prev_val
    
    def append(self, other: Trajectory) -> Trajectory:
        """ Append another trajectory to this trajectory.
            Will adjust timestamps on the appended trajectory so it starts immediately after the
            current trajectory ends.
            Skips the first element of the other trajectory to avoid repeats.
            
        Arguments:
            other: The other trajectory to append to this one.
        """

        # Create new trajectory based off of this one
        combined = Trajectory(self.times, self.states)
        # Adjust timestamps on other trajectory
        other.times = other.times + combined.end_time - other.start_time
        # Combine the time and states
        combined.times = np.concatenate((combined.times, other.times[1:]))
        combined.states = np.concatenate((combined.states, other.states[:,1:]), 1)
        # Update the end time
        combined.end_time = max(combined.times)
        return combined
    
    
    def to_table(self) -> np.ndarray:
        return np.concatenate((self.times, self.states.T), 1)

def from_coeffs(coeffs: np.matrix, t0, tf, n = 100) -> Trajectory:
        """ Generate a trajectory from a polynomial coefficients matrix.
        
        Arguments:
            coeffs: Polynomial coefficients as columns in increasing order.
                    Can have arbitrarily many columns.
            t0: time to start the interpolation
            tf: time to end the interpolation
            n: number of interpolation samples (default 100)
        
        Returns:
            Trajectory following the interpolation. The states will be in the form:
            [pos1, pos2, ... posn, vel1, vel2, ... veln, accel1, ... acceln]
            Where n is the number of columns in coeffs
        """
        order = np.size(coeffs, 0) - 1
        t = np.array([np.linspace(t0, tf, n)]).T
        pos_t_vec = np.power(t, np.arange(order + 1))
        pos_vec = pos_t_vec @ coeffs
        vel_t_vec = np.concatenate((np.zeros((n,1)), np.multiply(pos_t_vec[:, 0:-1], np.repeat(np.array([np.arange(order) + 1]), n, 0))), 1)
        vel_vec = vel_t_vec @ coeffs
        acc_t_vec = np.concatenate((np.zeros((n,2)), np.multiply(vel_t_vec[:, 1:-1], np.repeat(np.array([np.arange(order - 1) + 2]), n, 0))), 1)
        acc_vec = acc_t_vec @ coeffs

        states = np.concatenate((pos_vec, vel_vec, acc_vec), 1).T
        return Trajectory(t, states)

def interpolate_states(t0, tf, state0, statef):
    coeffs = __cubic_interpolation(t0, tf, state0, statef)
    return from_coeffs(coeffs, t0, tf)


def __cubic_interpolation(t0, tf, state0, statef):
    """Perform cubic interpolation between state0 at t = t0 and statef at t = tf.
    Solves using the matrix equation:
    -                    -   -        -       -        -
    | 1    t0   t0^2  t0^3 | | c01  c02 |     | x01  x02 |
    | 0    1   2t0   3t0^2 | | c11  c12 |  =  | v01  v02 |
    | 1    tf   tf^2  tf^3 | | c21  c22 |     | xf1  xf2 |
    | 0    1   2tf   3tf^2 | | c31  c32 |     | vf1  vf2 |
    -                    -   -        -       -        -
    
    To find the cubic polynomials:
    x1(t) = c01 + c11t + c21t^2 + c31t^3
    x2(t) = c02 + c12t + c22t^2 + c32t^3
    where x1 is the first joint position and x2 is the second joint position, such that
    the arm is in state0 [x01, x02, v01, v02].T at t0 and statef [xf1, xf2, vf1, vf2].T at tf.

    Make sure to only use the interpolated cubic for t between t0 and tf.

    Arguments:
        t0 - start time of interpolation
        tf - end time of interpolation
        state0 - start state [theta1, theta2, omega1, omega2].T
        statef - end state [theta1, theta2, omega1, omega2].T
    
    Returns:
        coeffs - 4x2 matrix containing the interpolation coefficients for joint 1 in
                column 1 and joint 2 in column 2
    """
    pos_row = lambda t: np.array([[1, t, t*t, t*t*t]])
    vel_row = lambda t: np.array([[0, 1, 2*t, 3*t*t]])

    # right hand side matrix
    rhs = np.concatenate((state0.reshape((2,2)), statef.reshape(2,2)))
    # left hand side matrix
    lhs = np.concatenate((pos_row(t0), vel_row(t0), pos_row(tf), vel_row(tf)))

    coeffs = np.linalg.inv(lhs) @ rhs
    return coeffs