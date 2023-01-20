import numpy as np

from frccontrol import trajectory

def test_insert():

    t = trajectory.Trajectory(np.array([0]), np.zeros((4,1)))
    (t1, s1) = (5, np.array([5,5,5,5]))
    (t2, s2) = (-5, np.array([[-5,-5,-5,-5]]).T)

    t.insert(t1, s1)
    t.insert(t2, s2)

    assert np.allclose(t.times, np.array([-5, 0, 5]))
    assert np.allclose(t.states, np.array([[-5, -5, -5, -5], [0, 0, 0, 0], [5, 5, 5, 5]]).T)

    assert np.allclose(t.sample(2), np.array([[2, 2, 2, 2]]).T)

    t.insert(t1, s2 + 2)
    t.insert(2, s2)
    assert np.allclose(t.times, np.array([-5, 0, 2, 5]))
    assert np.allclose(t.states, np.array([[-5, -5, -5, -5], [0, 0, 0, 0], [-5, -5, -5, -5], [-3, -3, -3, -3]]).T)

if __name__ == "__main__":
    test_insert()