"""Control system utility functions."""

import numpy as np


def ctrb(A, B):
    """Returns the controllability matrix for A and B

    Keyword arguments:
    A -- system matrix
    B -- input matrix
    """
    return np.hstack(
        [B] + [np.linalg.matrix_power(A, i) @ B for i in range(1, A.shape[0])]
    )


def obsv(A, C):
    """Returns the observability matrix for A and C

    Keyword arguments:
    A -- system matrix
    C -- output matrix
    """
    return np.vstack(
        [C] + [C @ np.linalg.matrix_power(A, i) for i in range(1, A.shape[0])]
    )
