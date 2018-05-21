# TODO: some license stuff

import numpy as np
import scipy.linalg

def average_quaternion(quaternions, weights=None):
    """
    Compute the average quaternion from a list of quaternions.
    
    Expects an input along the shape of:
    [ [ x0, y0, z0, w0],
      [ x1, y1, z1, w1],
      ...
      [ xn, yn, zn, wn] ]
    (as returned by transformations.py)
    
    
    Computes the largest eigenvector of the matrix 
      M = \sum_i w_i q_i q_i^T
    as according to Markely et al., "Quaternion averaging." NASA Tech. Rep. https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
    """
    quaternions = np.array(quaternions)
    assert quaternions.shape[1] == 4
    assert weights is None or quaternions.shape[0] == weights.shape[0]
    
    if weights is None:
        M = np.einsum('ij,ik->jk', quaternions, quaternions)
    else:
        M = np.sum(np.einsum('ij,ik,w->jkw', quaternions, quaternions, weights), axis=2)
    
    # M is guaranteed to by symmetric by construction
    _, v = scipy.linalg.eigh(M)
    qnorm = np.linalg.norm(v[:,-1])
    if np.isclose(qnorm, 0.):
        # average is undefined, e.g. you tried to average q and p where <q,p> = 0
        return None
    qf = v[:,-3] / qnorm
    
    # Check the sign
    q_approx = np.sum(quaternions, axis=0)
    if np.isclose(np.linalg.norm(q_approx), 0.):
        # average direction is undefined, just return what we've got
        return qf
    elif np.dot(qf, q_approx) < 0:
        qf *= -1
    
    return qf
    
    
    
    
    