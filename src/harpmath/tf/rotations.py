# TODO: some license stuff

import numpy as np
import scipy.linalg
import transformations

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
    # short circuit if only one quat
    if quaternions.ndim == 1:
        assert quaternions.shape[0] == 4
        return quaternions
    
    assert quaternions.shape[1] == 4
    assert weights is None or quaternions.shape[0] == weights.shape[0]
    
    if weights is None:
        M = np.einsum('ij,ik->jk', quaternions, quaternions)
    else:
        M = np.einsum('ij,ik,i->jk', quaternions, quaternions, weights)
    
    # M is guaranteed to by symmetric by construction
    _, v = scipy.linalg.eigh(M)
    qnorm = np.linalg.norm(v[:,-1])
    if np.isclose(qnorm, 0.):
        # average is undefined, e.g. you tried to average q and p where <q,p> = 0
        return None
    qf = v[:,-1] / qnorm
    
    # Check the sign
    q_approx = np.sum(quaternions, axis=0)
    if np.isclose(np.linalg.norm(q_approx), 0.):
        # average direction is undefined, just return what we've got
        return qf
    elif np.dot(qf, q_approx) < 0:
        qf *= -1
    
    return qf
    

def quaternion_covariance(quaternions, weights=None, avg=None):
    """
    Compute the covariance of the given quaternion set around each of the axes x, y, z
    """
    if avg is None:
        avg = average_quaternion(quaternions, weights)
    avg_inv = transformations.quaternion_inverse(avg)
    
    # Compute the Euler angle representation of each quaternion offset
    euler = []
    for q in quaternions:
        q_diff = transformations.quaternion_multiply(q, avg_inv)
        euler.append(transformations.euler_from_quaternion(q_diff))
    return np.std(euler, axis=0)

def quaternion_covariance_1d(quaternions, weights=None, avg=None):
    """
    Transform the 1-d standard deviation of the quaternions as
    \sigma = mean[ cos^{-1} (q_i \cdot \bar{q}) ]
    where
    \bar{q} = average_quaternion(quaternions)
    """
    if avg is None:
        avg = average_quaternion(quaternions, weights)
        
    ang = 2*np.arccos(np.dot(quaternions, avg))
    if weights is not None:
        ang = ang * weights
    return np.sqrt(np.sum(np.square(ang)) / (ang.size - 1))
    
    