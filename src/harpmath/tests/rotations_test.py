#!/usr/bin/env python

import numpy as np
from harpmath.tf import rotations, transformations
import unittest

class RotationTest(unittest.TestCase):
    def test1(self):
        th = np.random.rand(3)
        ax = np.array([ 0, 1, 0])
        q = [ np.append( np.sin(t/2)*ax, np.cos(t/2)) for t in th ]
        thn = np.mean(th)
        qn = np.append( np.sin(thn/2)*ax, np.cos(thn/2) )
        qn_c = rotations.average_quaternion(q)
        print(qn)
        print(qn_c)
        self.assertTrue(np.allclose(qn_c, qn, rtol=5e-2))

    def test2(self):
        th = np.random.rand(10)
        ax = np.array([0,1,0])
        q = [ np.append( np.sin(t/2)*ax, np.cos(t/2)) for t in th ]
        
        std = rotations.quaternion_covariance(q)
        print(std)
        print(np.std(th))
        self.assertTrue(np.allclose(std[1], np.std(th)))
        
    def test3(self):
        thx = np.random.rand(10)
        axx = np.array([1,0,0])
        thy = np.random.rand(10)
        axy = np.array([0,1,0])
        qx = [ np.append( np.sin(t/2)*axx, np.cos(t/2)) for t in thx ]
        qy = [ np.append( np.sin(t/2)*axy, np.cos(t/2)) for t in thy ]
        q = [ transformations.quaternion_multiply(q1, q2) for q1, q2 in zip(qx, qy) ]
        std = rotations.quaternion_covariance(q)
        print(std)
        print([np.std(thx), np.std(thy), 0])
        # NB: not the same

if __name__ == "__main__":
    unittest.main()
    