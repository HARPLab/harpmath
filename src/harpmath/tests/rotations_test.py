#!/usr/bin/env python

import numpy as np
from harpmath.tf import rotations
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

if __name__ == "__main__":
    unittest.main()
    