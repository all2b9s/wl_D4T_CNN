"""Exact coefficient identity for the F(4,3) cross-correlation transform."""
from fractions import Fraction as F
import unittest


class WinogradIdentity(unittest.TestCase):
    def test_every_convolution_coefficient(self):
        b=[[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],
           [0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]]
        g=[[F(1,4),0,0],[F(-1,6)]*3,[F(-1,6),F(1,6),F(-1,6)],
           [F(1,24),F(1,12),F(1,6)],[F(1,24),F(-1,12),F(1,6)],[0,0,1]]
        a=[[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]]
        for out in range(4):
            for kernel in range(3):
                for pixel in range(6):
                    coefficient=sum(F(a[out][t])*g[t][kernel]*b[t][pixel] for t in range(6))
                    self.assertEqual(coefficient,int(pixel==out+kernel))
