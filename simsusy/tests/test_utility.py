from nose.tools import ok_  # noqa
import simsusy.utility as u
import logging
import unittest
import random
import math
logger = logging.getLogger('test_info')


class TestUtility(unittest.TestCase):
    def test_trigonometric_random(self):
        for i in range(0, 4):
            theta = random.uniform(i * math.pi / 2, (i + 1) * math.pi / 2)
            sin = math.sin(theta)
            cos = math.cos(theta)
            tan = math.tan(theta)
            sin2 = math.sin(2 * theta)
            cos2 = math.cos(2 * theta)

            self.assertAlmostEqual(abs(sin), u.cos2sin(cos))
            self.assertAlmostEqual(abs(sin), u.tan2sin(tan))
            self.assertAlmostEqual(abs(cos), u.sin2cos(sin))
            self.assertAlmostEqual(abs(cos), u.tan2cos(tan))
            self.assertAlmostEqual(abs(tan), u.sin2tan(sin))
            self.assertAlmostEqual(abs(tan), u.cos2tan(cos))
            self.assertAlmostEqual(abs(sin2), u.tan2sintwo(tan))
            self.assertAlmostEqual(abs(cos2), u.tan2costwo(tan))

    def test_trigonometric_edge(self):
        for x in [-1, 0, 1]:
            self.assertAlmostEqual(1 - abs(x), u.cos2sin(x))
            self.assertAlmostEqual(1 - abs(x), u.sin2cos(x))
        self.assertAlmostEqual(0, u.tan2sin(0))
        self.assertAlmostEqual(1, u.tan2cos(0))
        self.assertAlmostEqual(0, u.sin2tan(0))
        self.assertAlmostEqual(0, u.cos2tan(1))
        self.assertAlmostEqual(0, u.cos2tan(-1))
        self.assertAlmostEqual(0, u.tan2sintwo(0))
        self.assertAlmostEqual(1, u.tan2costwo(0))
