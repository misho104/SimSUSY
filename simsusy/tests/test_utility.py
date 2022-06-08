"""A unit test for utility.py."""
import logging
import math
import random
import unittest
from typing import Optional, Union

import numpy as np
import numpy.typing

import simsusy.utility as u

ComplexMatrix = numpy.typing.NDArray[np.complex_]
RealMatrix = numpy.typing.NDArray[np.float_]
Matrix = numpy.typing.NDArray[Union[np.float_, np.complex_]]

logger = logging.getLogger("test_info")


class TestUtility(unittest.TestCase):
    """Simple tests for utility.py."""

    def __assert_sub(self, test, msg, a, b=None):
        # type: (Union[bool, np.bool_], str, Matrix, Optional[Matrix])->None
        if test:
            assert True
        else:
            logger.info("Tested objects are\n%s\n%s\n", a, b)
            self.fail(msg)

    def assert_matrix_almost_equal(self, a: Matrix, b: Matrix) -> None:
        self.__assert_sub(np.allclose(a, b), "Matrices not identical", a, b)

    def assert_matrix_equal(self, a: Matrix, b: Matrix) -> None:
        self.__assert_sub((a == b).all(), "Matrices not identical", a, b)

    def assert_all_real(self, a: Matrix) -> None:
        self.__assert_sub(np.isreal(a).all(), "Matrix not real", a)

    def assert_matrix_unitary(self, a: Matrix) -> None:
        self.__assert_sub(
            a.ndim == 2
            and a.shape[0] == a.shape[1]
            and np.allclose(a @ np.conjugate(a.T), np.identity(a.shape[0])),
            "Matrix not unitary",
            a,
        )

    def assert_sorted_in_absolute_value(self, a: Matrix) -> None:
        target = a.flatten()
        for i in range(0, len(target) - 1):
            self.__assert_sub(
                abs(target[i]) < abs(target[i + 1]),
                "Object not sorted in its absolute value",
                a,
            )

    def assert_sorted(self, a: Matrix) -> None:
        target = a.flatten()
        for i in range(0, len(target) - 1):
            self.__assert_sub(target[i] < target[i + 1], "Object not sorted", a)

    def assert_all_non_negative(self, a: Matrix) -> None:
        target = a.flatten()
        for i in range(0, len(target)):
            self.__assert_sub(target[i] >= 0, "Object not all-positive", a)

    def test_trigonometric_random(self):
        sin_regions = [i * math.pi / 4 for i in [-2, -1, 0, 1, 2]]
        cos_regions = [i * math.pi / 4 for i in [0, 1, 2, 3, 4]]
        for i in range(0, 4):
            # for sin2xxx and tan2xxx, where -pi/2 < theta < pi/2
            theta = random.uniform(sin_regions[i], sin_regions[i + 1])
            sin = math.sin(theta)
            cos = math.cos(theta)
            tan = math.tan(theta)
            sin2 = math.sin(2 * theta)
            cos2 = math.cos(2 * theta)
            tan2 = math.tan(2 * theta)
            self.assertAlmostEqual(cos, u.sin2cos(sin))
            self.assertAlmostEqual(tan, u.sin2tan(sin))
            self.assertAlmostEqual(sin, u.tan2sin(tan))
            self.assertAlmostEqual(cos, u.tan2cos(tan))
            self.assertAlmostEqual(sin2, u.tan2sintwo(tan))
            self.assertAlmostEqual(cos2, u.tan2costwo(tan))
            self.assertAlmostEqual(tan2, u.tan2tantwo(tan))

            # for cos2xxx, where 0 < theta < pi
            theta = random.uniform(cos_regions[i], cos_regions[i + 1])
            sin = math.sin(theta)
            cos = math.cos(theta)
            tan = math.tan(theta)
            self.assertAlmostEqual(sin, u.cos2sin(cos))
            self.assertAlmostEqual(tan, u.cos2tan(cos))

    def test_trigonometric_edge(self):
        # cos2xxx: (theta, s, c, t) = (0, 0, 1, 0), (pi/2, 1, 0, None), (pi, 0, -1, 0)
        for x in [-1, 1]:
            self.assertAlmostEqual(0, u.cos2sin(x))
            self.assertAlmostEqual(0, u.cos2tan(x))
        self.assertAlmostEqual(1, u.cos2sin(0))

        # sin2xxx and tan2xxx: (s, c, t) = (-1, 0, None), (0, 1, 0), (1, 0, None)
        for x in [-1, 1]:
            self.assertAlmostEqual(0, u.sin2cos(x))
        self.assertAlmostEqual(0, u.sin2tan(0))
        self.assertAlmostEqual(0, u.tan2sin(0))

        # tan2twoxxx: (+-pi/2, 0, -1, 0), (+-pi/4, +-1, 0, None), (0, 0, 1, 0),
        # i.e., (0, 0, 1, 0) is only relevant for -pi/2 < theta < pi/2.
        self.assertAlmostEqual(0, u.tan2tantwo(0))
        self.assertAlmostEqual(1, u.tan2costwo(0))
        self.assertAlmostEqual(0, u.tan2sintwo(0))

    def test_chop_matrix(self):
        r = 1e-12
        i = 1e-12j
        # remove off-diagonal small elements
        self.assert_matrix_equal(
            u.chop_matrix(np.array([[1, r, i], [i, 2, r], [r, i, -4j + i]])),
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, -4j + i]]),
        )
        # not remove if one of the relevant diagonal elements are small
        self.assert_matrix_equal(
            u.chop_matrix(np.array([[1, r, i], [i, 2, r], [i, r, r]])),
            np.array([[1, 0, i], [0, 2, r], [i, r, r]]),
        )
        # remove small real or imaginary part
        self.assert_matrix_equal(
            u.chop_matrix(
                np.array(
                    [
                        [3 + i, r + 3j, 3 + 3j, r + 2 * i],
                        [3 + r + i, r + 3j + i, -r + i, i],
                    ]
                )
            ),
            np.array([[3, 3j, 3 + 3j, r + 2 * i], [3 + r, 3j + i, -r + i, i]]),
        )
        # combination
        self.assert_matrix_equal(
            u.chop_matrix(
                np.array([[1 + i, r, i], [r + i, 2 + i, r], [-r, i, -4j + i + r]])
            ),
            np.array([[1, 0, 0], [0, 2, 0], [0, 0, -4j + i]]),
        )

    def test_autonne_takagi_real(self):
        # generate signed symmetric random matrix
        m: RealMatrix = np.random.rand(4, 4) - np.random.rand(4, 4)
        m = m + m.T

        # for real input, Autonne-Takagi decomposition will give
        #   - real eigenvalues with real mixing matrix (try_real_mixing = True)
        #   - positive eigenvalues with complex mixing matrix (try_real_mixing = False)

        d_vector, n = u.autonne_takagi(m, try_real_mixing=True)
        self.assert_all_real(n)
        self.assert_sorted_in_absolute_value(d_vector)
        self.assert_matrix_unitary(n)
        self.assert_matrix_almost_equal(
            np.conjugate(n) @ m @ np.conjugate(n.T), np.diag(d_vector)
        )

        d_vector, n = u.autonne_takagi(m, try_real_mixing=False)
        self.assert_all_non_negative(d_vector)
        self.assert_sorted_in_absolute_value(d_vector)
        self.assert_matrix_unitary(n)
        self.assert_matrix_almost_equal(
            np.conjugate(n) @ m @ np.conjugate(n.T), np.diag(d_vector)
        )

    def test_autonne_takagi_complex(self):
        m = (np.random.rand(4, 4) - np.random.rand(4, 4)) + (
            np.random.rand(4, 4) - np.random.rand(4, 4)
        ) * 1j
        m = m + m.T  # input must be symmetric

        # for complex input, Autonne-Takagi decomposition will give
        #   - real eigenvalues with complex mixing matrix (try_real_mixing = True)
        #   - positive eigenvalues with complex mixing matrix (try_real_mixing = False)

        d_vector, n = u.autonne_takagi(m, try_real_mixing=True)
        self.assert_sorted_in_absolute_value(d_vector)
        self.assert_matrix_unitary(n)
        self.assert_matrix_almost_equal(
            np.conjugate(n) @ m @ np.conjugate(n.T), np.diag(d_vector)
        )

        d_vector, n = u.autonne_takagi(m, try_real_mixing=False)
        self.assert_all_non_negative(d_vector)
        self.assert_sorted_in_absolute_value(d_vector)
        self.assert_matrix_unitary(n)
        self.assert_matrix_almost_equal(
            np.conjugate(n) @ m @ np.conjugate(n.T), np.diag(d_vector)
        )

    def test_singular_value_decomposition_real(self):
        m = np.random.rand(4, 4) - np.random.rand(4, 4)
        d_vector, uu, vv = u.singular_value_decomposition(m)
        self.assert_all_non_negative(d_vector)
        self.assert_sorted(d_vector)
        self.assert_matrix_unitary(uu)
        self.assert_matrix_unitary(vv)
        self.assert_all_real(uu)
        self.assert_all_real(vv)
        self.assert_matrix_almost_equal(
            np.conjugate(uu) @ m @ np.conjugate(vv.T), np.diag(d_vector)
        )

    def test_singular_value_decomposition_complex(self):
        m = (np.random.rand(4, 4) - np.random.rand(4, 4)) + (
            np.random.rand(4, 4) - np.random.rand(4, 4)
        ) * 1j
        d_vector, uu, vv = u.singular_value_decomposition(m)
        self.assert_all_non_negative(d_vector)
        self.assert_sorted(d_vector)
        self.assert_matrix_unitary(uu)
        self.assert_matrix_unitary(vv)
        self.assert_matrix_almost_equal(
            np.conjugate(uu) @ m @ np.conjugate(vv.T), np.diag(d_vector)
        )

    def test_mass_diagonalization_real(self):
        m: RealMatrix = np.random.rand(4, 4) - np.random.rand(4, 4)
        m = m + m.T  # symmetric
        d_vector, r = u.mass_diagonalization(m)
        self.assert_all_real(d_vector)
        self.assert_sorted(d_vector)
        self.assert_matrix_unitary(r)
        self.assert_all_real(r)
        self.assert_matrix_almost_equal(r @ m @ np.conjugate(r.T), np.diag(d_vector))

    def test_mass_diagonalization_complex(self):
        m = (np.random.rand(4, 4) - np.random.rand(4, 4)) + (
            np.random.rand(4, 4) - np.random.rand(4, 4)
        ) * 1j
        m = m + np.conjugate(m.T)  # hermitian
        d_vector, r = u.mass_diagonalization(m)
        self.assert_all_real(d_vector)
        self.assert_sorted(d_vector)
        self.assert_matrix_unitary(r)
        self.assert_matrix_almost_equal(r @ m @ np.conjugate(r.T), np.diag(d_vector))
