from typing import Tuple
import numpy as np
import numpy.linalg as LA
import cmath


def sin2cos(sin: float)->float:
    """returns Abs[Cos[ArcSin[x]]]"""
    return ((sin + 1) * (-sin + 1)) ** 0.5


cos2sin = sin2cos


def tan2sin(tan: float)->float:
    """returns Abs[Sin[ArcTan[x]]]"""
    return abs(tan) * (tan**2 + 1) ** (-0.5)


def tan2cos(tan: float)->float:
    """returns Abs[Cos[ArcTan[x]]]"""
    return (tan**2 + 1) ** (-0.5)


def sin2tan(sin: float)->float:
    """returns Abs[Tan[ArcSin[x]]]"""
    return abs(sin) * ((-sin + 1) * (sin + 1)) ** (-0.5)


def cos2tan(cos: float)->float:
    """returns Abs[Tan[ArcCos[x]]]"""
    a = cos**(-1)
    return ((a - 1) * (a + 1)) ** 0.5


def tan2costwo(tan: float)->float:
    """returns Abs[Cos[2*ArcTan[x]]]"""
    return abs((tan + 1) * (tan - 1) / (tan**2 + 1))


def tan2sintwo(tan: float)->float:
    """returns Abs[Sin[2*ArcTan[x]]]"""
    return abs(2 * tan / (tan**2 + 1))


def chop_matrix(m: np.ndarray, threshold=1E-7):
    nx, ny = m.shape
    for ix in range(0, nx):
        for iy in range(0, ny):
            v = m[ix, iy]
            # chop element if smaller than "key entries"
            if ix != iy and abs(v) < min(abs(m[ix, min(ix, ny - 1)]), abs(m[min(iy, nx - 1), iy])) * threshold:
                m[ix, iy] = 0
            # chop imaginary part if small
            elif v.real != 0 and v.imag != 0:
                ratio = abs(v.imag / v.real)
                if ratio < threshold:
                    m[ix, iy] = v.real
                elif ratio > 1 / threshold:
                    m[ix, iy] = v.imag * 1j
    return m


def autonne_takagi(m: np.ndarray, try_real_mixing=True)->Tuple[np.ndarray, np.ndarray]:
    """Perform Autonne-Takagi decomposition.

    :param m: an input matrix M.
    :param try_real_mixing: if true, try to set N as real by allowing negative D entries;
                            if false, D is positive and N may be complex.
    :returns: a tuple (d, N), where d is a 1d matrix containing the diagonal elements of
              a diagonal matrix D, and N is an unitary matrix, which satisfy N^* M N^† = D. (SLHA eq.12)
              N is real if possible and try_real_mixing=True, and d is sorted as ascending in its absolute value.
    """
    eigenvalues, eigenvectors = LA.eigh(np.conjugate(m) @ m)
    n = np.conjugate(eigenvectors.T)
    if try_real_mixing:
        phases = np.diag([abs(x) / x for x in n.diagonal()])
    else:
        d = (np.conjugate(n) @ m @ np.conjugate(n.T)).diagonal()
        phases = np.diag([cmath.exp(-cmath.phase(x) / 2j) for x in d])
    n = phases @ n
    return chop_matrix((np.conjugate(n) @ m @ np.conjugate(n.T))).diagonal(), n


def singular_value_decomposition(m: np.ndarray)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform singular value decomposition.

    :param m: an input matrix M.
    :returns: a tuple (d, U, V), where d is a 1d matrix containing the diagonal elements of
              a non-negative diagonal matrix D, and U and V are unitary matrices, which satisfy
              U^* M V^† = D. (SLHA eq.14 or SLHA2 eq.48)
              U and V are real for a real input M, and d is ascending.
    """
    u0, s, vh0 = LA.svd(m)  # u0 @ s @ vh0 = m, i.e. u0^† @ m @ v0h^† = s
    d, u, v = s[::-1], (u0.T)[::-1], vh0[::-1]  # to have ascending order
    return d, chop_matrix(u), chop_matrix(v)


def mass_diagonalization(m: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """Perform mass diagonalization.

    :param m: an input matrix M, which is Hermitian.
    :returns: a tuple (d, R), where d is a 1d matrix containing the diagonal elements of
              a real diagonal matrix D, and R is an unitary matrix, which satisfy
              R M R^† = D. (SLHA eq.16 and SLHA2 below Eq.10)
              R is real for a real input M, and d is ascending.
    """
    eigenvalues, eigenvectors = LA.eigh(m)
    r = np.conjugate(eigenvectors.T)
    return eigenvalues, chop_matrix(r)
