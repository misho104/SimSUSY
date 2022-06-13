import logging
import math
from typing import Any, Optional, TypeVar, Union

import numpy as np
import numpy.typing

from simsusy.abs_model import AbsModel
from simsusy.mssm.library import A, S
from simsusy.utility import sin2cos

T = TypeVar("T")
ComplexMatrix = numpy.typing.NDArray[np.complex_]
RealMatrix = numpy.typing.NDArray[np.float_]
Matrix = numpy.typing.NDArray[Union[np.float_, np.complex_]]

logger = logging.getLogger(__name__)


class MSSMInput(AbsModel):
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.slha1_compatible = not (
            any(self.modsel(i) for i in (4, 5, 6))
        )  # no RpV/CPV/FLV

    """
    High-level APIs are defined below, so that self.get would be used in special cases.

    NOTE: By defining a class to describe a parameter with its scale,
    one can extend functions below to return a parameter with scale.
    """

    def modsel(self, key: int) -> Optional[int]:
        return self.get_int("MODSEL", key)

    def sminputs(self, key: int) -> Optional[float]:
        return self.get_float("SMINPUTS", key)

    def mg(self, key: int) -> complex:
        """Return gaugino mass; key should be 1-3 (but no validation)."""
        value = self.get_complex("EXTPAR", key) or self.get_complex("MINPAR", 2)
        if value is not None:
            return value
        raise ValueError(f"Gaugino mass {key} is unset.")

    def ms2(self, species: S) -> Matrix:
        minpar = self.get_complex("MINPAR", 1)
        extpar = [
            self.get_complex_assert("EXTPAR", species.extpar + gen, default=minpar)
            for gen in [1, 2, 3]
        ]
        result = np.zeros((3, 3))
        for ix in (0, 1, 2):
            for iy in (0, 1, 2):
                v = self.get_complex(species.slha2_input, (ix + 1, iy + 1))
                if v is None:
                    if ix == iy:  # diagonal element must be specified
                        if extpar[ix] is None:
                            raise ValueError(
                                f"{species.slha2_input}({ix+1}{iy+1}) is not specified."
                            )
                        else:
                            result[ix, iy] = extpar[ix] * extpar[ix]
                elif ix <= iy:
                    result[ix, iy] = result[iy, ix] = v
                else:
                    logger.warning("%s is ignored.", (species.slha2_input, ix, iy))
        return result

    def a(self, species: A) -> Optional[Matrix]:
        """Return A-term matrix, but only if T-matrix is not specified in the
        input; otherwise return None, and one should read T-matrix."""
        for ix in (1, 2, 3):
            for iy in (1, 2, 3):
                if self.get_complex(species.slha2_input, (ix, iy)) is not None:
                    return None  # because T-matrix is specified.
        a33 = self.get_complex_assert(
            "EXTPAR", species.extpar, default=self.get_complex("MINPAR", 5)
        )
        return np.diag([0, 0, a33])

    def t(self, species: A) -> Optional[Matrix]:
        """Return T-term matrix if T-matrix is specified; corresponding EXTPAR
        entry is ignored and thus (3,3) element must be always specified."""
        specified = False
        matrix = np.diag([0, 0, np.nan])
        for ix in (1, 2, 3):
            for iy in (1, 2, 3):
                v = self.get_complex(species.slha2_input, (ix, iy))
                if v is not None:
                    matrix[ix - 1, iy - 1] = v
                    specified = True
        if not specified:
            return None  # and A-matrix should be specified instead.
        if math.isnan(matrix[2, 2]):
            ValueError(f"Block {species.slha2_input} needs (3,3) element.")
        return matrix

    def vckm(self) -> Matrix:
        lam = self.get_float_assert("VCKMIN", 1, default=0)
        a = self.get_float_assert("VCKMIN", 2, default=0)
        rhobar = self.get_float_assert("VCKMIN", 3, default=0)
        etabar = self.get_float("VCKMIN", 4)

        s12 = lam
        s23 = a * lam * lam
        c12 = sin2cos(s12)
        c23 = sin2cos(s23)
        r = rhobar + etabar * 1j if etabar else rhobar
        s13e = s12 * s23 * c23 * r / c12 / (1 - s23 * s23 * r)
        c13 = sin2cos(s13e.real)
        return np.array(
            [
                [c12 * c13, s12 * c13, s13e.conjugate()],
                [
                    -s12 * c23 - c12 * s23 * s13e,
                    c12 * c23 - s12 * s23 * s13e,
                    s23 * c13,
                ],
                [
                    s12 * s23 - c12 * c23 * s13e,
                    -c12 * s23 - s12 * c23 * s13e,
                    c23 * c13,
                ],
            ]
        )

    def upmns(self) -> ComplexMatrix:
        """return UPMNS matrix
        NOTE: SLHA2 convention uses theta-bars, while PDG2006 has only thetas.
              The difference should be ignored as it seems denoting MS-bar scheme.()
        """
        angles = [self.get_float_assert("UPMNSIN", i, default=0) for i in [1, 2, 3]]
        s12, s23, s13 = (math.sin(v) for v in angles)
        c12, c23, c13 = (math.cos(v) for v in angles)
        delta = self.get_float("UPMNSIN", 4)
        alpha1 = self.get_float_assert("UPMNSIN", 5, default=0)
        alpha2 = self.get_float_assert("UPMNSIN", 6, default=0)
        s13e: complex = s13 * np.exp(1j * delta) if delta else s13
        matrix = np.array(
            [
                [c12 * c13, s12 * c13, s13e.conjugate()],
                [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
                [s12 * s23 - c12 * c23 * s13, -c12 * s23 - s12 * c23 * s13, c23 * c13],
            ],
        )
        if alpha1 or alpha2:
            phase = np.diag([np.exp(0.5j * alpha1), np.exp(0.5j * alpha2), 1])
            return matrix @ phase  # type: ignore
        else:
            return matrix
