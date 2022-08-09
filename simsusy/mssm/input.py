import logging
import math
from typing import Any, Optional, TypeVar, Union, overload

import numpy as np
import numpy.typing

from simsusy.abs_model import RAISE_ERROR, AbsModel
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
            any(self.modsel(i, default=0) for i in (4, 5, 6))
        )  # no RpV/CPV/FLV

    """
    High-level APIs are defined below, so that self.get would be used in special cases.

    NOTE: By defining a class to describe a parameter with its scale,
    one can extend functions below to return a parameter with scale.
    """

    @overload
    def modsel(self, key: int) -> int:
        ...

    @overload
    def modsel(self, key: int, default: int) -> int:
        ...

    @overload
    def modsel(self, key: int, default: None) -> Optional[int]:
        ...

    def modsel(self, key: int, default: Any = RAISE_ERROR) -> Optional[int]:
        """Return MODSEL value."""
        return self.get_int("MODSEL", key, default=default)  # type: ignore

    @overload
    def sminputs(self, key: int) -> float:
        ...

    @overload
    def sminputs(self, key: int, default: Optional[float]) -> Optional[float]:
        ...

    def sminputs(self, key: int, default: Any = RAISE_ERROR) -> Optional[float]:
        """Return SMINPUTS value."""
        return self.get_float("MODSEL", key, default=default)  # type: ignore

    @overload
    def minpar(self, key: int) -> complex:
        ...

    @overload
    def minpar(self, key: int, default: complex) -> complex:
        ...

    @overload
    def minpar(self, key: int, default: None) -> Optional[complex]:
        ...

    def minpar(self, key: int, default: Any = RAISE_ERROR) -> Optional[complex]:
        """Return SMINPUTS value."""
        return self.get_complex("MINPAR", key, default=default)  # type: ignore

    @overload
    def extpar(self, key: int) -> complex:
        ...

    @overload
    def extpar(self, key: int, default: complex) -> complex:
        ...

    @overload
    def extpar(self, key: int, default: None) -> Optional[complex]:
        ...

    def extpar(self, key: int, default: Any = RAISE_ERROR) -> Optional[complex]:
        """Return SMINPUTS value."""
        return self.get_complex("EXTPAR", key, default=default)  # type: ignore

    def mg(self, key: int) -> complex:
        """Return gaugino mass; key should be 1-3 (but no validation)."""
        value = self.extpar(key, default=self.minpar(2, default=None))
        if value is not None:
            return value
        raise ValueError(f"Gaugino mass {key} is unset.")

    def ms2(self, species: S) -> Matrix:
        """Return scalar mass-squared matrix.

        MSx2IN-value, EXTPAR-value, and MINPAR-value are used in this order.
        """
        minpar = self.minpar(1, default=None)
        default = [
            self.extpar(species.extpar + gen, default=minpar) for gen in [1, 2, 3]
        ]
        result = np.zeros((3, 3))
        for (ix, iy), _ in np.ndenumerate(result):
            v = self.get_complex(species.slha2_input, (ix + 1, iy + 1), default=None)
            if v is None:
                if ix == iy:  # diagonal element must be specified
                    if (d := default[ix]) is None:
                        raise ValueError(
                            f"{species.slha2_input}({ix+1}{iy+1}) is not specified."
                        )
                    else:
                        result[ix, iy] = d * d
            elif ix <= iy:
                result[ix, iy] = result[iy, ix] = v
            else:
                logger.warning("%s is ignored.", (species.slha2_input, ix, iy))
        return result

    def a(self, species: A) -> Optional[Matrix]:
        """Return A-term matrix.

        If T-matrix is specified, None is returned. Note that only (3, 3)-element can
        be specified.
        """
        blocks = self.slha.blocks
        if species.slha2_input in blocks or "IM" + species.slha2_input in blocks:
            return None
        a33 = self.extpar(species.extpar, default=self.minpar(5, default=0))
        return np.diag([0, 0, a33])

    def t(self, species: A) -> Optional[Matrix]:
        """Return T-term matrix if T-matrix is specified.

        Corresponding EXTPAR entry is ignored and thus (3,3) element must always be
        specified.
        """
        matrix = self.get_complex_matrix(species.slha2_input, default=None)
        if matrix is None:
            return None  # and A-matrix should be specified instead.
        if self.get_complex(species.slha2_input, (3, 3), default=None) is None:
            ValueError(f"Block {species.slha2_input} always needs (3,3) element.")
        return matrix

    def vckm(self) -> Matrix:
        lam = self.get_float("VCKMIN", 1, default=0)
        a = self.get_float("VCKMIN", 2, default=0)
        rho_bar = self.get_float("VCKMIN", 3, default=0)
        eta_bar = self.get_float("VCKMIN", 4, default=None)

        s12 = lam
        s23 = a * lam * lam
        c12 = sin2cos(s12)
        c23 = sin2cos(s23)
        r = rho_bar + eta_bar * 1j if eta_bar is not None else rho_bar
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
        """Return UPMNS matrix.

        NOTE: SLHA2 convention uses theta-bars, while PDG2006 has only thetas.
              The difference should be ignored as it seems denoting MS-bar scheme.()
        """
        angles = [self.get_float("UPMNSIN", i, default=0) for i in [1, 2, 3]]
        s12, s23, s13 = (math.sin(v) for v in angles)
        c12, c23, c13 = (math.cos(v) for v in angles)
        delta = self.get_float("UPMNSIN", 4, default=None)
        alpha1 = self.get_float("UPMNSIN", 5, default=0)
        alpha2 = self.get_float("UPMNSIN", 6, default=0)
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
