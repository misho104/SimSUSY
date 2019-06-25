import math
from typing import (  # noqa: F401
    Any,
    List,
    Optional,
    Sequence,
    SupportsFloat,
    TypeVar,
    Union,
)

import numpy as np

from simsusy.abs_model import AbsModel
from simsusy.mssm.library import A, S
from simsusy.utility import sin2cos

T = TypeVar("T")
CFloat = Union[float, complex]


class MSSMInput(AbsModel):
    def __init__(self, *args):
        super().__init__(*args)
        self.slha1_compatible = not (
            any(self.modsel(i) for i in (4, 5, 6))
        )  # no RpV/CPV/FLV

    """
    High-level APIs are defined below, so that self.get would be used in special cases.

    NOTE: By defining a class to describe a parameter with its scale,
    one can extend functions below to return a parameter with scale.
    """

    @staticmethod
    def __value_or_unspecified_error(value: T, param_name) -> T:
        if isinstance(value, np.ndarray):
            if not np.any(np.equal(value, None)):
                return value
        else:
            if value is not None:
                return value
        raise ValueError(f"{param_name} is not specified.")

    @staticmethod
    def __complex_or_unspecified_error(
        value: Union[complex, SupportsFloat, str, bytes], param_name
    ) -> CFloat:
        value = MSSMInput.__value_or_unspecified_error(value, param_name)
        if isinstance(value, complex):
            return value
        try:
            float_value = float(value)
        except TypeError:
            raise ValueError(f"{param_name} is not a number.")
        return float_value

    @staticmethod
    def __complex_matrix_or_unspecified_error(
        value: np.ndarray, param_name
    ) -> np.ndarray:
        value = MSSMInput.__value_or_unspecified_error(value, param_name)
        for (i, j), v in np.ndenumerate(value):
            if isinstance(v, complex):
                pass
            try:
                value[i, j] = float(v)
            except TypeError:
                raise ValueError(f"{param_name}({i},{j}) is not a number.")
        return value

    def modsel(self, key: int) -> Union[int, float]:
        return self.get("MODSEL", key)

    def sminputs(self, key: int) -> float:
        return self.get("SMINPUTS", key)

    def mg(self, key: int) -> CFloat:
        """Return gaugino mass; key should be 1-3 (but no validation)."""
        value = self.get_complex("EXTPAR", key) or self.get_complex("MINPAR", 2)
        return self.__complex_or_unspecified_error(value, f"M_{key}")

    def ms2(self, species: S) -> np.ndarray:
        minpar_value = self.get_complex("MINPAR", 1)
        extpar_values = [
            self.get_complex("EXTPAR", species.extpar + gen) for gen in [1, 2, 3]
        ]
        value = np.diag(
            [
                extpar ** 2 if extpar is not None else minpar_value ** 2
                for extpar in extpar_values
            ]
        )
        for ix in (1, 2, 3):
            for iy in (1, 2, 3):
                v = self.get_complex(species.slha2_input, (ix, iy))
                if v is None:
                    pass
                elif ix <= iy:
                    value[ix, iy] = value[iy, ix] = v
                else:
                    pass  # error/warning should be raised in each calculator

        return self.__complex_matrix_or_unspecified_error(
            value, f"m_sfermion({species.name}) mass"
        )

    def a(self, species: A) -> np.ndarray:
        """Return A-term matrix, but only if T-matrix is not specified in the
        input; otherwise return None, and one should read T-matrix."""
        minpar_a33 = self.get_complex("MINPAR", 5)
        extpar_a33 = self.get_complex("EXTPAR", species.extpar)

        a33 = extpar_a33 if extpar_a33 is not None else minpar_a33

        for ix in (1, 2, 3):
            for iy in (1, 2, 3):
                if self.get_complex(species.slha2_input, (ix, iy)) is not None:
                    return None  # because T-matrix is specified.

        return self.__complex_matrix_or_unspecified_error(
            np.diag([0, 0, a33]), f"A({species.name})"
        )

    def t(self, species: A) -> np.ndarray:
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

        return self.__complex_matrix_or_unspecified_error(matrix, f"T({species.name})")

    def vckm(self) -> Optional[np.ndarray]:
        lam = self.get("VCKMIN", 1, default=0)
        a = self.get("VCKMIN", 2, default=0)
        rhobar = self.get("VCKMIN", 3, default=0)
        etabar = self.get("VCKMIN", 4)

        s12 = lam
        s23 = a * lam ** 2
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

    def upmns(self) -> np.ndarray:
        """return UPMNS matrix
        NOTE: SLHA2 convention uses theta-bars, while PDG2006 has only thetas.
              The difference should be ignored as it seems denoting MS-bar scheme.()
        """
        s12, s23, s13 = (math.sin(self.get("UPMNSIN", i, default=0)) for i in [1, 2, 3])
        c12, c23, c13 = (math.cos(self.get("UPMNSIN", i, default=0)) for i in [1, 2, 3])
        delta = self.get("UPMNSIN", 4)
        alpha1 = self.get("UPMNSIN", 5, default=0)
        alpha2 = self.get("UPMNSIN", 6, default=0)
        s13e = s13 * np.exp(1j * delta) if delta else s13

        matrix = np.array(
            [
                [c12 * c13, s12 * c13, s13e.conjugate()],
                [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
                [s12 * s23 - c12 * c23 * s13, -c12 * s23 - s12 * c23 * s13, c23 * c13],
            ]
        )
        if alpha1 or alpha2:
            phase = np.diag([np.exp(0.5j * alpha1), np.exp(0.5j * alpha2), 1])
            return matrix @ phase
        else:
            return matrix
