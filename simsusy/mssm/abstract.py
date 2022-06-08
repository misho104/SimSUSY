import json
import logging
import math
import pathlib
from typing import TYPE_CHECKING, List, Optional, Union  # noqa: F401

if TYPE_CHECKING:
    from simsusy.mssm.input import MSSMInput  # noqa: F401

logger = logging.getLogger(__name__)
CFloat = Union[complex, float]


class AbsSMParameters:
    DEFAULT_DATA = (
        pathlib.Path(__file__).parent.parent.resolve() / "default_values.json"
    )

    def default_value(self, key: str) -> float:
        default = self.default_values.get(key)
        if isinstance(default, dict):
            value = default.get("value")
            if isinstance(value, float):
                return value
        raise RuntimeError(
            f"Invalid parameter {key} in {self.DEFAULT_DATA}, which must be float."
        )

    def __init__(self, input):  # type: (MSSMInput) -> None
        with open(self.DEFAULT_DATA) as f:
            self.default_values = json.load(f)

        def get(key: int, default_key: str) -> float:
            value = input.sminputs(key)
            if isinstance(value, float):
                return value
            logger.info(
                f"Block SMINPUTS {key} is not specified; default value is used."
            )
            return self.default_value(default_key)

        self._alpha_em_inv = get(
            1, "alpha_EW_inverse@m_Z"
        )  # MS-bar, with 5 active flavors
        self._g_fermi = get(2, "G_F")
        self._alpha_s = get(3, "alpha_s@m_Z")  # MS-bar, with 5 active flavors
        self._mz = get(4, "m_Z")  # pole
        self._mb_mb = get(5, "m_b@m_b")  # MS-bar, at mb
        self._mt = get(6, "m_t")  # pole
        self._mtau = get(7, "m_tau")  # pole

        self._mnu1 = get(12, "m_nu1")  # pole
        self._mnu2 = get(14, "m_nu2")  # pole
        self._mnu3 = get(8, "m_nu3")  # pole
        self._me = get(11, "m_e")  # pole
        self._mmu = get(13, "m_mu")  # pole
        self._md_2gev = get(21, "m_d@2GeV")  # MS-bar, at 2GeV
        self._mu_2gev = get(22, "m_u@2GeV")  # MS-bar, at 2GeV
        self._ms_2gev = get(23, "m_s@2GeV")  # MS-bar, at 2GeV
        self._mc_mc = get(24, "m_c@m_c")  # MS-bar, at mc

    def sin_w_sq(self) -> float:
        return NotImplemented

    def cos_w_sq(self) -> float:
        return NotImplemented

    def mz(self) -> float:
        return NotImplemented

    def mw(self) -> float:
        return NotImplemented

    def gw(self) -> float:
        return NotImplemented

    def gy(self) -> float:
        return NotImplemented

    def gs(self) -> float:
        return NotImplemented

    def vev(self) -> float:
        return NotImplemented

    def mass(self, pid: int) -> float:  # pole mass
        if pid == 6:
            return self._mt
        elif pid == 11:
            return self._me
        elif pid == 13:
            return self._mmu
        elif pid == 15:
            return self._mtau
        elif pid == 12:
            return self._mnu1
        elif pid == 14:
            return self._mnu2
        elif pid == 16:
            return self._mnu3
        elif pid == 23:
            return self._mz
        else:
            return NotImplemented

    def mass_u(self) -> List[float]:
        return [self.mass(i) for i in (2, 4, 6)]

    def mass_d(self) -> List[float]:
        return [self.mass(i) for i in (1, 3, 5)]

    def mass_e(self) -> List[float]:
        return [self.mass(i) for i in (11, 13, 15)]

    def mass_n(self) -> List[float]:
        return [self.mass(i) for i in (12, 14, 16)]


class AbsEWSBParameters:
    sign_mu: Optional[complex]

    def __init__(self, model):  # type: (MSSMInput) -> None
        self.mh1_sq = model.get_complex("EXTPAR", 21)
        self.mh2_sq = model.get_complex("EXTPAR", 22)
        self.mu = model.get_complex("EXTPAR", 23)
        self.ma_sq = model.get_complex("EXTPAR", 24)
        self.ma0 = model.get_complex("EXTPAR", 26)
        self.mhc = model.get_complex("EXTPAR", 27)

        self.tan_beta = model.get_complex_assert(
            "EXTPAR", 25, default=model.get_complex("MINPAR", 3)
        )

        if self.mu is None:  # specified by MINPAR block
            sin_phi_mu = model.get_float("IMMINPAR", 4)
            if sin_phi_mu:  # CP-violated
                cos_phi_mu = model.get_float_assert("MINPAR", 4)
                if not (0.99 < (abs_sq := cos_phi_mu**2 + sin_phi_mu**2) < 1.01):
                    raise ValueError("Invalid mu-phase (MINPAR 4 and IMMINPAR 4)")
                self.sign_mu = complex(cos_phi_mu, sin_phi_mu) / math.sqrt(abs_sq)
            else:  # CP-conserved
                sign_mu = model.get_float_assert("MINPAR", 4)
                if not 0.9 < abs(sign_mu) < 1.1:
                    raise ValueError("Invalid EXTPAR 4; either 1 or -1.")
                self.sign_mu = -1 if sign_mu < 0 else 1
        else:
            self.sign_mu = None

        unspecified_param_count = self._count_unspecified_params()
        if unspecified_param_count > 4:
            m0 = model.get_float("EXTPAR", 1)
            m0sq = None if m0 is None else pow(m0, 2)
            self.mh1_sq = self.mh1_sq or m0sq
            self.mh2_sq = self.mh2_sq or m0sq
        self.validate()

    def _count_unspecified_params(self) -> int:
        return [
            self.mh1_sq,
            self.mh2_sq,
            self.mu,
            self.ma_sq,
            self.ma0,
            self.mhc,
        ].count(None)

    def validate(self) -> bool:
        unspecified_param_count = self._count_unspecified_params()
        if unspecified_param_count == 0:
            return True
        elif unspecified_param_count == 4:  # two must be specified
            if self.mh1_sq is not None or self.mh2_sq is not None:
                return True
            elif self.mu is not None:
                if (
                    self.ma_sq is not None
                    or self.ma0 is not None
                    or self.mhc is not None
                ):
                    return True
        raise ValueError("invalid specification of EWSB parameters")

    def is_set(self) -> bool:
        return self._count_unspecified_params() == 0

    def alpha(self) -> float:
        return NotImplemented
