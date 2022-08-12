"""Abstract base class for MSSM calculation."""

import json
import logging
import math
import pathlib
from typing import List, Optional

import simsusy.mssm.library
from simsusy.mssm.input import MSSMInput

logger = logging.getLogger(__name__)


class AbsSMParameters:
    """
    The abstract version of the Standard Model parameters.

    As an abstract class, this class only provides a basic I/O interface from
    the SMINPUTS block and from the default_value file. Values except pole
    masses are calculated in various loop levels, so left as unimplemented.

    Note that mz(), mw(), vev(), etc. may be scale dependent, while mass(pid)
    should return some scale-independent value (e.g., the pole mass).
    """

    DEFAULT_DATA = (
        pathlib.Path(__file__).parent.parent.resolve() / "default_values.json"
    )

    def default_value(self, key: str) -> float:
        """Return a value read from DEFAULT_DATA."""
        default = self.default_values.get(key)
        if isinstance(default, dict):
            if isinstance(value := default.get("value"), float):
                return value
        raise RuntimeError(
            f"Invalid parameter {key} in {self.DEFAULT_DATA}, which must be float."
        )

    def __init__(self, input: MSSMInput) -> None:  # noqa: A002
        with open(self.DEFAULT_DATA) as f:
            self.default_values = json.load(f)

        def get(key: int, default_key: str) -> float:
            value = input.sminputs(key, default=None)
            if isinstance(value, float):
                return value
            logger.info("Block SMINPUTS %d missing; default value is used.", key)
            return self.default_value(default_key)

        self._alpha_em_inv = get(1, "alpha_EW_inverse@m_Z")  # MS-bar (5 active flavors)
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

    # pole-mass handlers

    def mass(self, pid: int) -> float:
        """Return the pole mass of the particle with the given PDG ID."""
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
        """Return the up-type quark masses."""
        return [self.mass(i) for i in (2, 4, 6)]

    def mass_d(self) -> List[float]:
        """Return the down-type quark masses."""
        return [self.mass(i) for i in (1, 3, 5)]

    def mass_e(self) -> List[float]:
        """Return the charged lepton masses."""
        return [self.mass(i) for i in (11, 13, 15)]

    def mass_n(self) -> List[float]:
        """Return the neutrino masses."""
        return [self.mass(i) for i in (12, 14, 16)]

    # Weinberg angles, dependent on `_sin_sq_cos_sq`.

    def sin_w_sq(self) -> float:
        """Return sin^2(theta_w)."""
        r = self._sin_sq_cos_sq()
        return 2 * r / (1 + math.sqrt(1 - 4 * r))

    def cos_w_sq(self) -> float:
        """Return cos^2(theta_w)."""
        r = self._sin_sq_cos_sq()
        return (1 + math.sqrt(1 - 4 * r)) / 2

    # abstract functions

    def _sin_sq_cos_sq(self) -> float:
        """Return sin^2(theta_w)*cos^2(theta_w)."""
        return NotImplemented

    def mz(self) -> float:
        """Return the Z-boson mass, which may not be the pole mass."""
        return self._mz

    def mw(self) -> float:
        """Return the W-boson mass, which may not be the pole mass."""
        return NotImplemented

    def gw(self) -> float:
        """Return the SU(2)_weak coupling."""
        return NotImplemented

    def gy(self) -> float:
        """Return the U(1)_Y coupling."""
        return NotImplemented

    def gs(self) -> float:
        """Return the strong coupling."""
        return NotImplemented

    def vev(self) -> float:
        """Return the vacuum expectation value of Higgs."""
        return NotImplemented


class AbsEWSBParameters:
    """
    The abstract version of the MSSM EWSB parameters.

    As an abstract class, this class only provides I/O from the input file.
    The SLHA convention allows following inputs:

      - tan(beta) and two Higgs soft masses,
      - tan(beta), mu, and tree-level pseudo-scalar mass,
      - tan(beta), mu, and pseudo-scalar pole mass,
      - tan(beta), mu, and charged-Higgs pole mass.

    The input file should have a proper combination of those seven parameters
    and the calculator should be able to handle any of the combinations. Also,
    if mu is not specified, `sign_mu` is allowed as a complex input, which
    should be properly handled.
    """

    sign_mu: complex  # sign (or argument) of mu parameters
    _tb_ewsb: Optional[float]  # tan(beta) at the EWSB scale
    _tb_input: Optional[float]  # tan(beta) at the input scale
    mh1_sq: Optional[complex]  # down-type Higgs soft mass at the input scale
    mh2_sq: Optional[complex]  # up-type Higgs soft mass at the input scale
    mu: Optional[complex]  # mu-parameter at the input scale
    ma_sq: Optional[complex]  # tree-level mass of A at the input scale
    ma0: Optional[float]  # pole mass of A
    mhc: Optional[float]  # pole mass of H+

    # abstracts

    @property
    def tan_beta(self) -> float:
        """Return tan_beta at the input scale."""
        return NotImplemented  # should be implemented in the derived class

    def alpha(self) -> float:
        """Return the angle between the Higgses."""
        return NotImplemented

    # implementation

    def __init__(self, model: MSSMInput) -> None:
        """Fill the values from SLHA input."""
        self._tb_ewsb = model.get_float("MINPAR", 3, default=None)
        self._tb_input = model.get_float("EXTPAR", 25, default=None)
        self.mh1_sq = model.get_complex("EXTPAR", 21, default=None)
        self.mh2_sq = model.get_complex("EXTPAR", 22, default=None)
        self.mu = model.get_complex("EXTPAR", 23, default=None)
        self.ma_sq = model.get_complex("EXTPAR", 24, default=None)
        self.ma0 = model.get_float("EXTPAR", 26, default=None)
        self.mhc = model.get_float("EXTPAR", 27, default=None)

        # determine the sign of mu parameter
        if self.mu is not None:  # direct specification of mu parameter
            self.sign_mu = 1
        else:  # |mu| is determined by EWSB condition and sign_mu is required.
            sin_phi_mu = model.get_float("IMMINPAR", 4, default=None)
            if sin_phi_mu:  # CP-violated
                cos_phi_mu = model.get_float("MINPAR", 4)
                if not (0.99 < (abs_sq := cos_phi_mu**2 + sin_phi_mu**2) < 1.01):
                    raise ValueError("Invalid mu-phase (MINPAR 4 and IMMINPAR 4)")
                self.sign_mu = complex(cos_phi_mu, sin_phi_mu) / math.sqrt(abs_sq)
            else:  # CP-conserved
                sign_mu = model.get_float("MINPAR", 4)
                if not 0.9 < abs(sign_mu) < 1.1:
                    raise ValueError("Invalid EXTPAR 4; either 1 or -1.")
                self.sign_mu = -1 if sign_mu < 0 else 1

        # mh1_sq and mh2_sq may be specified in the MINPAR block.
        if self._count_unspecified_params() > 4:
            if (m0 := model.get_float("EXTPAR", 1, default=None)) is not None:
                if self.mh1_sq is None:
                    self.mh1_sq = m0 * m0
                if self.mh2_sq is None:
                    self.mh2_sq = m0 * m0

        # check if tan(beta) is properly set.
        if self.tan_beta == NotImplemented:
            raise RuntimeError("Missing implementation of tan_beta().")
        elif self.tan_beta is None:
            logger.error("invalid specification of tan_beta")

        # check if the parameters are set in one of the proper combinations.
        if self._count_unspecified_params() == 4:  # it must be four.
            if self.mh1_sq is not None and self.mh2_sq is not None:
                return  # pass
            elif self.mu is not None:
                if (
                    self.ma_sq is not None
                    or self.ma0 is not None
                    or self.mhc is not None
                ):
                    return  # pass
        logger.error("invalid specification of EWSB parameters")

    def _count_unspecified_params(self) -> int:
        return [
            self.mh1_sq,
            self.mh2_sq,
            self.mu,
            self.ma_sq,
            self.ma0,
            self.mhc,
        ].count(None)

    def is_set(self) -> bool:
        """Check if the EWSB parameter is calculated."""
        return (
            isinstance(self.tan_beta, (int, float))
            and self._count_unspecified_params() == 0
        )

    def yukawa(self, species: simsusy.mssm.library.A) -> List[float]:
        """Return the diagonal of Yukawa."""
        if species == simsusy.mssm.library.A.U:
            return self.yu()
        elif species == simsusy.mssm.library.A.D:
            return self.yd()
        elif species == simsusy.mssm.library.A.E:
            return self.ye()
        else:
            raise RuntimeError("invalid call of ewsb.yukawa")

    # virtual functions
    def yu(self) -> List[float]:
        """Return the diagonal of up-type Yukawa after super-CKM rotation."""
        raise NotImplementedError

    def yd(self) -> List[float]:
        """Return the diagonal of down-type Yukawa."""
        raise NotImplementedError

    def ye(self) -> List[float]:
        """Return the diagonal of charged-lepton Yukawa."""
        raise NotImplementedError

    def mass(self, pid: int) -> float:
        """Return the pole mass of the particle with the given PDG ID."""
        raise NotImplementedError
