"""
Tree-level MSSM calculator.

Convention of the output SLHA file can be specified in `SIMSUSY` block:

  - 101: whether to include (0, default) or hide (1) warnings in SPINFO.

  - 111: (SLHA2 only) whether to allow negative MASS value of neutralinos.
       * 0 (default) to take neutralino masses to be positive and allow complex
         NMIX matrix (with IMNMIX). This is the original SLHA2 convention.
       * 1 to mimic SLHA1 convention, where, as long as the neutralino sector
         has no CP-violation, NMIX is forced to be real and MASS block may have
         negative entries.

  - 131: (SLHA2 only) order of the down-type squark sector (DSQMIX)
  - 132: (SLHA2 only) order of the up-type squark (USQMIX)
  - 133: (SLHA2 only) order of the charged lepton sector (SELMIX)
  - 134: (SLHA2 only) order of the sneutrino sector (SNUMIX)
       * 0 (default) to order the sfermion ascending in their mass, which is
         the original SLHA2 convention.
       * 1xyz to sort them in the flavor order (e.g., u-u-c-c-t-t), where x, y,
         and z (0 or 1) specifies the order within each generation; 0 to set in
         the mass order, while 1 to set in the left-right order. For example,
         slepton-setting of 133 = 1110 mimics the SLHA1-like convention,
         where 1000011 (1000013) is left-handed selectron (smuon), 2000011
         (2000013) is the right-handed selectron (smuon), and 1000015 (2000015)
         is the lighter (heavier) stau.
         For sneutrinos, the last three digits have no effect.

  - 141: (SLHA1 only) whether to output lighter-generation Yukawa and A-terms
       * 0 (default) to remove first and second generation Yukawa and A-terms
         as in original SLHA1.
       * 1 to show all three generation. Off-diagonal terms are not included.
         This is requested by SDECAY.

  - 901: (SLHA2 only) options of soft-mass matrices (MSQ2 etc.)
  - 902: (SLHA2 only) options of sfermion mixing matrices (USQMIX etc.)
  - 903: (SLHA2 only) options of Yukawa and T-term matrices (YU, TU, etc.)
  - 904: (SLHA2 only) options of VCKM and UPMNS matrices.
       * 0 (default) to display all elements. Whole IM-matrix are displayed,
         but only if any element is complex.
       * 1 to remove entries with zero real part and zero imaginary part.
       * n (>=2) to remove entries whose absolute values (as a complex) are
         smaller than 10^(-n) .
"""
import itertools
import logging
import re
from math import atan, pi, sqrt
from typing import List, MutableMapping, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing
import yaslha

import simsusy.simsusy
from simsusy.abs_calculator import AbsCalculator
from simsusy.mssm.abstract import AbsEWSBParameters, AbsSMParameters
from simsusy.mssm.input import MSSMInput as Input
from simsusy.mssm.library import CPV, FLV, SMIX, A, S
from simsusy.mssm.model import MSSMModel as Output
from simsusy.utility import (
    autonne_takagi,
    is_diagonal_matrix,
    is_real_matrix,
    mass_diagonalization,
    sin2cos,
    singular_value_decomposition,
    tan2cos,
    tan2costwo,
    tan2sin,
    tan2sintwo,
    tan2tantwo,
)

ComplexMatrix = numpy.typing.NDArray[np.complex_]
RealMatrix = numpy.typing.NDArray[np.float_]
Matrix = numpy.typing.NDArray[Union[np.float_, np.complex_]]
logger = logging.getLogger(__name__)
T = TypeVar("T", RealMatrix, ComplexMatrix)

neutralino_pids = (1000022, 1000023, 1000025, 1000035)
chargino_pids = (1000024, 1000037)


def pow2(v: float) -> float:
    """Return the square of a number."""
    return v * v


class SMParameters(AbsSMParameters):
    """
    The Standard Model parameters.

    Since this is for tree_calculator, values are calculated at the tree level.
    The only exception is the W-boson mass; the method `self.mw()` returns the
    tree-level mass (less than 80 GeV), while `self.mass(24)` returns the pole
    mass (80.4 GeV).
    """

    def __init__(self, input: Input) -> None:  # noqa: A003
        super().__init__(input)

    """
    Mathematica code to check the calculation:
    {aEMi == 4 Pi/e^2,
     mZ   == Sqrt[(gY^2 + gW^2) (v1^2 + v2^2)/2],
     mW^2 == gW^2 (v1^2 + v2^2)/2,
     GF   == gW^2/(4*Sqrt[2]*mW^2),
     gW   == e/Sin[w],
     gY   == e/Cos[w]
    }
    """

    def _sin_sq_cos_sq(self) -> float:
        """Return sin^2(theta_w)*cos^2(theta_w)."""
        return pi / (sqrt(2) * self._alpha_em_inv * self._g_fermi * self._mz * self._mz)

    def mw(self) -> float:
        """Return the tree-level W-boson mass."""
        return self.mz() * sqrt(self.cos_w_sq())

    def gw(self) -> float:
        """Return the tree-level SU(2)_weak coupling."""
        return sqrt(4 * pi / self._alpha_em_inv / self.sin_w_sq())

    def gy(self) -> float:
        """Return the tree-level U(1)_Y coupling."""
        return sqrt(4 * pi / self._alpha_em_inv / self.cos_w_sq())

    def gs(self) -> float:
        """Return the tree-level strong coupling."""
        return sqrt(4 * pi * self._alpha_s)

    def vev(self) -> float:
        """Return the tree-level vacuum expectation value of Higgs."""
        # SLHA Last para of Sec.2.2
        return 2 * self.mz() / sqrt(pow2(self.gy()) + pow2(self.gw()))

    def mass(self, pid: int) -> float:
        """Return the pole mass of the specified SM particle."""
        if (value := super().mass(pid)) is not NotImplemented:
            return value
        elif pid == 1:
            return self._md_2gev
        elif pid == 2:
            return self._mu_2gev
        elif pid == 3:
            return self._ms_2gev
        elif pid == 4:
            return self._mc_mc
        elif pid == 5:
            return self._mb_mb
        elif pid == 24:
            return self.default_value("m_W")
        else:
            return NotImplemented


class EWSBParameters(AbsEWSBParameters):
    """The EWSB parameters calculated with the tree-level relations."""

    @property
    def tan_beta(self) -> float:
        """
        Return tan_beta.

        As for tree-level calculations, we just ignore the energy scale, i.e.,
        ignore EXTPAR(25) if MINPAR(3) is specified.
        """
        if self._tb_ewsb is not None:
            return self._tb_ewsb
        if self._tb_input is not None:
            return self._tb_input
        raise ValueError("tan(beta) is not specified.")

    def alpha(self) -> float:
        """Return the angle between the Higgses."""
        tan_twobeta, ma, mz = tan2tantwo(self.tan_beta), self.ma0, self.sm.mz()
        assert ma is not None
        return 0.5 * atan((pow2(ma) + pow2(mz)) / (ma + mz) / (ma - mz) * tan_twobeta)

    def __init__(self, input_obj: Input, sm: AbsSMParameters) -> None:
        super().__init__(input_obj)
        self.sm = sm
        self.calculate()
        assert self.is_set()

    def calculate(self) -> None:
        """
        Calculate the EWSB parameters at the tree-level.

        This function may raise ValueError, which should be caught by the
        caller to store the message in the SPINFO block. AssertionError may be
        raised, but it should be a bug of calculator, not due to invalid input.
        """
        for v in (
            self.mh1_sq,
            self.mh2_sq,
            self.mu,
            self.ma_sq,
            self.ma0,
            self.mhc,
            self.tan_beta,
            self.sign_mu,
        ):
            if v is not None and v.imag:
                raise ValueError("This calculator does not support CPV.")

        # see doc/convention.pdf for equations and discussion.
        cos_2beta = tan2costwo(self.tan_beta)
        sin_2beta = tan2sintwo(self.tan_beta)
        mz_sq = pow2(self.sm.mz())
        mw_sq = pow2(self.sm.mw())  # use tree level value (not 80.4)

        # first calculate mu if absent.
        if self.mu is None:
            assert isinstance(self.mh1_sq, (int, float))
            assert isinstance(self.mh2_sq, (int, float))
            # Martin (8.1.11)
            mu_sq = 0.5 * (
                -self.mh2_sq
                - self.mh1_sq
                - mz_sq
                + abs((self.mh1_sq - self.mh2_sq) / cos_2beta)
            )
            if mu_sq < 0:
                raise ValueError("Failed to get EWSB: mu^2 < 0")
            self.mu = self.sign_mu * sqrt(mu_sq)

            # ma_sq is now ready to set; GH (3.22) or Martin (8.1.10)
            self.ma_sq = self.mh1_sq + self.mh2_sq + 2.0 * mu_sq
            if self.ma_sq < 0:
                raise ValueError("Failed to get EWSB: mA^2 < 0")
        else:
            # set ma_sq
            if self.ma0 is not None:
                self.ma_sq = pow2(self.ma0)  # at the tree level
            elif self.mhc is not None:
                self.ma_sq = pow2(self.mhc) - mw_sq  # GH (3.17)
        assert isinstance(self.mu, (int, float))
        assert isinstance(self.ma_sq, (int, float))

        # now (mu, ma_sq) are set; mhc and ma0 are ready to set.
        if self.mhc is None:
            self.mhc = sqrt(self.ma_sq + mw_sq)
        if self.ma0 is None:
            self.ma0 = sqrt(self.ma_sq)
        # finally calculate h1_sq and mh2_sq if not set.
        if self.mh1_sq is None:
            m3_sq = self.ma_sq * sin_2beta / 2
            mu_sq = pow2(self.mu)
            self.mh1_sq = -mu_sq - (mz_sq * cos_2beta / 2) + m3_sq * self.tan_beta
            self.mh2_sq = -mu_sq + (mz_sq * cos_2beta / 2) + m3_sq / self.tan_beta

    def yu(self) -> List[float]:
        """Return the diagonal of up-type Yukawa after super-CKM rotation."""
        return [
            sqrt(2) * mass / self.sm.vev() / tan2sin(self.tan_beta)
            for mass in self.sm.mass_u()
        ]

    def yd(self) -> List[float]:
        """Return the diagonal of down-type Yukawa."""
        return [
            sqrt(2) * mass / self.sm.vev() / tan2cos(self.tan_beta)
            for mass in self.sm.mass_d()
        ]

    def ye(self) -> List[float]:
        """Return the diagonal of charged-lepton Yukawa."""
        return [
            sqrt(2) * mass / self.sm.vev() / tan2cos(self.tan_beta)
            for mass in self.sm.mass_e()
        ]

    def yukawa(self, species: A) -> List[float]:
        """Return the diagonal of Yukawa."""
        if species == A.U:
            return self.yu()
        elif species == A.D:
            return self.yd()
        elif species == A.E:
            return self.ye()
        else:
            raise RuntimeError("invalid call of ewsb.yukawa")

    def mass(self, pid: int) -> float:
        """
        Return the pole mass of the specified particle.

        The particles should be SM particle or heavy Higgses.
        """
        if (sm_particle_mass := self.sm.mass(pid)) is not NotImplemented:
            return sm_particle_mass

        if not self.is_set():
            raise RuntimeError("Invalid mass call before setting EWSB parameters.")
        assert self.ma0 is not None
        assert self.mhc is not None
        assert self.ma_sq is not None
        if self.ma_sq.imag:
            raise ValueError("This calculator does not support CPV.")

        if pid == 25 or pid == 35:  # Martin Eq.8.1.20 (see convention.pdf)
            sin_two_beta = tan2sintwo(self.tan_beta)
            a2, z2 = abs(self.ma_sq), pow2(self.sm.mz())
            factor = -1 if pid == 25 else 1
            return sqrt(
                (a2 + z2 + factor * sqrt(pow2(a2 - z2) + 4 * z2 * a2 * sin_two_beta))
                / 2
            )
        elif pid == 36:
            return self.ma0  # tree-level mass
        elif pid == 37:
            return self.mhc  # tree-level mass
        else:
            return NotImplemented


class Calculator(AbsCalculator):
    """The tree-level calculator of MSSM."""

    name = simsusy.simsusy.__pkgname__ + "/MSSMTree"
    version = simsusy.simsusy.__version__

    input: Input  # noqa: A003
    output: Output
    cpv: CPV
    flv: FLV

    document_blocks = [
        "MODSEL",
        "MINPAR",
        "EXTPAR",
        "VCKMIN",
        "UPMNSIN",
        "MSQ2IN",
        "MSU2IN",
        "MSD2IN",
        "MSL2IN",
        "MSE2IN",
        "TUIN",
        "TDIN",
        "TEIN",
    ]
    output_blocks_with_scale = [
        "HMIX",
        "GAUGE",
        "MSOFT",
        "MSQ2",
        "MSU2",
        "MSD2",
        "MSL2",
        "MSE2",
        "AU",
        "AD",
        "AE",
        "TU",
        "TD",
        "TE",
        "YU",
        "YD",
        "YE",
    ]

    def __init__(self, input: Input) -> None:  # noqa: A002
        super().__init__(logger=logger)
        self.input = input
        self.output = Output()
        self.cpv = CPV.NONE  # MODSEL parameters, set in load_modsel()
        self.flv = FLV.NONE

    def read_simsusy_options(self, slha1: bool) -> MutableMapping[int, int]:
        """Validate SIMSUSY block."""
        both_options = [101]
        slha1_options = [141]
        slha2_options = [111, 131, 132, 133, 134, 901, 902, 903, 904]
        option_keys = both_options + (slha1_options if slha1 else slha2_options)
        options = {k: 0 for k in option_keys}
        if self.input.slha.get("SIMSUSY") is None:
            return options
        for k, v0 in self.input.slha["SIMSUSY"].items():
            v = round(v0)
            if not 100 <= k <= 999:
                pass  # not calculator-level option
            elif k not in options:
                self.add_warning(f"Ignored unknown SIMSUSY option {k}={v}")
            elif v == 0:
                pass  # value zero is always allowed and default.
            elif (
                (k in (101, 111, 141) and v == 1)
                or (131 <= k <= 134 and re.match(r"^1[01][01][01]$", str(v)))
                or (901 <= k <= 904 and v >= 0)
            ):
                options[k] = v
            else:
                self.add_warning(f"Ignored invalid SIMSUSY option {k}={v}")
        return options

    def write_output(self, filename: Optional[str] = None, slha1: bool = False) -> None:
        """
        Output the results to a file.

        The only options for the `simsusy` script are `slha1` flag and an
        output path, and thus this method accepts those values. Other output
        options are specified in SIMSUSY block.
        """
        options = self.read_simsusy_options(slha1=slha1)
        self._output_preparation(slha1, options)
        self._output_prepare_spinfo()
        if options[101]:
            self.output.slha["SPINFO", 3] = []

        # dumper configuration
        self.output.dumper = yaslha.dumper.SLHADumper(
            separate_blocks=True,
            comments_preserve=yaslha.dumper.CommentsPreserve.TAIL,
            document_blocks=self.document_blocks,
        )
        self.output.write(filename)

    def _output_preparation(self, slha1, options):
        # type: (bool, MutableMapping[int, int]) -> None
        if slha1 and (self.cpv != CPV.NONE or self.flv != FLV.NONE):
            self.add_error("SLHA1 does not support CPV/FLV.")
            return
        if slha1:
            self._output_prepare_spinfo()
            self._output_reorder_sfermions(slha1=True, options=options)
            self._output_convert_slha2_to_slha1(options)
            self._output_remove_void_blocks()
        else:
            if options[111] == 0:
                self._output_rearrange_neutralino_negative_mass()
            self._output_reorder_sfermions(slha1=False, options=options)
            self._output_remove_void_blocks()
            self._output_suppress_small_matrix_entries(options)

        for tmp in self.output_blocks_with_scale:
            if tmp in self.output.blocks:
                self.output.blocks[tmp].q = 200  # TODO: more proper way...

    def _load_modsel(self) -> None:
        modsel = self.input.block("MODSEL")
        if not isinstance(modsel, yaslha.slha.Block):
            # MODSEL can be absent.
            return

        for k, v in modsel.items():
            if k == 1:
                if v not in (0, 1):
                    # accept only general MSSM or mSUGRA. (no distinction)
                    self.add_error(
                        f"Invalid MODSEL {k}={v}; should be 0 (or 1).",
                        f"Invalid MODSEL {k}={v}",
                    )
            elif k in [3, 4, 5] and v != 0:
                # tree_calculator handles only MSSM with no RpV/CPV.
                self.add_error(
                    f"Invalid MODSEL {k}={v}; should be 0.", f"Invalid MODSEL {k}={v}"
                )
            elif k == 6:
                try:
                    self.flv = FLV(v)
                except ValueError:
                    self.add_error(f"Invalid MODSEL {k}={v}")
            elif k == 11 and v != 1:
                self.add_warning(f"Ignored MODSEL {k}={v}")
            elif k == 12 or k == 21:  # defined in SLHA spec
                self.add_warning(f"Ignored MODSEL {k}={v}")
            else:
                self.add_warning(f"Ignored MODSEL {k}={v}")

    def _load_sminputs(self) -> None:
        sminputs = self.input.block("SMINPUTS")
        if isinstance(sminputs, yaslha.slha.Block):
            for k, v in sminputs.items():
                if not isinstance(k, int) or not (
                    1 <= k <= 8 or k in [11, 12, 13, 14, 21, 22, 23, 24]
                ):
                    self.add_warning(f"Invalid SMINPUTS {k}={v}")
        self.output.sm = SMParameters(self.input)

    def _load_ewsb_parameters(self) -> None:
        if (
            self.input.get_complex("MINPAR", 3, default=None) is not None
            and self.input.get_complex("EXTPAR", 25, default=None) is not None
        ):
            self.add_warning("TB in MINPAR ignored due to EXTPAR 25.")
        try:
            assert self.output.sm
            self.output.ewsb = EWSBParameters(self.input, self.output.sm)
        except ValueError as e:
            self.add_error(f"invalid EWSB spec ({e})")
        except AssertionError:
            logger.error("EWSB parameters not loaded.")

    def _check_other_input_validity(self) -> None:
        for name, content in self.input.blocks.items():
            if name in ["MODSEL", "SMINPUTS", "VCKMIN", "UPMNSIN"]:
                pass  # validity is checked when it is loaded
            elif name == "MINPAR":
                for k, v in content.items():
                    if not (isinstance(k, int) and 1 <= k <= 5):
                        self.add_warning(f"Ignored {name} {k}={v}")
            elif name == "EXTPAR":
                for k, v in content.items():
                    if not isinstance(k, int) or not (
                        k in [0, 1, 2, 3, 11, 12, 13]
                        or 21 <= k <= 27
                        or 31 <= k <= 36
                        or 41 <= k <= 49
                    ):
                        self.add_warning(f"Ignored {name} {k}={v}")
            elif name in ["QEXTPAR"]:
                self.add_warning(f"Ignored {name}: not supported.")
            elif name in ["MSQ2IN", "MSU2IN", "MSD2IN", "MSL2IN", "MSE2IN"]:
                for k, v in content.items():
                    if not (isinstance(k, tuple) and len(k) == 2):
                        self.add_error(f"Invalid {name} {k}={v}")
                    elif not (
                        1 <= k[0] <= 3 and k[0] <= k[1] <= 3
                    ):  # upper triangle only
                        self.add_warning(
                            f"Ignored {name} {k}={v}; used upper triangle.",
                            f"Ignored {name} {k}={v}",
                        )
            elif name in ["TUIN", "TDIN", "TEIN"]:
                for k, v in content.items():
                    if not (isinstance(k, tuple) and len(k) == 2):
                        self.add_error(f"Invalid {name} {k}={v}")
                    elif not (1 <= k[0] <= 3 and 1 <= k[1] <= 3):
                        self.add_warning(
                            f"Ignored {name} {k}={v}; used upper triangle.",
                            f"Ignored {name} {k}={v}",
                        )
            else:
                self.add_warning(f"Ignored {name}: unknown.")

    def _check_cpv_flv_consistency(self) -> None:
        # CPV consistency
        if self.cpv != CPV.FULL:
            for s_type in S:
                if not is_real_matrix(self.input.ms2(s_type)):
                    self.add_error(
                        f"CPV is conserved while complex entry in {s_type.slha2_input}."
                    )
            for a_type in A:
                m = self.input.a(a_type)
                if m is None:
                    m = self.input.t(a_type)
                if not is_real_matrix(m):
                    self.add_error(
                        "CPV is conserved while complex entry in A- or T-matrix."
                    )
            for i in [1, 2, 3]:
                if isinstance(self.input.mg(i), complex):
                    self.add_error("CPV is conserved while gaugino mass is not real")

            if self.cpv == CPV.NONE:
                if not is_real_matrix(self.input.vckm()):
                    self.add_warning("CKM CP-phase is ignored due to MODSEL-5")
                    self.input.slha["VCKMIN", 4] = 0
                if not is_real_matrix(self.input.upmns()):
                    self.add_warning("PMNS CP-phase is ignored due to MODSEL-5")
                    for i in (4, 5, 6):
                        self.input.slha["UPMNSIN", i] = 0

        if not self.flv.qfv():
            for matrix in [
                self.input.ms2(S.QL),
                self.input.ms2(S.UR),
                self.input.ms2(S.DR),
                self.input.a(A.U) if self.input.t(A.U) is None else self.input.t(A.U),
                self.input.a(A.D) if self.input.t(A.D) is None else self.input.t(A.D),
            ]:
                if not is_diagonal_matrix(matrix):
                    self.add_error("Quark flavor is set conserved in MODSEL.")
            if not is_diagonal_matrix(self.input.vckm()):
                self.add_warning("VCKMIN is ignored due to MODSEL-6.")
                self.input.remove_block("VCKMIN")

        if not self.flv.lfv():
            for matrix in [
                self.input.ms2(S.LL),
                self.input.ms2(S.ER),
                self.input.a(A.E) if self.input.t(A.E) is None else self.input.t(A.E),
            ]:
                if not is_diagonal_matrix(matrix):
                    self.add_error("Lepton flavor is set conserved in MODSEL.")
            if not is_diagonal_matrix(self.input.upmns()):
                self.add_warning("UPMNSIN is ignored due to MODSEL-6.")
                self.input.remove_block("UPMNSIN")

    def calculate(self) -> None:
        """Calculate the MSSM parameters."""
        self.output.input = self.input
        self._load_modsel()
        self._load_sminputs()
        self._load_ewsb_parameters()
        self._check_other_input_validity()
        self._check_cpv_flv_consistency()
        if self.output.sm is None or self.output.ewsb is None:
            logger.error("Model set-up failed.")
            exit(1)
        self._prepare_info()
        self._prepare_sm_ewsb()
        self._calculate_softmasses()
        self._calculate_higgses()
        self._calculate_neutralino()
        self._calculate_chargino()
        self._calculate_gluino()
        self._calculate_sfermion()

    def _prepare_info(self) -> None:
        self.output.slha["SPINFO", 1] = [self.name]
        self.output.slha["SPINFO", 2] = [self.version]
        self.output.slha["SPINFO", 3] = []
        self.output.slha["SPINFO", 4] = []

    def _prepare_sm_ewsb(self) -> None:
        assert self.output.sm
        assert self.output.ewsb

        for pid in [5, 6, 15, 23, 24]:
            self.output.set_mass(pid, self.output.sm.mass(pid))
        self.output.slha["ALPHA", None] = self.output.ewsb.alpha()
        self.output.slha["HMIX", 1] = self.output.ewsb.mu
        self.output.slha["HMIX", 2] = self.output.ewsb.tan_beta
        self.output.slha["HMIX", 3] = self.output.sm.vev()
        self.output.slha["HMIX", 4] = self.output.ewsb.ma_sq
        self.output.slha["GAUGE", 1] = self.output.sm.gy()
        self.output.slha["GAUGE", 2] = self.output.sm.gw()
        self.output.slha["GAUGE", 3] = self.output.sm.gs()

    def _calculate_softmasses(self) -> None:
        assert self.output.ewsb
        for i in [1, 2, 3]:
            self.output.slha["MSOFT", i] = self.input.mg(i)
        self.output.slha["MSOFT", 21] = self.output.ewsb.mh1_sq
        self.output.slha["MSOFT", 22] = self.output.ewsb.mh2_sq

        # Store according to SLHA2 scheme;
        # for SLHA1 output, convert them to SLHA1 format when output.
        self.output.set_matrix("VCKM", self.input.vckm().real)
        self.output.set_matrix("IMVCKM", np.zeros((3, 3)))  # CPV ignored
        self.output.set_matrix("UPMNS", self.input.upmns().real)
        self.output.set_matrix("IMUPMNS", np.zeros((3, 3)))  # CPV ignored
        for a_type in [A.U, A.D, A.E]:
            # y is diagonal in super-CKM basis
            y = np.diag(self.output.ewsb.yukawa(a_type))
            self.output.set_matrix(a_type.out_y, y, diagonal_only=True)

            a = self.input.a(a_type)
            t = self.input.t(a_type) if a is None else y * a
            assert t is not None
            # CPV ignored
            self.output.set_matrix(a_type.out_t, t.real)
        for s_type in [S.QL, S.UR, S.DR, S.LL, S.ER]:
            # CPV ignored
            self.output.set_matrix(s_type.slha2_output, self.input.ms2(s_type).real)

    def _calculate_higgses(self) -> None:
        assert self.output.ewsb
        for pid in [25, 35, 36, 37]:
            self.output.set_mass(pid, self.output.ewsb.mass(pid))

    def _neutralino_matrix(self) -> RealMatrix:
        """Return the neutralino mass matrix, ignoring CPV."""
        assert self.output.sm
        assert self.output.ewsb
        # CPV ignored
        cb = tan2cos(self.output.ewsb.tan_beta.real)
        sb = tan2sin(self.output.ewsb.tan_beta.real)
        sw = sqrt(self.output.sm.sin_w_sq())
        cw = sin2cos(sw)
        mz = self.output.sm.mz()
        m1 = self.output.get_float("MSOFT", 1)
        m2 = self.output.get_float("MSOFT", 2)
        mu = self.output.ewsb.mu
        assert mu is not None
        # Since CPV is not supported, m_psi0 is real and nmix will be real.
        return np.array(
            [
                [m1, 0, -mz * cb * sw, mz * sb * sw],
                [0, m2, mz * cb * cw, -mz * sb * cw],
                [-mz * cb * sw, mz * cb * cw, 0, -mu.real],
                [mz * sb * sw, -mz * sb * cw, -mu.real, 0],
            ]
        )  # SLHA.pdf Eq.(21)

    def _calculate_neutralino(self) -> None:
        masses, nmix = autonne_takagi(self._neutralino_matrix(), try_real_mixing=True)
        for i, pid in enumerate(neutralino_pids):
            self.output.set_mass(pid, masses[i])
        assert is_real_matrix(nmix)
        self.output.set_matrix("NMIX", nmix.real)

    def _calculate_chargino(self) -> None:
        assert self.output.sm
        assert self.output.ewsb
        cb = tan2cos(self.output.ewsb.tan_beta)
        sb = tan2sin(self.output.ewsb.tan_beta)
        sqrt2_mw = sqrt(2) * self.output.sm.mass(24)  # use pole value
        m2 = self.output.get_float("MSOFT", 2)
        mu = self.output.ewsb.mu

        m_psi_plus = np.array([[m2, sqrt2_mw * sb], [sqrt2_mw * cb, mu]])  # SLHA (22)
        masses, umix, vmix = singular_value_decomposition(m_psi_plus)
        for i, pid in enumerate(chargino_pids):
            self.output.set_mass(pid, masses[i])
        assert is_real_matrix(umix)
        assert is_real_matrix(vmix)
        self.output.set_matrix("UMIX", umix.real)
        self.output.set_matrix("VMIX", vmix.real)

    def _calculate_gluino(self) -> None:
        self.output.set_mass(1000021, self.output.get_float("MSOFT", 3))

    def _calculate_sfermion(self) -> None:
        assert self.output.sm
        assert self.output.ewsb
        mu = self.output.ewsb.mu
        tan_beta = self.output.ewsb.tan_beta
        assert mu is not None
        mz2_cos2b = (
            np.diag([1, 1, 1]) * pow2(self.output.sm.mz()) * tan2costwo(tan_beta)
        )
        sw2 = self.output.sm.sin_w_sq()
        mu_tan_b = mu * tan_beta
        mu_cot_b = mu / tan_beta
        ckm = self.output.get_matrix("VCKM", default=np.diag([1, 1, 1]))
        pmns = self.output.get_matrix("UPMNS", default=np.diag([1, 1, 1]))

        def dag(m: T) -> T:
            return np.conjugate(m.T)

        def m_join(m11: T, m12: T, m21: T, m22: T) -> T:
            return np.vstack([np.hstack([m11, m12]), np.hstack([m21, m22])])

        def mass_matrix(right: S) -> RealMatrix:
            assert self.output.sm
            assert self.output.ewsb
            mu = self.output.ewsb.mu
            tan_beta = self.output.ewsb.tan_beta
            assert mu is not None
            if right == S.UR:
                (left, a_species, mf) = (S.QL, A.U, self.output.sm.mass_u())
                dl, dr, mu = (1 / 2 - 2 / 3 * sw2), 2 / 3 * sw2, mu_cot_b
                vev = self.output.sm.vev() * tan2sin(tan_beta)
            elif right == S.DR:
                (left, a_species, mf) = (S.QL, A.D, self.output.sm.mass_d())
                dl, dr, mu = -(1 / 2 - 1 / 3 * sw2), -1 / 3 * sw2, mu_tan_b
                vev = self.output.sm.vev() * tan2cos(tan_beta)
            elif right == S.ER:
                (left, a_species, mf) = (S.LL, A.E, self.output.sm.mass_e())
                dl, dr, mu = -(1 / 2 - sw2), -sw2, mu_tan_b
                vev = self.output.sm.vev() * tan2cos(tan_beta)
            else:
                raise NotImplementedError
            msl2 = self.output.get_matrix(left.slha2_output)
            if right == S.UR:
                msl2 = ckm @ msl2 @ dag(ckm)
            msr2 = self.output.get_matrix(right.slha2_output)
            mf_mat = np.diag(mf)
            t = self.output.get_matrix(a_species.out_t)

            # SLHA Eq.23-25 and SUSY Primer (8.4.18)
            return m_join(
                msl2 + mf_mat * mf_mat + dl * mz2_cos2b,
                vev / sqrt(2) * dag(t) - mu.real * mf_mat,
                vev / sqrt(2) * t - np.conjugate(mu).real * mf_mat,
                msr2 + mf_mat * mf_mat + dr * mz2_cos2b,
            )

        def sneutrino_mass() -> ComplexMatrix:
            assert self.output.sm
            msl2 = self.output.get_matrix(S.LL.slha2_output)
            mf_sq: RealMatrix = np.diag([v * v for v in self.output.sm.mass_n()])
            dl = 1 / 2
            return pmns @ msl2 @ dag(pmns) + mf_sq + dl * mz2_cos2b

        def prettify_matrix(m: RealMatrix, threshold: float = 1e-10) -> RealMatrix:
            nx, ny = m.shape
            for i in range(nx):
                m[i] = m[i] * (1 if max(m[i], key=lambda v: abs(float(v))) > 0 else -1)
                for j in range(ny):
                    if abs(m[i, j]) < threshold:
                        m[i, j] = 0
            return m

        for (right, mix) in [(S.UR, SMIX.U), (S.DR, SMIX.D), (S.ER, SMIX.L)]:
            mass_sq, f = mass_diagonalization(mass_matrix(right))
            for i, pid in enumerate(mix.pids):
                self.output.set_mass(pid, sqrt(mass_sq[i]))
            assert is_real_matrix(f)
            self.output.set_matrix(mix.slha2mix, prettify_matrix(f.real))

        mass_sq, f = mass_diagonalization(sneutrino_mass())
        for i, pid in enumerate(SMIX.N.pids):
            self.output.set_mass(pid, sqrt(mass_sq[i]))
        assert is_real_matrix(f)
        self.output.set_matrix(SMIX.N.slha2mix, prettify_matrix(f.real))

    def _output_remove_void_blocks(self) -> None:
        """Remove blocks without items and IM-blocks with no non-zero entry."""
        kill_blocks: List[str] = []
        for name in self.output.blocks:
            block = self.output.slha[name]
            if len([block.keys()]) == 0:
                kill_blocks.append(name)
            elif name.startswith("IM"):
                to_kill = True
                for _, value in block.items():
                    if abs(value) > 0:
                        to_kill = False
                        break
                if to_kill:
                    kill_blocks.append(name)
        for block_name in kill_blocks:
            del self.output.blocks[block_name]

    def _output_rearrange_neutralino_negative_mass(self) -> None:
        """
        Handle negative values of neutralino masses.

        Since this calculator does not handle CPV, _calculate_neutralino
        asserts NMIX is a real matrix. If SLHA2 output is requested (and the
        option 111 is not set to 1), we rotate the NMIX to be complex and force
        the mass to be positive (SLHA2 default).
        """
        # let neutralino-mass positive with imaginary NMIX.
        for i, pid in enumerate(neutralino_pids):
            mass = self.output.mass(pid)
            for j in range(4):
                old_mix = self.output.get_complex("NMIX", i + 1, j + 1)
                new_mix = old_mix * (1j if mass < 0 else 1)
                self.output.slha["NMIX", i + 1, j + 1] = new_mix.real
                self.output.slha["IMNMIX", i + 1, j + 1] = new_mix.imag
            if mass < 0:
               self.output.set_mass(pid, abs(mass))

    def _output_reorder_sfermions(self, slha1, options):
        # type: (bool, MutableMapping[int, int]) -> None
        """Reorder sfermion blocks according to SIMSUSY options."""
        targets = [  # mixing matrix and SIMSUSY option key
            (SMIX.D, 131),
            (SMIX.U, 132),
            (SMIX.L, 133),
            (SMIX.N, 134),
        ]
        for (smix, option_key) in targets:
            if slha1:
                self._output_reorder_sfermion_in_flavor(smix, (1, 1, 0))
            elif (option := options[option_key]) == 0:
                # SLHA2 default
                self._output_reorder_sfermion_in_mass(smix)
            else:
                order = (option % 1000 // 100, option % 100 // 10, option % 10)
                self._output_reorder_sfermion_in_flavor(smix, order)

    def _output_reorder_sfermion_in_mass(self, smix: SMIX) -> None:
        masses = [(i, self.output.mass(p)) for i, p in enumerate(smix.pids)]
        masses.sort(key=lambda x: abs(x[1]) if x[1] else 0)
        order = [i for i, _ in masses]
        self.__reorder_sfermions(smix, order)

    def _output_reorder_sfermion_in_flavor(self, smix, order_by_lr_flags):
        # type: (SMIX, Tuple[int, int, int]) -> None
        mixing = self.output.get_complex_matrix(smix.slha2mix)
        assert (n := len(smix.pids)) in [3, 6]
        assert isinstance(mixing, np.ndarray)
        assert len(mixing.shape) == 2 and mixing.shape[0] == mixing.shape[1] == n
        m = mixing.copy()

        def get_largest() -> Tuple[int, int]:
            x, y, max_value = -1, -1, -1.0
            for i, j in itertools.product(range(n), range(n)):
                if (v := abs(m[i, j])) > max_value:
                    x, y, max_value = i, j, v
            return x, y

        # first find the ordering based on the gauge eigenstates.
        order = [-1 for i in range(n)]
        for i in range(n):
            pivot_x, pivot_y = get_largest()
            order[pivot_y] = pivot_x
            for j in range(n):
                m[pivot_x, j] = 0
        # then, if not order_by_lr for i-th generation, reorder in their masses.
        if n == 6:
            for i in range(3):
                if not order_by_lr_flags[i]:
                    il, ir = order[i], order[i + 3]
                    ml = self.output.mass(smix.pids[il])
                    mr = self.output.mass(smix.pids[ir])
                    if mr < ml:
                        order[i], order[i + 3] = ir, il
        self.__reorder_sfermions(smix, order)

    def __reorder_sfermions(self, smix: SMIX, order: List[int]) -> None:
        n = len(order)
        assert len(smix.pids) == n and sorted(order) == list(range(n))
        mixing = self.output.get_complex_matrix(smix.slha2mix)
        masses: List[float] = [self.output.mass(p) for p in smix.pids]

        im_matrix = "IM" + smix.slha2mix
        is_complex = im_matrix in self.output.blocks
        for i, x in enumerate(order):
            self.output.set_mass(smix.pids[i], masses[x])
            for j in range(n):
                self.output.slha[smix.slha2mix, i + 1, j + 1] = mixing[x, j].real
                if is_complex:
                    self.output.slha[im_matrix, i + 1, j + 1] = mixing[x, j].imag

    def _output_convert_slha2_to_slha1(self, options: MutableMapping[int, int]) -> None:
        # soft masses
        for s_type in S:
            for gen in (1, 2, 3):
                self.output.slha["MSOFT", s_type.extpar + gen] = sqrt(
                    self.output.get_float(s_type.slha2_output, gen, gen)
                )
            self.output.remove_block(s_type.slha2_output)

        for t_type in A:
            for i in (1, 2, 3):
                if options[141] == 0 and i != 3:
                    continue
                self.output.slha[t_type.out_a, i, i] = self.output.get_float(
                    t_type.out_t, i, i
                ) / self.output.get_float(t_type.out_y, i, i)
            self.output.remove_block(t_type.out_t)

        # mass and mixing: quark flavor rotation is already flavor-ordered.
        for smix in [SMIX.D, SMIX.U, SMIX.L]:
            r = self.output.get_matrix(smix.slha2mix)
            max_mix = (-1, -1, 0)
            for i in range(6):
                for j in range(6):
                    if i == j or (i in (2, 5) and j in (2, 5)):
                        pass
                    elif (v := abs(r[i, j])) > max_mix[2]:
                        max_mix = (i + 1, j + 1, v)
            if max_mix[2] > 0:
                self.add_warning(
                    "Ignored lighter-gen {} (max:{}{} = {:.2e})".format(
                        smix.slha2mix, *max_mix
                    )
                )

            self.output.slha[smix.slha1mix, 1, 1] = r[2][2]
            self.output.slha[smix.slha1mix, 1, 2] = r[2][5]
            self.output.slha[smix.slha1mix, 2, 1] = r[5][2]
            self.output.slha[smix.slha1mix, 2, 2] = r[5][5]
            self.output.remove_block(smix.slha2mix)

        assert is_diagonal_matrix(self.output.get_matrix("SNUMIX"))
        self.output.remove_block("SNUMIX")

        # remove SLHA2 blocks
        for block_name in ("VCKM", "IMVCKM", "UPMNS", "IMUPMNS"):
            self.output.remove_block(block_name)

    def _output_suppress_small_matrix_entries(self, options):
        # type: (MutableMapping[int, int]) -> None
        """Remove small matrix elements according to the options."""

        def remove(names: List[str], option_key: int) -> None:
            if (option := options[option_key]) == 0:
                return
            threshold_squared = None if option == 1 else pow(0.1, option * 2)
            for name in names:
                rb, ib = self.output.block(name), self.output.block("IM" + name)
                if isinstance(rb, yaslha.block.InfoBlock) or isinstance(
                    ib, yaslha.block.InfoBlock
                ):
                    continue
                re_keys = set(rb.keys()) if rb else set()
                im_keys = set(ib.keys()) if ib else set()
                for k in re_keys | im_keys:
                    r = rb.get(k, default=None) if rb else None
                    i = ib.get(k, default=None) if ib else None
                    if threshold_squared is None:
                        to_remove = (not r) and (not i)
                    else:
                        value = pow2(r if r else 0) + pow2(i if i else 0)
                        to_remove = value < threshold_squared
                    if to_remove and rb and r is not None:
                        del rb[k]
                    if to_remove and ib and i is not None:
                        del ib[k]

        remove(["MSQ2", "MSU2", "MSD2", "MSL2", "MSE2"], 901)
        remove(["USQMIX", "DSQMIX", "SELMIX", "SNUMIX"], 902)
        remove(["YU", "YD", "YE", "TU", "TD", "TE"], 903)
        remove(["VCKM", "UPMNS"], 904)
