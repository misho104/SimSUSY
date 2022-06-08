import logging
from math import atan, pi, sqrt
from typing import List, Optional, TypeVar, Union  # noqa: F401

import numpy as np
import numpy.typing
import yaslha
import simsusy.simsusy
from simsusy.abs_calculator import AbsCalculator
from simsusy.mssm.abstract import AbsEWSBParameters, AbsSMParameters
from simsusy.mssm.input import MSSMInput as Input
from simsusy.mssm.library import CPV, FLV, A, S
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


def pow2(v: float) -> float:
    return v * v


class SMParameters(AbsSMParameters):
    def __init__(self, input):  # type: (Input) -> None
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
    Note that all the variables are calculated at "Tree Level".
    """

    def _sin_sq_cos_sq(self) -> float:
        """Return sin^2(theta_w)*cos^2(theta_w)."""
        return pi / (sqrt(2) * self._alpha_em_inv * self._g_fermi * self._mz * self._mz)

    def sin_w_sq(self) -> float:
        """Return sin^2(theta_w)."""
        r = self._sin_sq_cos_sq()
        return 2 * r / (1 + sqrt(1 - 4 * r))

    def cos_w_sq(self) -> float:
        """Return cos^2(theta_w)."""
        r = self._sin_sq_cos_sq()
        return (1 + sqrt(1 - 4 * r)) / 2

    def mz(self) -> float:
        """Return the Z-boson mass."""
        return self._mz

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
        """Return the mass of the particle with the given PDG ID."""
        value = super().mass(pid)
        if value is not NotImplemented:
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
    def __init__(self, input_obj, sm):  # type: (Input, SMParameters) -> None
        super().__init__(input_obj)
        self.sm = sm
        self.calculate()
        assert self.is_set()

    def calculate(self) -> None:
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
            if isinstance(v, complex):
                raise ValueError("This calculator does not support CPV.")

        # see doc/convention.pdf for equations and discussion.
        assert isinstance(self.tan_beta, float)
        cos_2beta = tan2costwo(self.tan_beta)
        sin_2beta = tan2sintwo(self.tan_beta)
        mz_sq = pow2(self.sm.mz())
        mw_sq = pow2(self.sm.mw())

        # first calculate mu if absent.
        if self.mu is None:
            assert isinstance(self.mh1_sq, float)
            assert isinstance(self.mh2_sq, float)
            # Martin (8.1.11)
            mu_sq = 0.5 * (
                -self.mh2_sq
                - self.mh1_sq
                - mz_sq
                + abs((self.mh1_sq - self.mh2_sq) / cos_2beta)
            )
            if self.sign_mu is None:
                raise ValueError("Block MINPAR 4 is required.")
            if mu_sq < 0:
                raise ValueError("Failed to get EWSB: mu^2 < 0")
            self.mu = self.sign_mu * sqrt(mu_sq)

            # GH (3.22) or Martin (8.1.10)
            self.ma_sq = self.mh1_sq + self.mh2_sq + 2.0 * mu_sq
            if self.ma_sq < 0:
                raise ValueError("Failed to get EWSB: mA^2 < 0")
        else:
            # set mA^2.
            if self.ma0 is not None:
                assert isinstance(self.ma0, float)
                self.ma_sq = pow2(self.ma0)  # at the tree level
            elif self.mhc is not None:
                assert isinstance(self.mhc, float)
                self.ma_sq = pow2(self.mhc) - mw_sq  # GH (3.17)

        # now (mu, ma_sq) are set and (mh1_sq, mh2_sq) may be set.
        assert isinstance(self.ma_sq, float)
        if self.mhc is None:
            self.mhc = sqrt(self.ma_sq + mw_sq)
        if self.ma0 is None:
            self.ma0 = sqrt(self.ma_sq)

        # finally calculate h1_sq and mh2_sq if not set.
        if self.mh1_sq is None:
            assert isinstance(self.mu, float)
            m3_sq = self.ma_sq * sin_2beta / 2
            mu_sq = pow2(self.mu)
            self.mh1_sq = -mu_sq - (mz_sq * cos_2beta / 2) + m3_sq * self.tan_beta
            self.mh2_sq = -mu_sq + (mz_sq * cos_2beta / 2) + m3_sq / self.tan_beta

    def alpha(self) -> float:
        assert isinstance(self.tan_beta, float)
        tan_twobeta, ma, mz = tan2tantwo(self.tan_beta), self.ma0, self.sm.mz()
        assert isinstance(ma, float) and isinstance(mz, float)
        return 0.5 * atan((pow2(ma) + pow2(mz)) / (ma + mz) / (ma - mz) * tan_twobeta)

    def yu(self) -> List[float]:
        """Returns the diagonal elements of the Yukawa matrix (after super-CKM
        rotation)"""
        assert isinstance(self.tan_beta, float)
        return [
            sqrt(2) * mass / self.sm.vev() / tan2sin(self.tan_beta)
            for mass in self.sm.mass_u()
        ]

    def yd(self) -> List[float]:
        assert isinstance(self.tan_beta, float)
        return [
            sqrt(2) * mass / self.sm.vev() / tan2cos(self.tan_beta)
            for mass in self.sm.mass_d()
        ]

    def ye(self) -> List[float]:
        assert isinstance(self.tan_beta, float)
        return [
            sqrt(2) * mass / self.sm.vev() / tan2cos(self.tan_beta)
            for mass in self.sm.mass_e()
        ]

    def yukawa(self, species: A) -> List[float]:
        if species == A.U:
            return self.yu()
        elif species == A.D:
            return self.yd()
        elif species == A.E:
            return self.ye()
        else:
            raise RuntimeError("invalid call of ewsb.yukawa")

    def mass(self, pid: int) -> float:
        sm_value = self.sm.mass(pid)
        if isinstance(sm_value, float):
            return sm_value
        elif pid == 25 or pid == 35:  # Martin Eq.8.1.20 (see convention.pdf)
            sin_two_beta = tan2sintwo(self.tan_beta)
            a2, z2 = self.ma_sq, pow2(self.sm.mz())
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
    name = simsusy.simsusy.__pkgname__ + "/MSSMTree"
    version = simsusy.simsusy.__version__
    input: Input

    def __init__(self, input: Input) -> None:
        super().__init__(input=input, logger=logger)
        self.output = Output()  # type: Output
        # MODSEL parameters set in load_modsel
        self.cpv = CPV.NONE  # type: CPV
        self.flv = FLV.NONE  # type: FLV

    def write_output(self, filename: Optional[str] = None, slha1: bool = False) -> None:
        if slha1:
            if self.cpv == CPV.NONE and self.flv == FLV.NONE:
                self.convert_slha2_to_slha1()
            else:
                self.add_warning("SLHA1 does not support CPV/FLV.")
        for tmp in [
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
        ]:
            if tmp in self.output.blocks:
                self.output.blocks[tmp].q = 200  # TODO: more proper way...
        self.output.write(filename)

    def _load_modsel(self) -> None:
        modsel = self.input.block("MODSEL")
        assert isinstance(modsel, yaslha.slha.Block)
        for k, v in modsel.items():
            if k == 1 and v not in (
                0,
                1,
            ):  # accept only general MSSM or mSUGRA. (no distinction)
                self.add_error(
                    f"Block MODSEL: {k} = {v} is invalid; should be 0 (or 1)."
                )
            elif (
                k in [3, 4, 5] and v != 0
            ):  # tree_calculator handles only MSSM with no RpV/CPV.
                self.add_error(f"Block MODSEL: {k} = {v} is invalid; should be 0.")
            elif k == 6:
                try:
                    self.flv = FLV(v)
                except ValueError:
                    self.add_error(f"Block MODSEL: {k} = {v} is invalid.")
            elif k == 11 and v != 1:
                self.add_warning(f"Block MODSEL: {k} = {v} is ignored.")
            elif k == 12 or k == 21:  # defined in SLHA spec
                self.add_warning(f"Block MODSEL: {k} = {v} is ignored.")
            else:
                self.add_warning(f"Block MODSEL: {k} = {v} is ignored.")

    def _load_sminputs(self) -> None:
        sminputs = self.input.block("SMINPUTS")
        assert isinstance(sminputs, yaslha.slha.Block)
        for k, v in sminputs.items():
            if not isinstance(k, int) or not (
                1 <= k <= 8 or k in [11, 12, 13, 14, 21, 22, 23, 24]
            ):
                self.add_warning(f"Block SMINPUTS: {k} = {v} is invalid.")
        self.output.sm = SMParameters(self.input)

    def _load_ewsb_parameters(self) -> None:
        if self.input.get_complex("MINPAR", 3) and self.input.get_complex("EXTPAR", 25):
            self.add_warning("TanBeta in MINPAR is ignored due to EXTPAR-25.")
        try:
            assert isinstance(self.output.sm, SMParameters)
            self.output.ewsb = EWSBParameters(self.input, self.output.sm)
        except ValueError as e:
            self.add_error(f"invalid EWSB specification ({e})")
        except AssertionError:
            logger.error("EWSB parameters are not loaded.")

    def _check_other_input_validity(self) -> None:
        for name, content in self.input.blocks.items():
            if name in ["MODSEL", "SMINPUTS", "VCKMIN", "UPMNSIN"]:
                pass  # validity is checked when it is loaded
            elif name == "MINPAR":
                for k, v in content.items():
                    if not (isinstance(k, int) and 1 <= k <= 5):
                        self.add_warning(f"Block {name}: {k} = {v} is ignored.")
            elif name == "EXTPAR":
                for k, v in content.items():
                    if not isinstance(k, int) or not (
                        k in [0, 1, 2, 3, 11, 12, 13]
                        or 21 <= k <= 27
                        or 31 <= k <= 36
                        or 41 <= k <= 49
                    ):
                        self.add_warning(f"Block {name}: {k} = {v} is ignored.")
            elif name in ["QEXTPAR"]:
                self.add_warning(f"BLOCK {name} is not supported and ignored.")
            elif name in ["MSQ2IN", "MSU2IN", "MSD2IN", "MSL2IN", "MSE2IN"]:
                for k, v in content.items():
                    if not (isinstance(k, tuple) and len(k) == 2):
                        self.add_error(f"Block {name}: {k} = {v} is invalid.")
                    elif not (
                        1 <= k[0] <= 3 and k[0] <= k[1] <= 3
                    ):  # upper triangle only
                        self.add_warning(
                            f"Block {name}: {k} = {v} is ignored; upper triangle is used."
                        )
            elif name in ["TUIN", "TDIN", "TEIN"]:
                for k, v in content.items():
                    if not (isinstance(k, tuple) and len(k) == 2):
                        self.add_error(f"Block {name}: {k} = {v} is invalid.")
                    elif not (1 <= k[0] <= 3 and 1 <= k[1] <= 3):
                        self.add_warning(
                            f"Block {name}: {k} = {v} is ignored; upper triangle is used."
                        )
            else:
                self.add_warning(f"Unknown block {name} is ignored.")

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
                    self.add_error(f"CPV is conserved while gaugino mass is not real")

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
        self.output.input = self.input
        self._load_modsel()
        self._load_sminputs()
        self._load_ewsb_parameters()
        self._check_other_input_validity()
        self._check_cpv_flv_consistency()
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
        assert self.output.sm is not None
        assert self.output.ewsb is not None

        assert isinstance(self.output.ewsb.mu, float)
        assert isinstance(self.output.ewsb.tan_beta, float)
        assert isinstance(self.output.ewsb.ma_sq, float)

        # isinstance(self.ewsb, EWSBParameters)
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
        assert isinstance(self.output.ewsb, EWSBParameters)
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
        assert isinstance(self.output.ewsb, EWSBParameters)
        for pid in [25, 35, 36, 37]:
            self.output.set_mass(pid, self.output.ewsb.mass(pid))

    def _calculate_neutralino(self) -> None:
        assert isinstance(self.output.sm, SMParameters)
        assert isinstance(self.output.ewsb, EWSBParameters)
        # CPV ignored
        cb = tan2cos(self.output.ewsb.tan_beta.real)
        sb = tan2sin(self.output.ewsb.tan_beta.real)
        sw = sqrt(self.output.sm.sin_w_sq())
        cw = sin2cos(sw)
        mz = self.output.sm.mz()
        m1 = self.output.get_float_assert("MSOFT", 1)
        m2 = self.output.get_float_assert("MSOFT", 2)
        mu = self.output.ewsb.mu
        assert mu

        # Since CPV is not supported, m_psi0 is real and nmix will be real.
        m_psi0 = np.array(
            [
                [m1, 0, -mz * cb * sw, mz * sb * sw],
                [0, m2, mz * cb * cw, -mz * sb * cw],
                [-mz * cb * sw, mz * cb * cw, 0, -mu.real],
                [mz * sb * sw, -mz * sb * cw, -mu.real, 0],
            ]
        )  # SLHA (21)
        masses, nmix = autonne_takagi(m_psi0, try_real_mixing=True)
        self.output.set_mass(1000022, masses[0])
        self.output.set_mass(1000023, masses[1])
        self.output.set_mass(1000025, masses[2])
        self.output.set_mass(1000035, masses[3])
        assert is_real_matrix(nmix)
        self.output.set_matrix("NMIX", nmix.real)

    def _calculate_chargino(self) -> None:
        assert isinstance(self.output.sm, SMParameters)
        assert isinstance(self.output.ewsb, EWSBParameters)
        assert isinstance(self.output.ewsb.tan_beta, float)
        cb = tan2cos(self.output.ewsb.tan_beta)
        sb = tan2sin(self.output.ewsb.tan_beta)
        sqrt2_mw = sqrt(2) * self.output.sm.mw()
        m2 = self.output.get_float("MSOFT", 2)
        mu = self.output.ewsb.mu

        m_psi_plus = np.array([[m2, sqrt2_mw * sb], [sqrt2_mw * cb, mu]])  # SLHA (22)
        masses, umix, vmix = singular_value_decomposition(m_psi_plus)
        self.output.set_mass(1000024, masses[0])
        self.output.set_mass(1000037, masses[1])
        assert is_real_matrix(umix)
        assert is_real_matrix(vmix)
        self.output.set_matrix("UMIX", umix.real)
        self.output.set_matrix("VMIX", vmix.real)

    def _calculate_gluino(self) -> None:
        self.output.set_mass(1000021, self.output.get_float_assert("MSOFT", 3))

    def _calculate_sfermion(self) -> None:
        assert isinstance(self.output.sm, SMParameters)
        assert isinstance(self.output.ewsb, EWSBParameters)
        assert isinstance(self.output.ewsb.tan_beta, float)
        mu = self.output.ewsb.mu
        tan_beta = self.output.ewsb.tan_beta
        assert mu
        assert isinstance(tan_beta, float)
        mz2_cos2b = (
            np.diag([1, 1, 1])
            * pow2(self.output.sm.mz())
            * tan2costwo(self.output.ewsb.tan_beta)
        )
        sw2 = self.output.sm.sin_w_sq()
        mu_tan_b = mu * self.output.ewsb.tan_beta
        mu_cot_b = mu / self.output.ewsb.tan_beta
        ckm = self.output.get_matrix_assert("VCKM", default=np.diag([1, 1, 1]))
        pmns = self.output.get_matrix_assert("UPMNS", default=np.diag([1, 1, 1]))

        def dag(m: T) -> T:
            return np.conjugate(m.T)

        def m_join(m11: T, m12: T, m21: T, m22: T) -> T:
            return np.vstack([np.hstack([m11, m12]), np.hstack([m21, m22])])

        def mass_matrix(right: S) -> RealMatrix:
            assert self.output.sm and self.output.ewsb
            mu = self.output.ewsb.mu
            tan_beta = self.output.ewsb.tan_beta
            assert isinstance(mu, float)
            assert isinstance(tan_beta, float)
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
            msl2 = self.output.get_matrix_assert(left.slha2_output)
            if right == S.UR:
                msl2 = ckm @ msl2 @ dag(ckm)
            msr2 = self.output.get_matrix_assert(right.slha2_output)
            mf_mat = np.diag(mf)
            t = self.output.get_matrix_assert(a_species.out_t)

            # SLHA Eq.23-25 and SUSY Primer (8.4.18)
            return m_join(
                msl2 + mf_mat * mf_mat + dl * mz2_cos2b,
                vev / sqrt(2) * dag(t) - mu.real * mf_mat,
                vev / sqrt(2) * t - np.conjugate(mu).real * mf_mat,
                msr2 + mf_mat * mf_mat + dr * mz2_cos2b,
            )

        def sneutrino_mass() -> ComplexMatrix:
            assert self.output.sm
            msl2 = self.output.get_matrix_assert(S.LL.slha2_output)
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

        for (right, pid, mix_name) in (
            (S.UR, 1, "USQMIX"),
            (S.DR, 0, "DSQMIX"),
            (S.ER, 10, "SELMIX"),
        ):
            mass_sq, f = mass_diagonalization(mass_matrix(right))
            self.output.set_mass(1000001 + pid, sqrt(mass_sq[0]))
            self.output.set_mass(1000003 + pid, sqrt(mass_sq[1]))
            self.output.set_mass(1000005 + pid, sqrt(mass_sq[2]))
            self.output.set_mass(2000001 + pid, sqrt(mass_sq[3]))
            self.output.set_mass(2000003 + pid, sqrt(mass_sq[4]))
            self.output.set_mass(2000005 + pid, sqrt(mass_sq[5]))
            assert is_real_matrix(f)
            self.output.set_matrix(mix_name, prettify_matrix(f.real))

        mass_sq, f = mass_diagonalization(sneutrino_mass())
        self.output.set_mass(1000012, sqrt(mass_sq[0]))
        self.output.set_mass(1000014, sqrt(mass_sq[1]))
        self.output.set_mass(1000016, sqrt(mass_sq[2]))
        assert is_real_matrix(f)
        self.output.set_matrix("SNUMIX", prettify_matrix(f.real))

    def convert_slha2_to_slha1(self) -> None:
        if not (self.cpv == CPV.NONE and self.flv == FLV.NONE):
            self.add_error("SLHA1 does not support Flavor/CP violations.")
            return
        # soft masses
        for s_type in S:
            for gen in (1, 2, 3):
                self.output.slha["MSOFT", s_type.extpar + gen] = sqrt(
                    self.output.get_float_assert(s_type.slha2_output, (gen, gen))
                )
            self.output.remove_block(s_type.slha2_output)

        # A-terms and yukawas: only (3,3) elements are allowed
        for t_type in A:
            a33 = self.output.get_float_assert(
                t_type.out_t, (3, 3)
            ) / self.output.get_float_assert(t_type.out_y, (3, 3))
            self.output.slha[t_type.out_a, 3, 3] = a33
            self.output.remove_block(t_type.out_t)
            for i in (1, 2):
                self.output.remove_value(t_type.out_y, (i, i))

        # mass and mixing: quark flavor rotation is reverted.
        pid_base = (1000001, 1000003, 1000005, 2000001, 2000003, 2000005)
        self._reorder_no_flv_mixing_matrix(
            "USQMIX", [pid + 1 for pid in pid_base], lighter_lr_mixing=False
        )
        self._reorder_no_flv_mixing_matrix(
            "DSQMIX", [pid for pid in pid_base], lighter_lr_mixing=False
        )
        self._reorder_no_flv_mixing_matrix(
            "SELMIX", [pid + 10 for pid in pid_base], lighter_lr_mixing=False
        )
        self._reorder_no_flv_mixing_matrix(
            "SNUMIX", [1000012, 1000014, 1000016], lighter_lr_mixing=False
        )

        for slha2, slha1 in (
            ("USQMIX", "STOPMIX"),
            ("DSQMIX", "SBOTMIX"),
            ("SELMIX", "STAUMIX"),
        ):
            r = self.output.get_matrix(slha2)
            assert r is not None
            self.output.slha[slha1, 1, 1] = r[2][2]
            self.output.slha[slha1, 1, 2] = r[2][5]
            self.output.slha[slha1, 2, 1] = r[5][2]
            self.output.slha[slha1, 2, 2] = r[5][5]
            self.output.remove_block(slha2)

        assert is_diagonal_matrix(self.output.get_matrix("SNUMIX"))
        self.output.remove_block("SNUMIX")

        # remove SLHA2 blocks
        for block_name in ("VCKM", "IMVCKM", "UPMNS", "IMUPMNS"):
            self.output.remove_block(block_name)

    def _reorder_no_flv_mixing_matrix(self, matrix_name, pids, lighter_lr_mixing=True):
        # type: (str, List[int], bool) -> None
        mixing = self.output.get_matrix(matrix_name)

        assert isinstance(mixing, np.ndarray)
        assert (
            len(mixing.shape) == 2
            and mixing.shape[0] == mixing.shape[1]
            and mixing.shape[0] in (3, 6)
        )

        def failed() -> None:
            raise RuntimeError(f"SLHA2 to SLHA1 conversion failed: {mixing}")

        threshold = 1e-10
        n = mixing.shape[0]
        order = [-1 for i in range(n)]
        for j in range(1, n + 1):
            states = list()
            for i in range(1, n + 1):
                if abs(mixing[i - 1][j - 1]) > threshold:
                    states.append(i)
            if not 1 <= len(states) <= n // 3:
                failed()
            if n == 6 and (j % 3 == 0 or lighter_lr_mixing):  # with left-right mixing
                if len(states) == 2:
                    if j < 4:
                        order[j - 1], order[j + 2] = states
                    else:
                        if not (
                            order[j - 4] == states[0] and order[j - 1] == states[1]
                        ):
                            failed()
                else:  # (1,0)(0,1) or (0,1)(1,0)
                    order[j - 1] = states[0]
                    if j > 3 and order[j - 4] > order[j - 1]:
                        order[j - 4], order[j - 1] = order[j - 1], order[j - 4]
            else:
                if len(states) == 1:
                    order[j - 1] = states[0]
                else:
                    if abs(mixing[states[0] - 1][j - 1]) < abs(
                        mixing[states[1] - 1][j - 1]
                    ):
                        major, minor = states[1], states[0]
                    else:
                        major, minor = states[0], states[1]
                    if abs(mixing[minor - 1][j - 1] > 0.01):
                        self.add_warning(
                            f"large mixing {mixing[minor-1][j-1]} found in {pids[minor-1]}"
                        )
                    order[j - 1] = major
                for i in range(1, n + 1):
                    mixing[i - 1][j - 1] = (
                        1 if i == order[j - 1] else 0
                    )  # to chop the left-right mixing

        masses = [self.output.mass(p) or float("nan") for p in pids]
        new_mixing = mixing.copy()
        for i in range(n):
            new_mixing[i] = mixing[order[i] - 1]
            self.output.set_mass(pids[i], masses[order[i] - 1])
        self.output.set_matrix(matrix_name, new_mixing)
