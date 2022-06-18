"""MSSM tree-level calculator with tan-beta resummation."""


import logging
from math import log, pi, sqrt
from typing import Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing
import yaslha

import simsusy.mssm.tree_calculator
import simsusy.simsusy
from simsusy.mssm.input import MSSMInput as Input  # noqa: F401
from simsusy.mssm.model import MSSMModel as Output  # noqa: F401
from simsusy.utility import (
    autonne_takagi,
    is_real_matrix,
    mass_diagonalization,
    tan2cos,
    tan2costwo,
)

ComplexMatrix = numpy.typing.NDArray[np.complex_]
RealMatrix = numpy.typing.NDArray[np.float_]
Matrix = numpy.typing.NDArray[Union[np.float_, np.complex_]]
logger = logging.getLogger(__name__)
T = TypeVar("T", RealMatrix, ComplexMatrix)


def pow2(x: float) -> float:
    """Square the value."""
    return x * x


class SleptonProperty:
    """Slepton property in flavor order."""

    def __init__(self, mass, snu, mixing):
        # type: (Tuple[float, float], float, ComplexMatrix) -> None
        """Initialize.

        The i-th row of mixing corresponds to the particle with i-th mass, while the
        first (second) column corresponds to slep_L (slep_R).
        """

        self.mass = mass
        self.snu = snu
        self.mixing = mixing

    def angle_sin(self):
        # type: () -> complex
        """Return sine of the mixing angle."""
        return complex(self.mixing[0, 1])

    def mixing_magnitude(self):
        # type: () -> float
        """Return the magnitude of the mixing."""
        return float(min(abs(self.mixing[0, 0]), abs(self.mixing[0, 1])))

    def __repr__(self) -> str:
        return f"m={self.mass}, mix={self.mixing}".replace("\n", "")


class Arxiv08081530:
    @staticmethod
    def __is_close(*v: float) -> bool:
        return abs(1 - min(v) / max(v)) < 1e-4

    @staticmethod
    def i_abc(a: float, b: float, c: float) -> float:
        """Special function Eq. (7) of arXiv:0808.1530."""
        return Arxiv08081530.i2abc(a * a, b * b, c * c)

    @staticmethod
    def i2abc(a: float, b: float, c: float) -> float:
        """Special function Eq. (7) (squared version) of arXiv:0808.1530."""
        if Arxiv08081530.__is_close(a, b, c):
            return Arxiv08081530.__i2aaa(a, b, c)
        elif Arxiv08081530.__is_close(a, b):
            return Arxiv08081530.__i2aac(a, b, c)
        elif Arxiv08081530.__is_close(a, c):
            return Arxiv08081530.__i2aac(a, c, b)
        elif Arxiv08081530.__is_close(b, c):
            return Arxiv08081530.__i2aac(b, c, a)
        else:
            return (a * b * log(a / b) + b * c * log(b / c) + c * a * log(c / a)) / (
                (a - b) * (b - c) * (a - c)
            )

    @staticmethod
    def __i2aaa(a: float, b0: float, c0: float) -> float:
        b, c = b0 - a, c0 - a
        return (
            0.5 / a - (b + c) / (6 * a * a) + (b * b + b * c + c * c) / (12 * a * a * a)
        )

    @staticmethod
    def __i2aac(a: float, b: float, c: float) -> float:
        s, d, r = (a + b) / 2, (a - b) / 2, (a + b) / 2 - c
        r2 = r * r
        return (
            1 / r
            + pow2(d / s) * (c * c / r2 + 1.5 * c / r + 1 / 3) / r
            + c / r2 * (1 + d * d / r2) * log(c / s)
        )

    @staticmethod
    def calc_m1sq_m2sq(m2: float, mu: float, mw: float) -> Tuple[float, float]:
        """Return M1^2 and M2^2 of Eq. (9) of arXiv:0808.1530."""
        product = m2 * m2 * mu * mu  # this is m1^2 * m2^2
        tmp = m2 * m2 + mu * mu + 2 * mw * mw
        m2sq = 0.5 * (tmp + sqrt(tmp * tmp - 4 * product))
        m1sq = product / m2sq
        return m1sq, m2sq

    @staticmethod
    def b0(m1sq: float, m2sq: float, scale_sq: float, delta: float = 0.0) -> float:
        """Return Passarino-Veltman B0 of arXiv:0808.1530."""
        return (
            (1 + delta)
            - m1sq / (m1sq - m2sq) * log(m1sq / scale_sq)
            - m2sq / (m2sq - m1sq) * log(m2sq / scale_sq)
        )


class Calculator(simsusy.mssm.tree_calculator.Calculator):
    """MSSM tree-level calculator with tan-beta resummation."""

    name = simsusy.simsusy.__pkgname__ + "/MSSMTbResum"
    version = simsusy.simsusy.__version__

    def __init__(self, mssm_input):
        # type: (Input)->None
        """Initialize calculator."""
        super().__init__(input=mssm_input)

    def __get_slepton_property_each(self):
        # type: ()->Tuple[SleptonProperty, SleptonProperty, SleptonProperty]
        """Return slepton property for each flavor."""
        pids = [1000011, 1000013, 1000015, 2000011, 2000013, 2000015]
        order = self._get_mixing_order_flavor("SELMIX", pids, (True, True, True))
        masses = [self.output.mass_assert(pids[order[i]]) for i in range(6)]
        mixing = self.output.get_complex_matrix_assert("SELMIX")[order]

        snu_pids = [1000012, 1000014, 1000016]
        snu_order = self._get_mixing_order_flavor("SNUMIX", snu_pids)
        snu_masses = [self.output.mass_assert(snu_pids[snu_order[i]]) for i in range(3)]
        return (
            SleptonProperty((masses[0], masses[3]), snu_masses[0], mixing[0::3, 0::3]),
            SleptonProperty((masses[1], masses[4]), snu_masses[1], mixing[1::3, 1::3]),
            SleptonProperty((masses[2], masses[5]), snu_masses[2], mixing[2::3, 2::3]),
        )

    def _calculate_higgses(self) -> None:
        super()._calculate_higgses()
        assert self.output.sm
        self.output.set_mass(25, self.output.sm.default_value("m_H"))

    def __slepton_delta_l(self, large_tan_beta_limit=False, exclude_wino=False):
        # type: (bool, bool)->Tuple[float, float, float]
        assert self.output.sm is not None
        assert self.output.ewsb is not None
        mw = self.output.sm.mass(24)
        mu = self.output.get_float_assert("HMIX", 1)
        tb = self.output.get_float_assert("HMIX", 2)
        vv = self.output.get_float_assert("HMIX", 3)
        m1 = self.output.get_float_assert("MSOFT", 1)
        m2 = self.output.get_float_assert("MSOFT", 2)
        gysq = pow2(self.output.get_float_assert("GAUGE", 1))
        g2sq = pow2(self.output.get_float_assert("GAUGE", 2))
        scale_sq = [
            sqrt(
                self.output.get_float_assert("MSL2", f, f)
                * self.output.get_float_assert("MSE2", f, f)
            )
            for f in range(1, 4)
        ]

        m_leptons = self.output.sm.mass_e()
        vd = vv / sqrt(2) * tan2cos(tb)
        y0 = [m / vd for m in m_leptons]
        yl = [self.output.get_float_assert("YE", i, i) for i in [1, 2, 3]]

        if exclude_wino:
            g2sq = 0  # to exclude wino-contribution automatically
        sleptons = self.__get_slepton_property_each()
        delta_l = [0.0, 0.0, 0.0]
        if large_tan_beta_limit:
            m = [sqrt(r) for r in Arxiv08081530.calc_m1sq_m2sq(m2, mu, mw)]
            for i in range(3):
                if (mix := abs(sleptons[i].mixing[0, 1])) > 0.1:
                    self.add_warning(f"Ignored slepton{i+1} mixing: {mix:.2f}")
                msl, msr = sleptons[i].mass
                delta_l[i] = (-mu * tb / 16 / pi / pi) * (
                    (g2sq * m2) * Arxiv08081530.i_abc(m[0], m[1], sleptons[i].snu)
                    + (g2sq * m2 / 2) * Arxiv08081530.i_abc(m[0], m[1], msl)
                    + (gysq * m1) * Arxiv08081530.i_abc(mu, m1, msr)
                    - (gysq * m1 / 2) * Arxiv08081530.i_abc(mu, m1, msl)
                    - (gysq * m1) * Arxiv08081530.i_abc(m1, msl, msr)
                )
            return (delta_l[0], delta_l[1], delta_l[2])
        char = [self.output.mass_assert(p) for p in (1000024, 1000037)]
        umix = self.output.get_complex_matrix_assert("UMIX")
        vmix = self.output.get_complex_matrix_assert("VMIX")
        n_matrix = self._neutralino_matrix()
        if exclude_wino:
            n_matrix = np.delete(np.delete(n_matrix, 1, axis=0), 1, axis=1)
            neut, nmix = autonne_takagi(n_matrix, try_real_mixing=True)
            nmix = np.insert(nmix, 1, 0, axis=1)
            char = [mu]
            umix = vmix = np.array([[0, 1], [0, 0]], dtype=float)
        else:
            neut, nmix = autonne_takagi(n_matrix, try_real_mixing=True)

        def sigma_neut(f: int, i: int, m: int) -> float:
            nl = float(
                sqrt(gysq / 2) * nmix[i, 0] * sleptons[f].mixing[m, 0].conjugate()
                + sqrt(g2sq / 2) * nmix[i, 1] * sleptons[f].mixing[m, 0].conjugate()
                - yl[f] * nmix[i, 2] * sleptons[f].mixing[m, 1].conjugate()
            )
            nr = float(
                sqrt(2 * gysq) * nmix[i, 0] * sleptons[f].mixing[m, 1]
                + yl[f] * nmix[i, 2] * sleptons[f].mixing[m, 0]
            )
            m_neut = float(neut[i])  # mypy/numpy
            return (1 / (16 * pi * pi) * m_neut * (nl * nr).real) * Arxiv08081530.b0(
                pow2(m_neut), pow2(sleptons[f].mass[m]), scale_sq[f]
            )

        def sigma_char(f: int, i: int) -> float:
            cl, cr = float(sqrt(g2sq) * vmix[i, 0]), float(yl[f] * umix[i, 1])
            return (1 / (16 * pi * pi) * char[i] * (cl * cr).real) * Arxiv08081530.b0(
                pow2(char[i]), pow2(sleptons[f].snu), scale_sq[f]
            )

        for f in range(3):
            sigma_sum = sum(
                sigma_neut(f, i, m)
                for i in range(3 if exclude_wino else 4)
                for m in range(2)
            ) + sum(sigma_char(f, i) for i in range(1 if exclude_wino else 2))
            # original convention of 0910.2663
            # delta_l = sigma_sum / m_leptons[f]
            # Motoi's modification : delta equals to sigma(1+delta)/m
            # so that y vd == m - sigma = m / (1 + delta).
            # Here, (1+delta) = y0 / yl.
            delta_l[f] = sigma_sum * (y0[f] / yl[f]) / m_leptons[f]
        return (delta_l[0], delta_l[1], delta_l[2])

    def _calculate_sfermion(self) -> None:
        super()._calculate_sfermion()
        assert self.output.sm and self.output.ewsb
        mu = self.output.ewsb.mu
        tan_beta = self.output.ewsb.tan_beta
        assert isinstance(mu, float)
        assert isinstance(tan_beta, float)
        mz = self.output.sm.mz()
        vd = self.output.sm.vev() * tan2cos(tan_beta) / sqrt(2)
        mf_mat = np.diag(self.output.sm.mass_e())
        mz2_cos2b = np.diag([1, 1, 1]) * pow2(mz) * tan2costwo(tan_beta)
        sw2 = self.output.sm.sin_w_sq()
        mu_tan_b = mu * self.output.ewsb.tan_beta
        msl2 = self.output.get_matrix_assert("MSL2")
        msr2 = self.output.get_matrix_assert("MSE2")
        t = self.output.get_matrix_assert("TE")

        def m_join(m11: T, m12: T, m21: T, m22: T) -> T:
            return np.vstack([np.hstack([m11, m12]), np.hstack([m21, m22])])

        def update(delta_l):
            # type: (Tuple[float, float, float]) -> None
            # SLHA Eq.23-25 and SUSY Primer (8.4.18)
            # A-term is assumed unaffected, thus T is affected.
            d = np.diag([1 / (1 + d) for d in delta_l])
            matrix = m_join(
                msl2 + mf_mat * mf_mat + (-1 / 2 + sw2) * mz2_cos2b,
                vd * np.conjugate(t.T @ d) - mu_tan_b.real * mf_mat @ d,
                vd * t @ d - np.conjugate(mu_tan_b).real * mf_mat @ d,
                msr2 + mf_mat * mf_mat + (-sw2) * mz2_cos2b,
            )
            mass_sq, f = mass_diagonalization(matrix)
            self.output.set_mass(1000011, sqrt(mass_sq[0]))
            self.output.set_mass(1000013, sqrt(mass_sq[1]))
            self.output.set_mass(1000015, sqrt(mass_sq[2]))
            self.output.set_mass(2000011, sqrt(mass_sq[3]))
            self.output.set_mass(2000013, sqrt(mass_sq[4]))
            self.output.set_mass(2000015, sqrt(mass_sq[5]))
            assert is_real_matrix(f)
            self.output.set_matrix("SELMIX", f.real)
            for i, delta in enumerate(delta_l):
                self.output.slha["YE"].comment[i + 1, i + 1] = f"delta = {delta:.6f}"
                self.output.slha["YE", i + 1, i + 1] = mf_mat[i, i] / (1 + delta) / vd

        for i in range(4):
            delta_l = self.__slepton_delta_l(
                large_tan_beta_limit=False, exclude_wino=True
            )
            update(delta_l)
        self._chop_mixing_matrix("SELMIX")

    def write_output(self, filename=None, slha1=False):
        # type: (Optional[str], bool)->None
        """Write output to file."""

        # It seems that SLHA2 does not allow negative mass for neutralinos.
        # Then it is better to always provide IMNMIX.
        if not slha1:
            neut_masses = [
                (pid, self.output.mass_assert(pid))
                for pid in [1000022, 1000023, 1000025, 1000035]
            ]
            for i, (pid, mass) in enumerate(neut_masses):
                for j in range(4):
                    mix = self.output.get_complex_assert("NMIX", i + 1, j + 1)
                    if mass < 0:
                        self.output.set_mass(pid, abs(mass))
                        mix = mix / 1j
                    self.output.slha["NMIX", i + 1, j + 1] = mix.real
                    self.output.slha["IMNMIX", i + 1, j + 1] = mix.imag

        pid_base = [1000001, 1000003, 1000005, 2000001, 2000003, 2000005]
        pid_snu = [1000012, 1000014, 1000016]
        self._reorder_mixing_matrix_in_flavor("DSQMIX", pid_base)
        self._reorder_mixing_matrix_in_flavor("USQMIX", [p + 1 for p in pid_base])
        self._reorder_mixing_matrix_in_flavor("SELMIX", [p + 10 for p in pid_base])
        self._reorder_mixing_matrix_in_flavor("SNUMIX", pid_snu)
        self._chop_mixing_matrix("DSQMIX")
        self._chop_mixing_matrix("USQMIX")
        self._chop_mixing_matrix("SELMIX")
        self._chop_mixing_matrix("SNUMIX")

        # dumper configuration
        self.output.dumper = yaslha.dumper.SLHADumper(
            separate_blocks=True,
            comments_preserve=yaslha.dumper.CommentsPreserve.TAIL,
            document_blocks=[
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
            ],
        )
        # done
        super().write_output(filename, slha1)
