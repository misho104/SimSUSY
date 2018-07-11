import simsusy.simsusy
from simsusy.abs_model import Info
from simsusy.abs_calculator import AbsCalculator
from simsusy.mssm.model import MSSMModel as Output
from simsusy.mssm.input import MSSMInput as Input
from simsusy.mssm.input import S, A, SLHAVersion
from simsusy.mssm.abstract import AbsEWSBParameters, AbsSMParameters
from simsusy.utility import tan2costwo, tan2sintwo, tan2tantwo, tan2cos, tan2sin, sin2cos
from simsusy.utility import autonne_takagi, singular_value_decomposition, mass_diagonalization
import math
import numpy as np
from typing import List, Optional  # noqa: F401


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
        """Return sin^2(theta_w)*cos^2(theta_w)"""
        return math.pi / ((2**0.5) * self._alpha_em_inv * self._g_fermi * self._mz**2)

    def sin_w_sq(self) -> float:
        r = self._sin_sq_cos_sq()
        return 2 * r / (1 + (1 - 4 * r)**.5)

    def cos_w_sq(self) -> float:
        r = self._sin_sq_cos_sq()
        return (1 + (1 - 4 * r)**.5) / 2

    def mz(self) -> float: return self._mz

    def mw(self) -> float: return self.mz() * (self.cos_w_sq()**.5)

    def gw(self) -> float: return (4 * math.pi / self._alpha_em_inv / self.sin_w_sq())**.5

    def gy(self) -> float: return (4 * math.pi / self._alpha_em_inv / self.cos_w_sq())**.5

    def gs(self) -> float: return (4 * math.pi * self._alpha_s)**.5

    def vev(self) ->float:
        return 2 * self.mz() / (self.gy()**2 + self.gw()**2)**0.5  # SLHA Last para of Sec.2.2

    def mass(self, pid)->Optional[float]:
        value = super().mass(pid)
        if value is not NotImplemented:
            return value
        elif pid == 1:
            return self._md_2gev or 0.  # TODO: improve
        elif pid == 2:
            return self._mu_2gev or 0.  # TODO: improve
        elif pid == 3:
            return self._ms_2gev or 0.1  # TODO: improve
        elif pid == 4:
            return self._mc_mc or 1.3  # TODO: improve
        elif pid == 5:
            return self._mb_mb   # identify msbar mass as pole mass
        elif pid == 24:
            return self.mw()  # TODO: improve
        else:
            return NotImplemented


class EWSBParameters(AbsEWSBParameters):
    def __init__(self, input, sm):  # type: (Input, SMParameters) -> None
        super().__init__(input)
        self.sm = sm
        self.calculate()
        assert self.is_set()

    def calculate(self)->None:
        # see doc/convention.pdf for equations and discussion.
        assert isinstance(self.tan_beta, float)
        cos_2beta = tan2costwo(self.tan_beta)
        sin_2beta = tan2sintwo(self.tan_beta)
        mz_sq = self.sm.mz() ** 2
        mw_sq = self.sm.mw() ** 2

        # first calculate mu if absent.
        if self.mu is None:
            assert isinstance(self.mh1_sq, float)
            assert isinstance(self.mh2_sq, float)
            # Martin (8.1.11)
            mu_sq = 0.5 * (- self.mh2_sq - self.mh1_sq - mz_sq
                           + abs((self.mh1_sq - self.mh2_sq) / cos_2beta))
            if self.sign_mu is None:
                raise ValueError('Block MINPAR 4 is required.')
            if mu_sq < 0:
                raise ValueError('Failed to get EWSB: mu^2 < 0')
            self.mu = self.sign_mu * (mu_sq ** 0.5)

            # GH (3.22) or Martin (8.1.10)
            self.ma_sq = self.mh1_sq + self.mh2_sq + 2. * mu_sq
            if self.ma_sq < 0:
                raise ValueError('Failed to get EWSB: mA^2 < 0')
        else:
            # set mA^2.
            if self.ma0 is not None:
                self.ma_sq = self.ma0 ** 2.  # at the tree level
            elif self.mhc is not None:
                self.ma_sq = self.mhc ** 2. - mw_sq  # GH (3.17)

        # now (mu, ma_sq) are set and (mh1_sq, mh2_sq) may be set.
        assert isinstance(self.ma_sq, float)
        if self.mhc is None:
            self.mhc = (self.ma_sq + mw_sq) ** 0.5
        if self.ma0 is None:
            self.ma0 = self.ma_sq ** 0.5

        # finally calculate h1_sq and mh2_sq if not set.
        if self.mh1_sq is None:
            m3_sq = self.ma_sq * sin_2beta / 2
            self.mh1_sq = -(self.mu**2) - (mz_sq * cos_2beta / 2) + m3_sq * self.tan_beta
            self.mh2_sq = -(self.mu**2) + (mz_sq * cos_2beta / 2) + m3_sq / self.tan_beta

    def alpha(self)->float:
        assert isinstance(self.tan_beta, float)
        tan_twobeta, ma, mz = tan2tantwo(self.tan_beta), self.ma0, self.sm.mz()
        assert isinstance(ma, float) and isinstance(mz, float)
        return 0.5 * math.atan((ma**2 + mz**2) / (ma + mz) / (ma - mz) * tan_twobeta)

    def yu(self)->List[float]:
        """Returns the diagonal elements of the Yukawa matrix (after super-CKM
        rotation)"""
        return [(2**0.5) * mass / self.sm.vev() / tan2sin(self.tan_beta) for mass in self.sm.mass_u()]

    def yd(self)->List[float]:
        return [(2**0.5) * mass / self.sm.vev() / tan2cos(self.tan_beta) for mass in self.sm.mass_d()]

    def ye(self)->List[float]:
        return [(2**0.5) * mass / self.sm.vev() / tan2cos(self.tan_beta) for mass in self.sm.mass_e()]

    def yukawa(self, species: A)->List[float]:
        if species == A.U:
            return self.yu()
        elif species == A.D:
            return self.yd()
        elif species == A.E:
            return self.ye()
        else:
            raise RuntimeError('invalid call of ewsb.yukawa')

    def mass(self, pid: int)->float:
        sm_value = self.sm.mass(pid)
        if isinstance(sm_value, float):
            return sm_value
        elif pid == 25 or pid == 35:  # Martin Eq.8.1.20 (see convention.pdf)
            sin_two_beta = tan2sintwo(self.tan_beta)
            a2, z2 = self.ma_sq, self.sm.mz()**2
            factor = -1 if pid == 25 else 1
            return ((a2 + z2 + factor * ((a2 - z2)**2 + 4 * z2 * a2 * sin_two_beta)**0.5) / 2)**0.5
        elif pid == 36:
            return self.ma0  # tree-level mass
        elif pid == 37:
            return self.mhc  # tree-level mass
        else:
            return NotImplemented


class Calculator(AbsCalculator):
    name = simsusy.simsusy.__pkgname__ + '/MSSMTree'
    version = simsusy.simsusy.__version__

    def __init__(self, input: Input)->None:
        self.input = input       # type: Input
        self.output = Output()   # type: Output
        # super().__init__(input=input)

    def write_output(self, filename: Optional[str]=None)->None:
        self.output.write(filename)

    def calculate(self):
        self.output.input = self.input
        self.output.spinfo = Info(self.name, self.version)
        self.output.sm = SMParameters(self.input)
        self.output.ewsb = EWSBParameters(self.input, self.output.sm)
        self._calculate_softmasses()
        self._calculate_higgses()
        self._calculate_neutralino()
        self._calculate_chargino()
        self._calculate_gluino()
        self._calculate_sfermion()

    def _calculate_softmasses(self):
        for i in [1, 2, 3]:
            self.output.set('MSOFT', i, self.input.mg(i))
        self.output.set('MSOFT', 21, self.output.ewsb.mh1_sq)
        self.output.set('MSOFT', 22, self.output.ewsb.mh2_sq)

        def assert_diagonal_matrix(matrix: np.ndarray, size: int):
            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == (size, size)
            for i in range(size):
                for j in range(size):
                    assert i == j or matrix[i, j] == 0

        if self.input.version == SLHAVersion.SLHA1:
            for species in [A.U, A.D, A.E]:
                matrix = self.input.a(species)
                assert_diagonal_matrix(matrix, 3)
                self.output.set(species.out_a, (3, 3), matrix[2, 2])
            for species in [S.QL, S.UR, S.DR, S.LL, S.ER]:
                m_sq = self.input.ms2(species)
                assert_diagonal_matrix(m_sq, 3)
                for gen in [1, 2, 3]:
                    self.output.set('MSOFT', species.extpar + gen, m_sq[gen - 1, gen - 1] ** 0.5)
            self.output.set('YU', (3, 3), self.output.ewsb.yu()[2])
            self.output.set('YD', (3, 3), self.output.ewsb.yd()[2])
            self.output.set('YE', (3, 3), self.output.ewsb.ye()[2])
        else:
            self.output.set_matrix('VCKM', self.input.vckm())
            self.output.set_matrix('IMVCKM', np.zeros((3, 3)))  # CPV ignored
            self.output.set_matrix('UPMNS', self.input.upmns())
            self.output.set_matrix('IMUPMNS', np.zeros((3, 3)))  # CPV ignored
            for species in [A.U, A.D, A.E]:
                y = np.diag(self.output.ewsb.yukawa(species))  # diagonal in super-CKM basis
                self.output.set_matrix(species.out_y, y, diagonal_only=True)

                a = self.input.a(species)
                if a is None:
                    t = self.input.t(species)
                else:
                    t = y * a
                self.output.set_matrix(species.out_t, t)
            for species in [S.QL, S.UR, S.DR, S.LL, S.ER]:
                self.output.set_matrix(species.slha2_output, self.input.ms2(species))

    def _calculate_higgses(self):
        assert self.output.ewsb.is_set()
        for pid in [25, 35, 36, 37]:
            self.output.set_mass(pid, self.output.ewsb.mass(pid))

    def _calculate_neutralino(self):
        assert self.output.sm is not None
        assert self.output.ewsb.is_set()
        cb = tan2cos(self.output.ewsb.tan_beta)
        sb = tan2sin(self.output.ewsb.tan_beta)
        sw = self.output.sm.sin_w_sq() ** 0.5
        cw = sin2cos(sw)
        mz = self.output.sm.mz()
        m1 = self.output.get_float('MSOFT', 1)
        m2 = self.output.get_float('MSOFT', 2)
        mu = self.output.ewsb.mu

        # Since CPV is not supported, m_psi0 is real and nmix will be real.
        m_psi0 = np.array([
            [m1, 0, -mz * cb * sw, mz * sb * sw],
            [0, m2, mz * cb * cw, -mz * sb * cw],
            [-mz * cb * sw, mz * cb * cw, 0, -mu],
            [mz * sb * sw, -mz * sb * cw, -mu, 0]])  # SLHA (21)
        masses, nmix = autonne_takagi(m_psi0, try_real_mixing=True)
        self.output.set_mass(1000022, masses[0])
        self.output.set_mass(1000023, masses[1])
        self.output.set_mass(1000025, masses[2])
        self.output.set_mass(1000035, masses[3])
        self.output.set_matrix('NMIX', nmix)

    def _calculate_chargino(self):
        assert self.output.sm is not None
        assert self.output.ewsb.is_set()
        cb = tan2cos(self.output.ewsb.tan_beta)
        sb = tan2sin(self.output.ewsb.tan_beta)
        sqrt2_mw = 2**0.5 * self.output.sm.mw()
        m2 = self.output.get_float('MSOFT', 2)
        mu = self.output.ewsb.mu

        m_psi_plus = np.array([
            [m2, sqrt2_mw * sb],
            [sqrt2_mw * cb, mu]])  # SLHA (22)
        masses, umix, vmix = singular_value_decomposition(m_psi_plus)
        self.output.set_mass(1000024, masses[0])
        self.output.set_mass(1000037, masses[1])
        self.output.set_matrix('UMIX', umix)
        self.output.set_matrix('VMIX', vmix)

    def _calculate_gluino(self):
        self.output.set_mass(1000021, self.output.get('MSOFT', 3))

    def _calculate_sfermion(self):
        if self.input.version == SLHAVersion.SLHA1:
            return self._calculate_sfermion_slha1()
        else:
            return self._calculate_sfermion_slha2()

    def _calculate_sfermion_slha1(self):
        mz2_cos2b = self.output.sm.mz()**2 * tan2costwo(self.output.ewsb.tan_beta)
        sw2 = self.output.sm.sin_w_sq()
        mu_tan_b = self.output.ewsb.mu * self.output.ewsb.tan_beta
        mu_cot_b = self.output.ewsb.mu / self.output.ewsb.tan_beta

        # SLHA Eq.23-25 and SUSY Primer (8.4.18)
        def mass_matrix(right: S, gen: int) -> np.ndarray:
            if right == S.UR:
                (left, a_species, mf) = (S.QL, A.U, self.output.sm.mass_u()[gen - 1])
                dl, dr, mu = (1 / 2 - 2 / 3 * sw2), 2 / 3 * sw2, mu_cot_b
            elif right == S.DR:
                (left, a_species, mf) = (S.QL, A.D, self.output.sm.mass_d()[gen - 1])
                dl, dr, mu = -(1 / 2 - 1 / 3 * sw2), -1 / 3 * sw2, mu_tan_b
            elif right == S.ER:
                (left, a_species, mf) = (S.LL, A.E, self.output.sm.mass_e()[gen - 1])
                dl, dr, mu = -(1 / 2 - sw2), -sw2, mu_tan_b
            else:
                raise NotImplementedError
            msl = self.output.get_float('MSOFT', left.extpar + gen)
            msr = self.output.get_float('MSOFT', right.extpar + gen)
            a = self.output.get(a_species.out_a, (gen, gen)) if gen == 3 else 0

            return np.array([
                [msl**2 + mf**2 + dl * mz2_cos2b, mf * (a - mu)],
                [mf * (a - mu), msr**2 + mf**2 + dr * mz2_cos2b]])

        def sneutrino_mass(gen: int)->float:
            msl = self.output.get_float('MSOFT', S.LL.extpar + gen)
            mf = self.output.sm.mass_n()[gen - 1]
            dl = 1 / 2
            return msl**2 + mf**2 + dl * mz2_cos2b

        for (right, pid, mix_name) in ((S.UR, 6, 'STOPMIX'), (S.DR, 5, 'SBOTMIX'), (S.ER, 15, 'STAUMIX')):
            mass_sq, f = mass_diagonalization(mass_matrix(right, 3))
            self.output.set_mass(1000000 + pid, mass_sq[0]**0.5)
            self.output.set_mass(2000000 + pid, mass_sq[1]**0.5)
            self.output.set_matrix(mix_name, f)

        for gen in [1, 2]:
            for (right, pid) in ((S.UR, 2), (S.DR, 1), (S.ER, 11)):
                mass_sq = mass_matrix(right, gen)
                self.output.set_mass(1000000 + pid + 2 * (gen - 1), mass_sq[0, 0]**0.5)
                self.output.set_mass(2000000 + pid + 2 * (gen - 1), mass_sq[1, 1]**0.5)

        for gen in [1, 2, 3]:
            self.output.set_mass(1000010 + 2 * gen, sneutrino_mass(gen)**0.5)

    def _calculate_sfermion_slha2(self):
        mz2_cos2b = np.diag([1, 1, 1]) * self.output.sm.mz()**2 * tan2costwo(self.output.ewsb.tan_beta)
        sw2 = self.output.sm.sin_w_sq()
        mu_tan_b = self.output.ewsb.mu * self.output.ewsb.tan_beta
        mu_cot_b = self.output.ewsb.mu / self.output.ewsb.tan_beta
        ckm = self.output.get_matrix('VCKM')
        pmns = self.output.get_matrix('UPMNS')

        def dag(m: np.ndarray) -> np.ndarray:
            return np.conjugate(m.T)

        def m_join(m11: np.ndarray, m12: np.ndarray, m21: np.ndarray, m22: np.ndarray) -> np.ndarray:
            return np.vstack([np.hstack([m11, m12]), np.hstack([m21, m22])])

        def mass_matrix(right: S) -> np.ndarray:
            if right == S.UR:
                (left, a_species, mf) = (S.QL, A.U, self.output.sm.mass_u())
                dl, dr, mu = (1 / 2 - 2 / 3 * sw2), 2 / 3 * sw2, mu_cot_b
                vev = self.output.sm.vev() * tan2sin(self.output.ewsb.tan_beta)
            elif right == S.DR:
                (left, a_species, mf) = (S.QL, A.D, self.output.sm.mass_d())
                dl, dr, mu = -(1 / 2 - 1 / 3 * sw2), -1 / 3 * sw2, mu_tan_b
                vev = self.output.sm.vev() * tan2cos(self.output.ewsb.tan_beta)
            elif right == S.ER:
                (left, a_species, mf) = (S.LL, A.E, self.output.sm.mass_e())
                dl, dr, mu = -(1 / 2 - sw2), -sw2, mu_tan_b
                vev = self.output.sm.vev() * tan2cos(self.output.ewsb.tan_beta)
            else:
                raise NotImplementedError
            msl2 = self.output.get_matrix(left.slha2_output)
            if right == S.UR:
                msl2 = ckm @ msl2 @ dag(ckm)
            msr2 = self.output.get_matrix(right.slha2_output)
            mf = np.diag(mf)
            t = self.output.get_matrix(a_species.out_t)

            # SLHA Eq.23-25 and SUSY Primer (8.4.18)
            return m_join(
                msl2 + mf**2 + dl * mz2_cos2b, vev / (2**0.5) * dag(t) - mf * mu,
                vev / (2**0.5) * t - mf * np.conjugate(mu), msr2 + mf**2 + dr * mz2_cos2b
            )

        def sneutrino_mass()->np.ndarray:
            msl2 = self.output.get_matrix(S.LL.slha2_output)
            mf = self.output.sm.mass_n()
            dl = 1 / 2
            return pmns @ msl2 @ dag(pmns) + np.diag(mf)**2 + dl * mz2_cos2b

        def prettify_matrix(m: np.ndarray, threshold=1e-10)->np.ndarray:
            nx, ny = m.shape
            for i in range(nx):
                m[i] = m[i] * (1 if max(m[i], key=lambda v: abs(v)) > 0 else -1)
                for j in range(ny):
                    if abs(m[i, j]) < threshold:
                        m[i, j] = 0
            return m

        for (right, pid, mix_name) in ((S.UR, 1, 'USQMIX'), (S.DR, 0, 'DSQMIX'), (S.ER, 10, 'SELMIX')):
            mass_sq, f = mass_diagonalization(mass_matrix(right))
            self.output.set_mass(1000001 + pid, mass_sq[0]**0.5)
            self.output.set_mass(1000003 + pid, mass_sq[1]**0.5)
            self.output.set_mass(1000005 + pid, mass_sq[2]**0.5)
            self.output.set_mass(2000001 + pid, mass_sq[3]**0.5)
            self.output.set_mass(2000003 + pid, mass_sq[4]**0.5)
            self.output.set_mass(2000005 + pid, mass_sq[5]**0.5)
            self.output.set_matrix(mix_name, prettify_matrix(f))

        mass_sq, f = mass_diagonalization(sneutrino_mass())
        self.output.set_mass(1000012, mass_sq[0]**0.5)
        self.output.set_mass(1000014, mass_sq[1]**0.5)
        self.output.set_mass(1000016, mass_sq[2]**0.5)
        self.output.set_matrix('SNUMIX', prettify_matrix(f))
