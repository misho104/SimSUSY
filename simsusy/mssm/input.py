from simsusy.abs_model import AbsModel, SLHAVersion
from simsusy.mssm.ewsb_parameters import EWSBParameters
from simsusy.utility import sin2cos
import logging
import enum
import numpy as np
import math
from typing import List, Optional, Sequence, Any  # noqa: F401
logger = logging.getLogger(__name__)


class S(enum.Enum):
    """Object to represent the SLHA sfermion codes.

    obj[0] : the numbers correspond to EXTPAR number minus 1-3.
    obj[1] : SLHA2 input block name
    """
    QL = (40, 'MSQ2IN')
    UR = (43, 'MSU2IN')
    DR = (46, 'MSD2IN')
    LL = (30, 'MSL2IN')
    ER = (33, 'MSE2IN')


class A(enum.Enum):
    """Object to represent the SLHA A-term codes.

    obj[0] : the numbers correspond to EXTPAR number.
    obj[1] : SLHA2 input block name.
    """
    U = (11, 'TUIN')
    D = (12, 'TDIN')
    E = (13, 'TEIN')


class MSSMInput(AbsModel):
    SLHA1_BLOCKS = ['MODSEL', 'SMINPUTS', 'MINPAR', 'EXTPAR']
    SLHA2_BLOCKS = SLHA1_BLOCKS + [
        'VCKMIN', 'UPMNSIN', 'TUIN', 'TDIN', 'TEIN',
        'MSQ2IN', 'MSU2IN', 'MSD2IN', 'MSL2IN', 'MSE2IN']

    def __init__(self, *args):
        super().__init__(*args)
        self.version = self.check_validity_and_guess_version()  # type: SLHAVersion

    def check_validity_and_guess_version(self)->SLHAVersion:
        version = SLHAVersion.SLHA1
        invalid = []  # type: List[Sequence[Any]]
        ignore = []   # type: List[Sequence[Any]]
        for block in self._slha.blocks:
            name = ''.join(block)
            if name in self.SLHA1_BLOCKS:
                pass
            elif name in self.SLHA2_BLOCKS:
                version = SLHAVersion.SLHA2
            else:
                logger.warning(f'Unknown block {name} is ignored.')
                continue

            content = self._slha.blocks[block]
            if name == 'MODSEL':
                for k, v in content.items():
                    if k == 1:
                        if v not in [0, 1]:  # accept only general MSSM or mSUGRA. (no distinction)
                            invalid.append((name, k, v, (0, 1)))
                    elif k in [3, 4, 5]:
                        if v != 0:  # MSSM, no RpV, no CPV.
                            invalid.append((name, k, v, 0))
                    elif k == 6:
                        if v == 0:
                            pass   # no flavor violation
                        elif v in [1, 2, 3]:
                            version = SLHAVersion.SLHA2  # with flavor violation
                        else:
                            invalid.append((name, k, v))
                    elif k == 11:
                        if v != 1:
                            ignore.append((name, k, v))
                    elif k == 12 or k == 21:  # defined in SLHA spec
                        ignore.append((name, k, v))
                    else:
                        ignore.append((name, k, v))
            elif name == 'SMINPUTS':
                for k, v in content.items():
                    if 1 <= k <= 7:
                        pass
                    elif k in [8, 11, 12, 13, 14, 21, 22, 23, 24]:
                        version = SLHAVersion.SLHA2
                    else:
                        ignore.append((name, k, v))
            elif name == 'MINPAR':
                for k, v in content.items():
                    if not 1 <= k <= 5:
                        ignore.append((name, k, v))
            elif name == 'EXTPAR':
                for k, v in content.items():
                    if k == 0:
                        # input scale is not supported
                        ignore.append((name, k, v))
                    elif not (k in [1, 2, 3, 11, 12, 13] or
                              21 <= k <= 27 or
                              31 <= k <= 36 or
                              41 <= k <= 49):
                        ignore.append((name, k, v))
                # Higgs parameters validity
                try:
                    EWSBParameters(self)
                except ValueError:
                    invalid.append((name, 'invalid EWSB parameter specification.'))
            elif name == 'QEXTPAR':
                # input scale as SLHA2 extension is not supported
                logger.warning(f'Block {name} is unsupported and ignored.')
            elif name == 'VCKMIN':
                flag = 0  # any better way?
                for k, v in content.items():
                    if 1 <= k <= 4:
                        flag |= 2 ** (k - 1)
                    else:
                        ignore.append((name, k, v))
                if flag & 7 != 7:  # CPV (VCKMIN 4) is not supported
                    invalid.append((name, 'parameter is missing'))
            elif name == 'UPMNSIN':
                flag = 0
                for k, v in content.items():
                    if not 1 <= k <= 6:
                        ignore.append((name, k, v))
                if flag & 7 != 7:  # CPV (UPMNSIN 4-6) is not supported
                    invalid.append((name, 'parameter is missing'))
            elif name in ['MSQ2IN', 'MSU2IN', 'MSD2IN', 'MSL2IN', 'MSE2IN']:
                for k, v in content.items():
                    if not (isinstance(k, tuple) and len(k) == 2):
                        invalid.append((name, k, v))
                    elif not (1 <= k[0] <= 3 and k[0] <= k[1] <= 3):  # upper triangle only
                        ignore.append((name, k, v))
            elif name in ['TUIN', 'TDIN', 'TEIN']:
                for k, v in content.items():
                    if not (isinstance(k, tuple) and len(k) == 2):
                        invalid.append((name, k, v))
                    elif not (1 <= k[0] <= 3 and 1 <= k[1] <= 3):
                        ignore.append((name, k, v))

        if invalid:
            for i in invalid:
                if len(i) == 4:
                    msg = f'Block {i[0]}: {i[1]} = {i[2]} is invalid; should be {i[3]}.'
                elif len(i) == 3:
                    msg = f'Block {i[0]}: {i[1]} = {i[2]} is invalid.'
                else:
                    msg = f'Block {i[0]}: {i[1]}'
                logger.error(msg)
            raise ValueError
        for i in ignore:
            logger.warning(f'Block {i[0]}: {i[1]} = {i[2]} is ignored.')

        # tan-beta
        if self.get('MINPAR', 3) and self.get('EXTPAR', 25):
            logger.warning(f'TanBeta in MINPAR is ignored due to EXTPAR-25.')

        return version

    """
    High-level APIs are defined below, so that self.get would be used in special cases.

    NOTE: By defining a class to describe a parameter with its scale,
    one can extend functions below to return a parameter with scale.
    """

    @staticmethod
    def __value_or_unspecified_error(value, param_name):
        if isinstance(value, np.ndarray):
            if not np.any(np.equal(value, None)):
                return value
        else:
            if value is not None:
                return value
        raise ValueError(f'{param_name} is not specified.')

    def modsel(self, key: int)->float:
        """Returns MODSEL block; note that MODSEL-6 is only relevant for tree-
        level spectrum calculation."""
        return self.get('MODSEL', key)

    def sminputs(self, key: int)->float:
        return self.get('SMINPUTS', key)

    def tan_beta(self) -> float:
        value = self.get('EXTPAR', 25) or self.get('MINPAR', 3)
        return self.__value_or_unspecified_error(value, 'tan_beta')

    def sign_mu(self) -> Optional[int]:
        value = self.get('MINPAR', 4)
        if value is None:
            return None
        elif 0.9 < abs(value) < 1.1:
            return 1 if value > 0 else -1
        raise ValueError(f'EXTPAR 4 should be 1 or -1.')

    def mg(self, key: int)->float:
        """Return gaugino mass; key should be 1-3 (but no validation)."""
        value = self.get('EXTPAR', key) or self.get('MINPAR', 2)
        return self.__value_or_unspecified_error(value, f'M_{key}')

    def ms(self, species: S)->np.ndarray:
        m0 = self.get('MINPAR', 1)
        value = np.diag([self.get('EXTPAR', species.value[0] + gen) or m0 for gen in [1, 2, 3]])

        slha2block = self.block(species.value[1])
        if slha2block:
            for ix in range(1, 4):
                for iy in range(ix, 4):
                    v = slha2block.get(ix, iy)
                    if v:
                        value[ix, iy] = value[iy, ix] = v

        return self.__value_or_unspecified_error(value, f'm_sfermion({species.name}) mass')

    def a(self, species: A) -> np.ndarray:
        """Return A-term matrix, but only if T-matrix is not specified in the
        input; otherwise return None, and one should read T-matrix."""
        a33 = self.get('EXTPAR', species.value[0]) or self.get('MINPAR', 5)

        slha2block = self.block(species.value[1])
        if slha2block:
            for k, v in slha2block.items():
                if v != 0:
                    return None  # because T-matrix is specified.

        return self.__value_or_unspecified_error(np.diag([0, 0, a33]), f'A({species.name})')

    def t(self, species: A) -> np.ndarray:
        """Return T-term matrix if T-matrix is specified; corresponding EXTPAR
        entry is ignored and thus (3,3) element must be always specified."""
        slha2block = self.block(species.value[1])
        if not slha2block:
            return None

        matrix = np.diag([0, 0, np.nan])
        for k, v in slha2block.items():
            (x, y) = k
            matrix[x - 1, y - 1] = v
        if math.isnan(matrix[2, 2]):
            ValueError(f'Block {species.value[1]} needs (3,3) element.')

        return matrix

    def ewsb_parameters(self) -> EWSBParameters:
        return EWSBParameters(self)

    def vckm(self) -> np.ndarray:
        lam = self.get('VCKMIN', 1)
        a = self.get('VCKMIN', 2)
        rho = self.get('VCKMIN', 3)
        if self.get('VCKMIN', 4):
            logger.warning('CPV is not supported and VCKMIN 4 is ignored.')

        s12 = lam
        s23 = a * lam**2
        c12 = sin2cos(s12)
        c23 = sin2cos(s23)
        s13 = s12 * s23 * c23 * rho / c12 / (1 - s23 * s23 * rho)
        c13 = sin2cos(s13)
        return np.array([
            [c12 * c13, s12 * c13, s13],
            [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
            [s12 * s23 - c12 * c23 * s13, -c12 * s23 - s12 * c23 * s13, c23 * c13]])

    def upmns(self)->np.ndarray:
        """return UPMNS matrix
        NOTE: SLHA2 convention uses theta-bars, while PDG2006 has only thetas.
              The difference should be ignored as it seems denoting MS-bar scheme.()
        """
        s12, s23, s13 = (math.sin(self.get('UPMNSIN', i)) for i in [1, 2, 3])
        c12, c23, c13 = (math.cos(self.get('UPMNSIN', i)) for i in [1, 2, 3])
        if self.get('UPMNSIN', 4) or self.get('UPMNSIN', 5) or self.get('UPMNSIN', 6):
            logger.warning('CPV is not supported and UPMNSIN 4-6 is ignored.')
        return np.array([
            [c12 * c13, s12 * c13, s13],
            [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
            [s12 * s23 - c12 * c23 * s13, -c12 * s23 - s12 * c23 * s13, c23 * c13]])
