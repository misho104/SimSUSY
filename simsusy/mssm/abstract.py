import logging
import json
import pathlib
from typing import TYPE_CHECKING, List, Optional  # noqa: F401
if TYPE_CHECKING:
    from simsusy.mssm.input import MSSMInput  # noqa: F401

logger = logging.getLogger(__name__)


class AbsSMParameters():
    DEFAULT_DATA = pathlib.Path(__file__).parent.parent.resolve() / 'default_values.json'

    def __init__(self, input):  # type: (MSSMInput) -> None
        with open(self.DEFAULT_DATA) as f:
            self.default_values = json.load(f)

        def get(key: int, default_key: str)->float:
            value = input.sminputs(key)
            if isinstance(value, float):
                return value
            if key <= 7:
                logger.warning(f'Block SMINPUTS {key} is not specified; default value is used.')
            else:
                logger.debug(f'Block SMINPUTS {key} is not specified; default value is used.')

            default = self.default_values.get(default_key)
            if isinstance(default, dict):
                value = default.get('value')
                if isinstance(value, float):
                    return value
            else:
                raise RuntimeError(f'Invalid parameter {default_key} in {self.DEFAULT_DATA}, which must be float.')

        self._alpha_em_inv = get(1, 'alpha_EW_inverse@m_Z')  # MS-bar, with 5 active flavors
        self._g_fermi = get(2, 'G_F')
        self._alpha_s = get(3, 'alpha_s@m_Z')                # MS-bar, with 5 active flavors
        self._mz = get(4, 'm_Z')                             # pole
        self._mb_mb = get(5, 'm_b@m_b')                      # MS-bar, at mb
        self._mt = get(6, 'm_t')                             # pole
        self._mtau = get(7, 'm_tau')                         # pole

        self._mnu1 = get(12, 'm_nu1')                        # pole
        self._mnu2 = get(14, 'm_nu2')                        # pole
        self._mnu3 = get(8, 'm_nu3')                         # pole
        self._me = get(11, 'm_e')                            # pole
        self._mmu = get(13, 'm_mu')                          # pole
        self._md_2gev = get(21, 'm_d@2GeV')                  # MS-bar, at 2GeV
        self._mu_2gev = get(22, 'm_u@2GeV')                  # MS-bar, at 2GeV
        self._ms_2gev = get(23, 'm_s@2GeV')                  # MS-bar, at 2GeV
        self._mc_mc = get(24, 'm_c@m_c')                     # MS-bar, at mc

    def sin_w_sq(self) -> float: return NotImplemented

    def cos_w_sq(self) -> float: return NotImplemented

    def mz(self) -> float: return NotImplemented

    def mw(self) -> float: return NotImplemented

    def gw(self) -> float: return NotImplemented

    def gy(self) -> float: return NotImplemented

    def gs(self) -> float: return NotImplemented

    def vev(self)->float: return NotImplemented

    def mass(self, pid) ->float:  # pole mass
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

    def mass_u(self)->List[float]:
        return [self.mass(i) for i in (2, 4, 6)]

    def mass_d(self)->List[float]:
        return [self.mass(i) for i in (1, 3, 5)]

    def mass_e(self)->List[float]:
        return [self.mass(i) for i in(11, 13, 15)]

    def mass_n(self)->List[float]:
        return [self.mass(i) for i in(12, 14, 16)]


class AbsEWSBParameters():
    def __init__(self, model):  # type: (MSSMInput) -> None
        self.mh1_sq = model.get_float('EXTPAR', 21)  # type: Optional[float]
        self.mh2_sq = model.get_float('EXTPAR', 22)  # type: Optional[float]
        self.mu = model.get_float('EXTPAR', 23)      # type: Optional[float]
        self.ma_sq = model.get_float('EXTPAR', 24)   # type: Optional[float]
        self.ma0 = model.get_float('EXTPAR', 26)     # type: Optional[float]
        self.mhc = model.get_float('EXTPAR', 27)     # type: Optional[float]

        self.tan_beta = model.get_float('EXTPAR', 25) or model.get_float('MINPAR', 3)  # type: Optional[float]
        if self.tan_beta is None:
            raise ValueError('tanbeta not specified.')

        sign_mu = model.get('MINPAR', 4)
        if not(sign_mu is None or 0.9 < abs(sign_mu) < 1.1):
            raise ValueError(f'Invalid EXTPAR 4; either 1 or -1.')
        self.sign_mu = (1 if sign_mu > 0 else -1) if sign_mu is not None else None  # type: Optional[int]

        unspecified_param_count = self._count_unspecified_params()
        if unspecified_param_count > 4:
            self.mh1_sq = self.mh1_sq or model.get('MINPAR', 1) ** 2
            self.mh2_sq = self.mh2_sq or model.get('MINPAR', 1) ** 2
        self.validate()

    def _count_unspecified_params(self)->int:
        return [self.mh1_sq, self.mh2_sq, self.mu, self.ma_sq, self.ma0, self.mhc].count(None)

    def validate(self):
        unspecified_param_count = self._count_unspecified_params()
        if unspecified_param_count == 0:
            return True
        elif unspecified_param_count == 4:  # two must be specified
            if self.mh1_sq is not None or self.mh2_sq is not None:
                return True
            elif self.mu is not None:
                if self.ma_sq is not None or self.ma0 is not None or self.mhc is not None:
                    return True
        raise ValueError('invalid specification of EWSB parameters')

    def is_set(self)->bool:
        return self._count_unspecified_params() == 0

    def alpha(self)->float: return NotImplemented
