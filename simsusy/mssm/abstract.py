from typing import Tuple, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from simsusy.mssm.input import MSSMInput  # noqa: F401


class AbsSMParameters():
    def __init__(self, input):  # type: (MSSMInput) -> None
        def get(key: int)->float:
            value = input.sminputs(key)
            if value is None:
                raise ValueError(f'Block SMINPUTS {key} is not specified.')
            return value

        self._alpha_em_inv = get(1)  # MS-bar, at mZ, with 5 active flavors
        self._g_fermi = get(2)
        self._alpha_s = get(3)       # MS-bar, at mZ, with 5 active flavors
        self._mz = get(4)            # pole
        self._mb_mb = get(5)         # MS-bar, at mb
        self._mt = get(6)            # pole
        self._mtau = get(7)          # pole

        self._mnu1 = input.get_float('SMINPUTS', 12)  # pole
        self._mnu2 = input.get_float('SMINPUTS', 14)  # pole
        self._mnu3 = input.get_float('SMINPUTS', 8)   # pole
        self._me = input.get_float('SMINPUTS', 11)    # pole
        self._mmu = input.get_float('SMINPUTS', 13)   # pole
        self._md_2gev = input.get_float('SMINPUTS', 21)  # MS-bar, at 2GeV
        self._mu_2gev = input.get_float('SMINPUTS', 22)  # MS-bar, at 2GeV
        self._ms_2gev = input.get_float('SMINPUTS', 23)  # MS-bar, at 2GeV
        self._mc_mc = input.get_float('SMINPUTS', 24)    # MS-bar, at mc

    def sin_w_sq(self) -> float: return NotImplemented

    def cos_w_sq(self) -> float: return NotImplemented

    def mz(self) -> float: return NotImplemented

    def mw(self) -> float: return NotImplemented

    def gw(self) -> float: return NotImplemented

    def gy(self) -> float: return NotImplemented

    def gs(self) -> float: return NotImplemented

    def mass(self, pid) -> Optional[float]:  # pole mass
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
