from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from simsusy.mssm.input import MSSMInput  # noqa: F401


class EWSBParameters():
    def __init__(self, model):  # type: (MSSMInput) -> None
        self.mh1_sq = model.get('EXTPAR', 21)    # type: Optional[float]
        self.mh2_sq = model.get('EXTPAR', 22)    # type: Optional[float]
        self.mu = model.get('EXTPAR', 23)     # type: Optional[float]
        self.ma_sq = model.get('EXTPAR', 24)   # type: Optional[float]
        self.ma0 = model.get('EXTPAR', 26)    # type: Optional[float]
        self.mhc = model.get('EXTPAR', 27)    # type: Optional[float]

        unspecified_param_count = self._tuple().count(None)
        if unspecified_param_count > 4:
            self.mh1_sq = self.mh1_sq or model.get('MINPAR', 1) ** 2
            self.mh2_sq = self.mh2_sq or model.get('MINPAR', 1) ** 2
        self.validate()

    def _tuple(self)->Tuple[Optional[float], ...]:
        return (self.mh1_sq, self.mh2_sq, self.mu, self.ma_sq, self.ma0, self.mhc)

    def validate(self):
        unspecified_param_count = self._tuple().count(None)
        if unspecified_param_count == 4:  # two must be specified
            if self.mh1_sq is not None or self.mh2_sq is not None:
                return True
            elif self.mu is not None:
                if self.ma_sq is not None or self.ma0 is not None or self.mhc is not None:
                    return True
        raise ValueError('invalid specification of EWSB parameters')
