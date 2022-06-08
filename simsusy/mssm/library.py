import enum


class CPV(enum.Enum):
    NONE = 0
    CKM_ONLY = 1
    FULL = 2


class FLV(enum.Enum):
    NONE = 0
    QUARK = 1
    LEPTON = 2
    FULL = 3

    def qfv(self) -> bool:
        return bool(self.value & 1)

    def lfv(self) -> bool:
        return bool(self.value & 2)


class S(enum.Enum):
    """Object to represent the SLHA sfermion codes.

    obj[0] : the numbers correspond to EXTPAR/MSOFT number minus 1-3.
    obj[1] : SLHA2 input block name
    """

    QL = (40, "MSQ2IN")
    UR = (43, "MSU2IN")
    DR = (46, "MSD2IN")
    LL = (30, "MSL2IN")
    ER = (33, "MSE2IN")

    def __init__(self, extpar, slha2_input):
        # type: (int, str) -> None
        self.extpar = extpar
        self.slha2_input = slha2_input
        self.slha2_output = slha2_input[0:4]


class A(enum.Enum):
    """Object to represent the SLHA A-term codes.

    obj[0] : the numbers correspond to EXTPAR number.
    obj[1] : SLHA2 input block name.
    """

    U = (11, "TUIN", "AU", "TU", "YU")
    D = (12, "TDIN", "AD", "TD", "YD")
    E = (13, "TEIN", "AE", "TE", "YE")

    def __init__(self, extpar, slha2_input, out_a, out_t, out_y):
        # type: (int, str, str, str, str) -> None
        self.extpar = extpar  # type: int
        self.slha2_input = slha2_input  # type: str
        self.out_a = out_a  # type: str
        self.out_t = out_t  # type: str
        self.out_y = out_y  # type: str
