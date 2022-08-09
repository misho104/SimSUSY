"""MSSM library."""
import enum
from typing import Sequence


class CPV(enum.Enum):
    """MSSM choice of CP-violation."""

    NONE = 0
    CKM_ONLY = 1
    FULL = 2


class FLV(enum.Enum):
    """MSSM choice of flavor-violation."""

    NONE = 0
    QUARK = 1  # only in (s)quark sector
    LEPTON = 2  # only in (s)lepton sector
    FULL = 3

    def qfv(self) -> bool:
        """Return if (s)quark flavor-violation is enabled."""
        return bool(self.value & 1)

    def lfv(self) -> bool:
        """Return if (s)lepton flavor-violation is enabled."""
        return bool(self.value & 2)


class S(enum.Enum):
    """
    Object to represent the SLHA sfermion codes.

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
    """
    Object to represent the SLHA A-term codes.

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


class SMIX(enum.Enum):
    """
    Object to represent the sfermion mixings.

    obj[0] : the SLHA2 mixing block
    obj[1] : the SLHA1 mixing block for third generation
    obj[2] : pids
    """

    D = ("DSQMIX", "SBOTMIX", [1000001, 1000003, 1000005, 2000001, 2000003, 2000005])
    U = ("USQMIX", "STOPMIX", [1000002, 1000004, 1000006, 2000002, 2000004, 2000006])
    L = ("SELMIX", "STAUMIX", [1000011, 1000013, 1000015, 2000011, 2000013, 2000015])
    N = ("SNUMIX", "", [1000012, 1000014, 1000016])

    def __init__(self, slha2mix, slha1mix, pids):
        # type: (str, str, Sequence[int]) -> None
        self.slha2mix = slha2mix
        self.slha1mix = slha1mix
        self.pids = pids
