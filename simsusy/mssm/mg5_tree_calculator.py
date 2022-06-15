import logging
from typing import Optional

import yaslha

import simsusy.mssm.tree_calculator
import simsusy.simsusy
from simsusy.mssm.input import MSSMInput as Input  # noqa: F401
from simsusy.mssm.library import CPV, FLV
from simsusy.mssm.model import MSSMModel as Output  # noqa: F401

logger = logging.getLogger(__name__)


class Calculator(simsusy.mssm.tree_calculator.Calculator):
    name = simsusy.simsusy.__pkgname__ + "/MSSMTree"
    version = simsusy.simsusy.__version__

    def __init__(self, input: Input) -> None:
        super().__init__(input=input)

    def write_output(self, filename=None, slha1=False):
        # type: (Optional[str], bool)->None
        # MSSM_SLHA2 does not accept SLHA2 format of sfermion mixing.
        pid_base = [1000001, 1000003, 1000005, 2000001, 2000003, 2000005]
        self._reorder_mixing_matrix_in_flavor(
            "USQMIX", [pid + 1 for pid in pid_base], lighter_gen_in_flavor=True
        )
        self._reorder_mixing_matrix_in_flavor(
            "DSQMIX", pid_base, lighter_gen_in_flavor=True
        )
        self._reorder_mixing_matrix_in_flavor(
            "SELMIX", [pid + 10 for pid in pid_base], lighter_gen_in_flavor=True
        )
        self._reorder_mixing_matrix_in_flavor("SNUMIX", [1000012, 1000014, 1000016])
        for i in ["USQMIX", "DSQMIX", "SELMIX", "SNUMIX"]:
            self._kill_lighter_gen_mixing(i)

        # prepare DECAY blocks with zero width, since mg5's `compute_width` fails if
        # these are not provided.
        for pid in [6, 23, 24]:
            self.output.slha.decays[pid] = yaslha.slha.Decay(pid)
        #            if self.output.mass(pid) is None:
        #                self.output.set_mass(pid, self.output.ewsb.mass(pid))
        mass_block = self.output.block("MASS")
        assert isinstance(mass_block, yaslha.slha.Block)
        for pid2 in mass_block.keys():
            assert isinstance(pid2, int)
            self.output.slha.decays[pid2] = yaslha.slha.Decay(pid2)

        # remove unsupported blocks (IMVCKM, IMUPMNS, GAUGE)
        for name in ["IMVCKM", "IMUPMNS"]:
            unsupported_block = self.output.block(name)
            if unsupported_block:
                for key, value in unsupported_block.items():
                    if isinstance(value, str) or abs(value) > 1e-18:
                        logger.warning(
                            "MG5 does not allow non-zero %s block: %s = %s is ignored.",
                            name,
                            key,
                            value,
                        )
            self.output.remove_block(name)
        self.output.remove_block("GAUGE")

        # use FRALPHA instead of ALPHA
        self.output.slha["FRALPHA", 1] = self.output.get_float_assert("ALPHA", None)
        self.output.remove_block("ALPHA")

        # dumper configuration
        self.output.dumper = yaslha.dumper.SLHADumper(
            separate_blocks=True,
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

    def _load_modsel(self) -> None:
        super()._load_modsel()
        if self.cpv != CPV.NONE:
            self.add_error("This calculator does not support CPV.")
        if self.flv != FLV.NONE:
            self.logger.warning(
                "Advisory warning: MSSM_SLHA2 model in MG5_aMC does not support "
                "flavor violation; you must create FeynRules model."
            )
