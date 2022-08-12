"""
Tree-level MSSM calculator for MadGraph5 MSSM_SLHA2 model.

- SLHA2 is forced.
- SIMSUSY 111 = 1 because requested by MSSM_SLHA2.
- SIMSUSY 131-141 are forced to 1110 because requested by MSSM_SLHA2.
"""

import logging
from typing import Any, Tuple, Optional, MutableMapping

import yaslha

import simsusy.mssm.tree_calculator
import simsusy.simsusy
from simsusy.mssm.input import MSSMInput as Input  # noqa: F401
from simsusy.mssm.library import CPV, FLV
from simsusy.mssm.model import MSSMModel as Output  # noqa: F401

logger = logging.getLogger(__name__)


class Calculator(simsusy.mssm.tree_calculator.Calculator):
    """The tree-level calculator of MSSM for MSSM_SLHA2 model."""

    name = simsusy.simsusy.__pkgname__ + "/MSSMMG5Tree"
    version = simsusy.simsusy.__version__

    def __init__(self, input: Input) -> None:
        super().__init__(input=input)

    def write_output(self, filename: Optional[str] = None, slha1: bool = False) -> None:
        """
        Output the results to a file.

        The only options for the `simsusy` script are `slha1` flag and an
        output path, and thus this method accepts those values. Other output
        options are specified in SIMSUSY block.
        """
        if slha1:
            logger.warning("MSSM_SLHA2 model requires SLHA2. Output will be invalid.")
            super().write_output(filename, slha1)
            return

        options = self.read_simsusy_options(slha1=slha1)
        # MSSM_SLHA2 does not support IMNMIX but allows negative mass.
        options[111] = 1
        # MSSM_SLHA2 does not accept SLHA2 format of sfermion mixing.
        for i in (131, 132, 133, 134):
            if options[i] not in (0, 1110):
                self.add_warning(f"Ignored invalid option {i}={options[i]}.")
            options[i] = 1110

        # prepare as in TreeCalculator
        self._output_preparation(slha1, options)
        self._output_prepare_spinfo()
        if options[101]:
            self.output.slha["SPINFO", 3] = []

        # dumper configuration
        self.output.dumper = yaslha.dumper.SLHADumper(
            separate_blocks=True,
            comments_preserve=yaslha.dumper.CommentsPreserve.TAIL,
            document_blocks=self.document_blocks,
        )
        self.output.write(filename)

    def _output_preparation(self, slha1, options):
        # type: (bool, MutableMapping[int, int]) -> None
        super()._output_preparation(slha1, options)
        if slha1:
            return

        # MSSM_SLHA2 does not accept SLHA2 format of sfermion mixing.
        for name in ["USQMIX", "DSQMIX", "SELMIX", "SNUMIX"]:
            r = self.output.get_matrix(name)
            max_mix = (-1, -1, 0)
            for i in range(3 if name == "SNUMIX" else 6):
                for j in range(3 if name == "SNUMIX" else 6):
                    if i in (2, 5) and j in (2, 5):
                        continue
                    if i != j and (v := abs(r[i, j])) > max_mix[2]:
                        max_mix = (i + 1, j + 1, v)
                    # diagonal mixing for lighter generation
                    self.output.slha[name, i + 1, j + 1] = 1.0 if i == j else 0.0
            if max_mix[2] > 0:
                self.add_warning(
                    "Ignored lighter-gen {} (max:{}{} = {:.2e})".format(name, *max_mix)
                )

        # use FRALPHA instead of ALPHA
        self.output.slha["FRALPHA", 1] = self.output.get_float("ALPHA", None)
        self.output.remove_block("ALPHA")

        # prepare DECAY blocks with zero width
        # because mg5's `compute_width` fails if these are not provided.
        for pid in [6, 23, 24]:
            self.output.slha.decays[pid] = yaslha.slha.Decay(pid)
        # if self.output.mass(pid) is None:
        #     self.output.set_mass(pid, self.output.ewsb.mass(pid))
        mass_block = self.output.block("MASS")
        assert isinstance(mass_block, yaslha.slha.Block)
        for pid2 in mass_block.keys():
            if isinstance(pid2, int):
                self.output.slha.decays[pid2] = yaslha.slha.Decay(pid2)

        # remove unsupported blocks (IMVCKM, IMUPMNS, GAUGE)
        for name in ["IMVCKM", "IMUPMNS"]:
            unsupported_block = self.output.block(name)
            if not isinstance(unsupported_block, yaslha.slha.Block):
                continue
            max_entry: Tuple[Any, int] = (None, 0)
            for key, value in unsupported_block.items():
                if isinstance(value, str):
                    self.add_warning(f"Removed strange entry {name}:{key}={value}")
                elif (v := abs(value)) > max_entry[1]:
                    max_entry = (key, v)
            if max_entry[1] > 0:
                self.add_warning("Ignored unacceptable {name} (max: {key}={value})")
            self.output.remove_block(name)
        self.output.remove_block("GAUGE")

    def _load_modsel(self) -> None:
        super()._load_modsel()
        if self.cpv != CPV.NONE:
            self.add_error("This calculator does not support CPV.")
        if self.flv != FLV.NONE:
            self.logger.warning(
                "Advisory warning: MSSM_SLHA2 model in MG5_aMC does not support "
                "flavor violation; you must create FeynRules model."
            )
