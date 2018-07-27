import logging
from collections import OrderedDict
from typing import List, Optional  # noqa: F401

import yaslha

from simsusy.mssm.library import CPV, FLV
import simsusy.simsusy
import simsusy.mssm.tree_calculator
from simsusy.mssm.model import MSSMModel as Output  # noqa: F401
from simsusy.mssm.input import MSSMInput as Input  # noqa: F401

logger = logging.getLogger(__name__)


class Calculator(simsusy.mssm.tree_calculator.Calculator):
    name = simsusy.simsusy.__pkgname__ + '/MSSMTree'
    version = simsusy.simsusy.__version__

    def __init__(self, input: Input)->None:
        super().__init__(input=input)

    def write_output(self, filename: Optional[str]=None, slha1: bool=False)->None:
        # MSSM_SLHA2 does not accept SLHA2 format of sfermion mixing.
        pid_base = (1000001, 1000003, 1000005, 2000001, 2000003, 2000005)
        self._reorder_no_flv_mixing_matrix('USQMIX', [pid + 1 for pid in pid_base], lighter_lr_mixing=False)
        self._reorder_no_flv_mixing_matrix('DSQMIX', [pid for pid in pid_base], lighter_lr_mixing=False)
        self._reorder_no_flv_mixing_matrix('SELMIX', [pid + 10 for pid in pid_base], lighter_lr_mixing=False)
        self._reorder_no_flv_mixing_matrix('SNUMIX', [1000012, 1000014, 1000016], lighter_lr_mixing=False)

        # prepare DECAY blocks with zero width, since mg5's `compute_width` fails if these are not provided.
        for pid in [6, 23, 24]:
            self.output.decays[pid] = yaslha.Decay(pid)
#            if self.output.mass(pid) is None:
#                self.output.set_mass(pid, self.output.ewsb.mass(pid))
        for pid in self.output.block('MASS').keys():
            self.output.decays[pid] = yaslha.Decay(pid)

        # remove unsupported blocks (IMVCKM, IMUPMNS, GAUGE)
        for name in ['IMVCKM', 'IMUPMNS']:
            unsupported_block = self.output.block(name)
            if unsupported_block:
                for key, value in unsupported_block.items():
                    if abs(value) > 1e-18:
                        logger.warning(f'MG5 does not support non-zero {name} block: {key} = {value} is ignored.')
            self.output.remove_block(name)
        self.output.remove_block('GAUGE')

        # use FRALPHA insstead of ALPHA
        self.output.set('FRALPHA', 1, self.output.get('ALPHA', None))
        self.output.remove_block('ALPHA')

        # dumper configuration
        self.output.dumper = yaslha.dumper.SLHADumper(
            separate_blocks=True,
            document_blocks=[
                'MODSEL', 'MINPAR', 'EXTPAR',
                'VCKMIN', 'UPMNSIN', 'MSQ2IN', 'MSU2IN', 'MSD2IN', 'MSL2IN', 'MSE2IN', 'TUIN', 'TDIN', 'TEIN',
            ])

        # done
        super().write_output(filename, slha1)

    def _load_modsel(self):
        super()._load_modsel()
        if self.cpv != CPV.NONE:
            self.add_error('This calculator does not support CPV.')
        if self.flv != FLV.NONE:
            self.logger.warning(
                'Advisory warning: MSSM_SLHA2 model in MG5_aMC does not support flavor violation; '
                'you must create FeynRules model.')
