"""
Abstract class for calculators.

A calculator will have an input "Model" and an output "Model". In addition, a
handler of errors and warnings is provided, where messages are stored in SPINFO
blocks of the output SLHA file as well as shown on screen.
"""


import logging
from typing import List, Optional, Tuple, Union  # noqa: F401

from simsusy.abs_model import AbsModel

MessageType = Union[str, Tuple[Union[str, int, float]]]


class AbsCalculator:
    """Abstract class for calculators."""

    input: AbsModel  # noqa: A003
    output: AbsModel
    logger: logging.Logger

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger if logger else logging.getLogger(__name__)

        # logger to output as SPINFO 3/4
        self._errors = []  # type: List[str]
        self._warnings = []  # type: List[str]

    def write_output(self, filename: Optional[str] = None, slha1: bool = False) -> None:
        """
        Output the results to a file.

        The only options for the `simsusy` script are `slha1` flag and an
        output path, and thus this method accepts those values.
        """
        self._output_prepare_spinfo()
        if "SIMSUSY" in self.output.blocks:
            del self.output.slha["SIMSUSY"]
        self.output.write(filename)

    def calculate(self) -> None:
        """Do the calculation."""
        raise NotImplementedError

    @staticmethod
    def to_message(obj: Union[str, Tuple[Union[str, int, float]]]) -> str:
        """Convert an error object to a string."""
        if isinstance(obj, str):
            return obj
        else:
            return obj.__repr__()

    def add_error(self, obj, obj_slha=None):
        # type: (MessageType, Optional[MessageType])->None
        """
        Add an error message.

        A shorter version for the SPINFO block may be specified in `obj_slha`.
        """
        message_shown = self.to_message(obj)
        message_written = self.to_message(obj_slha) if obj_slha else message_shown
        self._errors.append(message_written)
        self.logger.error(message_shown)

    def add_warning(self, obj, obj_slha=None):
        # type: (MessageType, Optional[MessageType])->None
        """
        Add a warning message.

        A shorter version for the SPINFO block may be specified in `obj_slha`.
        """
        message_shown = self.to_message(obj)
        message_written = self.to_message(obj_slha) if obj_slha else message_shown
        self._warnings.append(message_written)
        self.logger.warning(message_shown)

    def _output_prepare_spinfo(self) -> None:
        """Prepare SPINFO block based on stored errors and warnings."""
        self.output.slha["SPINFO", 3] = sorted(set(self._warnings))
        self.output.slha["SPINFO", 4] = sorted(set(self._errors))
