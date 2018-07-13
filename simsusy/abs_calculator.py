from simsusy.abs_model import AbsModel
from typing import Optional, List, Tuple, Union  # noqa: F401
import logging


class AbsCalculator:
    def __init__(self, input: AbsModel, logger: Optional[logging.Logger]=None)->None:
        self.input = input            # type: AbsModel
        self.output = NotImplemented  # type: AbsModel
        self.logger = logger if logger else logging.getLogger(__name__)  # type: logging.Logger

        # logger to output as SPINFO 3/4
        self._errors = []        # type: List[str]
        self._warnings = []      # type: List[str]

    def write_output(self, filename: Optional[str]=None, slha1: bool=False)->None:
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    @staticmethod
    def to_message(obj: Union[str, Tuple[Union[str, int, float]]])->str:
        if isinstance(obj, str):
            return obj
        else:
            return obj.__repr__()

    def add_error(self, obj: Union[str, Tuple[Union[str, int, float]]]):
        message = self.to_message(obj)
        self._errors.append(message)
        self.logger.error(message)

    def add_warning(self, obj: Union[str, Tuple[Union[str, int, float]]]):
        message = self.to_message(obj)
        self._warnings.append(message)
        self.logger.warning(message)
