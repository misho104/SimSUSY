from simsusy.abs_model import AbsModel
from typing import Optional  # noqa: F401


class AbsCalculator:
    def __init__(self, input: AbsModel)->None:
        self.input = input            # type: AbsModel
        self.output = NotImplemented  # type: AbsModel

    def write_output(self, filename: Optional[str]=None):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError
