from simsusy.abs_model import AbsModel


class AbsCalculator:
    def __init__(self, input: AbsModel)->None:
        self.input = input

    def calculate(self):
        raise NotImplementedError
