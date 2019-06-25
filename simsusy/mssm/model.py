from typing import Dict, List, Optional  # noqa: F401

from simsusy.abs_model import AbsModel
from simsusy.mssm.abstract import AbsEWSBParameters, AbsSMParameters  # noqa: F401
from simsusy.mssm.input import A, MSSMInput, S  # noqa: F401


class MSSMModel(AbsModel):
    def __init__(self, *args):
        super().__init__(*args)
        self.input = None  # type: Optional[MSSMInput]
        self.sm = None  # type: Optional[AbsSMParameters]
        self.ewsb = None  # type: Optional[AbsEWSBParameters]

    def write(self, filename: Optional[str] = None) -> None:
        self._prepare_input_parameters()
        super().write(filename)

    def _prepare_input_parameters(self):
        assert self.input is not None

        for block_name, key_max in [("VCKMIN", 4), ("UPMNSIN", 6)]:
            block = self.input.block(block_name)
            for key in range(1, key_max + 1):
                v = block.get(key, default=None) if block else None
                if v is not None:
                    self.set(block_name, key, v)

        minpar_used = {
            1: False,
            2: False,
            3: True,
            4: True,
            5: False,
        }  # type: Dict[int, bool]
        extpar_used = dict()  # type: Dict[int, bool]
        for sfermion in [S.QL, S.UR, S.DR, S.LL, S.ER]:
            block = self.input.block(sfermion.slha2_input)
            for i in [1, 2, 3]:
                for j in [1, 2, 3]:
                    v = block.get((i, j), default=None) if block else None
                    if i == j:
                        extpar_used[sfermion.extpar + i] = v is None
                    if v is not None:
                        self.set(sfermion.slha2_input, (i, j), v)
        for a_term in [A.U, A.D, A.E]:
            block = self.input.block(a_term.slha2_input)
            for i in [1, 2, 3]:
                for j in [1, 2, 3]:
                    v = block.get((i, j), default=None) if block else None
                    if i == j == 3:
                        extpar_used[a_term.extpar] = v is None
                    if v is not None:
                        self.set(a_term.slha2_input, (i, j), v)

        # EXTPAR
        block = self.input.block("EXTPAR")
        for key in [1, 2, 3]:
            v = block[key] if block else None
            if v is not None:
                self.set("EXTPAR", key, v)
            else:
                minpar_used[2] = True
        for sfermion in [S.QL, S.UR, S.DR, S.LL, S.ER]:
            for gen in [1, 2, 3]:
                key = sfermion.extpar + gen
                if extpar_used[key]:
                    v = block.get(key, default=None) if block else None
                    if v is not None:
                        self.set("EXTPAR", key, v)
                    else:
                        minpar_used[1] = True
        for a_term in [A.U, A.D, A.E]:
            key = a_term.extpar
            if extpar_used[key]:
                v = block.get(key, default=None) if block else None
                if v is not None:
                    self.set("EXTPAR", key, v)
                else:
                    minpar_used[5] = True

        ewsb_params = list()  # Type: List[int]
        for key in [21, 22, 23, 24, 25, 26, 27]:
            v = block.get(key, default=None) if block else None
            if v is not None:
                ewsb_params.append(key)
                self.set("EXTPAR", key, v)
                if key == 23:
                    minpar_used[4] = False
                elif key == 25:
                    minpar_used[3] = False
        if len(ewsb_params) < 2:
            minpar_used[1] = True

        # MINPAR
        block = self.input.block("MINPAR")
        for key in [1, 2, 3, 4, 5]:
            if minpar_used[key]:
                v = block.get(key, default=None) if block else None
                if v is not None:
                    self.set("MINPAR", key, v)

        # SMINPUTS and MODSEL
        block = self.input.block("SMINPUTS")
        if block:
            for k, v in block.items():
                if 1 <= k <= 7 or k in [8, 11, 12, 13, 14, 21, 22, 23, 24]:
                    self.set("SMINPUTS", k, v)
        block = self.input.block("MODSEL")
        if block:
            for k, v in block.items():
                if k == 1:
                    self.set("MODSEL", k, v)
