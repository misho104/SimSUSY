import enum
import pyslha
import pathlib
from typing import Dict, Optional, Sequence, Tuple, Union, Any  # noqa: F401


class SLHAVersion(enum.Enum):
    SLHA1 = 1
    SLHA2 = 2


class AbsModel:
    """Abstract model as a wrapper of a SLHA object."""

    def __init__(self, obj: Union[pyslha.Doc, str])->None:
        if isinstance(obj, pyslha.Doc):
            self._slha = obj
        elif isinstance(obj, pathlib.Path):
            self._slha = pyslha.readSLHAFile(str(obj), ignorenomass=True)
        elif isinstance(obj, str):
            if '\n' in obj or '\r' in obj:
                # if multiline, assume obj as the SLHA content.
                self._slha = pyslha.readSLHA(obj, ignorenomass=True)
            else:
                # if single line, it is a file path.
                self._slha = pyslha.readSLHAFile(obj, ignorenomass=True)
        else:
            raise ValueError('invalid initialization of Model')

    def block(self, name: str) -> Optional[pyslha.Block]:
        try:
            return self._slha.blocks[name.upper()]
        except KeyError:
            return None

    def get(self, block_name: str, key, default=None) -> Any:
        # we introduce this because pyslha.get raises keyerror if block is not found.
        block = self.block(block_name)
        if block:
            try:
                return block[key]
            except KeyError:
                pass
        return default

    def mass(self, pid: int)->Optional[float]:
        return self.get('MASS', pid)

    def width(self, pid: int)->Optional[float]:
        try:
            return self._slha.decays[pid].totalwidth
        except KeyError:
            return None

    def br(self, pid: int, *daughters: int) -> Optional[float]:
        n_decay = len(daughters)
        sorted_daughters = sorted(daughters)
        try:
            particle = self._slha.decays[pid]
        except KeyError:
            return None
        for c in particle.decays:
            if n_decay == c.nda and sorted(c.ids) == sorted_daughters:
                return c.br
        return 0

    def br_list(self, pid: int) -> Optional[Dict[Sequence[int], float]]:
        try:
            particle = self._slha.decays[pid]
        except KeyError:
            return None
        return dict([(tuple(sorted(c.ids)), c.br) for c in particle.decays])
