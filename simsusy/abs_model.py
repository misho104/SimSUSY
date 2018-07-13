import pyslha
import pathlib
import numpy as np
from typing import Dict, Optional, Sequence, List, Tuple, Union, Any, MutableMapping  # noqa: F401

from simsusy.pyslha_customize import KeyType, ValueType, CommentType, writeSLHABlocks, writeSLHADecays
pyslha.writeSLHABlocks = writeSLHABlocks
pyslha.writeSLHADecays = writeSLHADecays


class AbsModel:
    """Abstract model as a wrapper of a SLHA object."""

    def __init__(self, obj: Union[pyslha.Doc, str, None]=None)->None:
        self._matrix_cache = dict()  # type: Dict[str, np.ndarray]
        if obj is None:
            self._slha = pyslha.Doc(blocks=pyslha._dict('Blocks'))
        elif isinstance(obj, pyslha.Doc):
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

    def get_complex(self, block_name: str, key, default=None)->Union[float, complex, None]:
        real = self.get_float(block_name, key)
        imaginary = self.get('IM' + block_name, key)
        if real is None and imaginary is None:
            return default
        elif imaginary:
            return complex(0 if real is None else real, imaginary)
        else:
            return real

    def get_float(self, block_name: str, key, default=None)->Optional[float]:
        value = self.get(block_name, key, default)
        return None if value is None else float(value)

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

    def set(self, block_name: str, key: KeyType, value: ValueType, comment: CommentType='')->None:
        block_name = block_name.upper()
        if self.block(block_name) is None:
            self._slha.blocks[block_name] = pyslha.Block(block_name)
        self._slha.blocks[block_name][key] = value
        # TODO: handle comment...

    def set_mass(self, key: int, mass: float)->None:  # just a wrapper
        self.set('MASS', key, mass)

    def set_matrix(self, block_name: str, matrix: np.ndarray, diagonal_only=False)->None:
        self._matrix_cache[block_name] = matrix
        nx, ny = matrix.shape
        for i in range(0, nx):
            for j in range(0, ny):
                if diagonal_only and i != j:
                    continue
                self.set(block_name, (i + 1, j + 1), matrix[i, j])

    def get_matrix(self, block_name: str)->Optional[np.ndarray]:
        cache = self._matrix_cache.get(block_name)
        if isinstance(cache, np.ndarray):
            return cache
        block = self.block(block_name)
        if not block:
            return None
        nx_ny = max(block.keys())
        assert isinstance(nx_ny, tuple) and len(nx_ny) == 2 and all(isinstance(nx_ny[i], int) for i in nx_ny)
        matrix = np.zeros(nx_ny)
        for x in range(0, nx_ny[0]):
            for y in range(0, nx_ny[1]):
                matrix[x, y] = block.get((x, y), default=0)
        self._matrix_cache[block_name] = matrix
        return self._matrix_cache[block_name]

    def set_q(self, block_name, q: float):
        block_name = block_name.upper()
        if self.block(block_name) is None:
            self._slha.blocks[block_name] = pyslha.Block(block_name)
        self._slha.blocks[block_name].q = q

    def remove_block(self, block_name):
        try:
            del self._slha.blocks[tuple(block_name.upper())]
        except KeyError:
            pass

    def remove_value(self, block_name, key):
        if self.get(block_name, key) is not None:
            del self._slha.blocks[block_name].entries[key]

    def write(self, filename: Optional[str]=None, ignorenobr: bool=True, precision: int=8) -> None:
        """provide own version of write, because pyslha.Doc.write has a bug."""
        if filename is None:
            print(pyslha.writeSLHA(self._slha, ignorenobr=ignorenobr, precision=precision))
        else:
            pyslha.write(filename, self._slha, ignorenobr=ignorenobr, precision=precision)


class Info:
    def __init__(self, name: str, version: str)->None:
        self.name = name          # type: str
        self.version = version    # type: str
        self.errors = list()      # type: List[str]
        self.warnings = list()    # type: List[str]

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)
