import pathlib
from typing import Dict, Optional, Sequence, List, Tuple, Union, Any, MutableMapping  # noqa: F401

import numpy as np
import yaslha


class AbsModel(yaslha.SLHA):
    """Abstract model as a wrapper of a SLHA object."""

    def __init__(self, obj: Union[None, str, pathlib.Path]=None)->None:
        if obj is None:
            super().__init__()
        elif isinstance(obj, str) and ('\n' in obj or '\r' in obj):
            super().__init__(yaslha.parse(obj))        # multiline: SLHA data itself
        else:
            super().__init__(yaslha.parse_file(obj))   # singleline: file path

        self._matrix_cache = dict()  # type: Dict[str, np.ndarray]
        self.dumper = None           # type: Optional[yaslha.dumper.SLHADumper]

    def block(self, block_name: str)->Optional[yaslha.Block]:
        return self.blocks.get(block_name.upper(), None)

    def get_float(self, block_name: str, key, default=None)->Optional[float]:
        value = self.get(block_name, key, default)
        return None if value is None else float(value)

    def get_complex(self, block_name: str, key, default=None)->Union[float, complex, None]:
        real = self.get_float(block_name, key)
        imaginary = self.get('IM' + block_name, key)
        if real is None and imaginary is None:
            return default
        elif imaginary:
            return complex(0 if real is None else real, imaginary)
        else:
            return real

    def mass(self, pid: int)->Optional[float]:
        return self.get('MASS', pid)

    def width(self, pid: int)->float:
        try:
            return self.decays[pid].width
        except KeyError:
            return None

    def br_list(self, pid: int)->Optional[MutableMapping[Tuple[int, ...], float]]:
        try:
            decay = self.decays[pid]
        except KeyError:
            return None
        return dict([(tuple(sorted(k)), v) for k, v in decay.items_br()])

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

    def remove_block(self, block_name):
        try:
            del self.blocks[block_name.upper()]
        except KeyError:
            pass

    def remove_value(self, block_name, key):
        try:
            del self.blocks[block_name][key]
        except KeyError:
            pass

    def write(self, filename: Optional[str]=None)->None:
        dumper = self.dumper or yaslha.dumper.SLHADumper(separate_blocks=True)
        slha_text = yaslha.dump(self, dumper=dumper)

        # append trivial comments because some old tools requires a comment on every line,
        # TODO: change the content to meaningful ones
        lines = slha_text.splitlines()
        for i, v in enumerate(lines):
            if len(v) != 1 and v.endswith('#'):
                lines[i] = v + ' ...'
        slha_text = '\n'.join(lines)

        if filename is None:
            # print(yaslha.dump(self, dumper=dumper))
            print(slha_text)
        else:
            # yaslha.dump_file(self, filename, dumper=dumper)
            with open(filename, 'w') as f:
                f.write(slha_text)
