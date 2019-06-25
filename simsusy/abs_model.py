import pathlib
from typing import (  # noqa: F401
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import yaslha


class AbsModel:
    slha: yaslha.slha.SLHA
    _matrix_cache: Dict[str, np.ndarray]
    dumper: Optional[yaslha.dumper.SLHADumper]

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        if path:
            self.slha = yaslha.parse_file(path)
        else:
            self.slha = yaslha.slha.SLHA()

        self._matrix_cache = dict()
        self.dumper = None

    @property
    def blocks(self):
        return self.slha.blocks

    def block(self, block_name: str) -> Optional[yaslha.block.Block]:
        return self.slha.blocks.get(block_name, None)

    def get(self, *key: Any, default: Any = None) -> Any:
        return self.slha.get(*key, default=default)

    def get_float(self, *key: Any, default: Any = None) -> Optional[float]:
        value = self.get(*key, default=default)
        return None if value is None else float(value)

    def get_complex(
        self, block_name: str, key, default=None
    ) -> Union[float, complex, None]:
        real = self.get_float(block_name, key)
        imaginary = self.get_float("IM" + block_name, key)
        if real is None and imaginary is None:
            return default
        elif imaginary:
            return complex(0 if real is None else real, imaginary)
        else:
            return real

    def mass(self, pid: int) -> Optional[float]:
        return self.get_float("MASS", pid)

    def width(self, pid: int) -> Optional[float]:
        try:
            return self.slha.decays[pid].width
        except KeyError:
            return None

    def br_list(self, pid: int) -> Optional[MutableMapping[Tuple[int, ...], float]]:
        try:
            decay = self.slha.decays[pid]
        except KeyError:
            return None
        return dict(decay.items_br())

    def set_mass(self, key: int, mass: float) -> None:  # just a wrapper
        self.slha["MASS", key] = mass

    def set_matrix(self, block_name: str, m: np.ndarray, diagonal_only=False) -> None:
        self._matrix_cache[block_name] = m
        nx, ny = m.shape
        for i in range(0, nx):
            for j in range(0, ny):
                if diagonal_only and i != j:
                    continue
                self.slha[block_name, i + 1, j + 1] = m[i, j]

    def get_matrix(self, block_name: str) -> Optional[np.ndarray]:
        cache = self._matrix_cache.get(block_name)
        if isinstance(cache, np.ndarray):
            return cache
        block = self.slha.blocks[block_name]
        if not block:
            return None
        nx_ny = max(block.keys())
        assert (
            isinstance(nx_ny, tuple)
            and len(nx_ny) == 2
            and all(isinstance(nx_ny[i], int) for i in nx_ny)
        )
        matrix = np.zeros(nx_ny)
        for x in range(0, nx_ny[0]):
            for y in range(0, nx_ny[1]):
                matrix[x, y] = block.get((x, y), default=0)
        self._matrix_cache[block_name] = matrix
        return self._matrix_cache[block_name]

    def remove_block(self, block_name):
        try:
            del self.slha.blocks[block_name]
        except KeyError:
            pass

    def remove_value(self, block_name, key):
        try:
            del self.slha.blocks[block_name][key]
        except KeyError:
            pass

    def write(self, filename: Optional[str] = None) -> None:
        dumper = self.dumper or yaslha.dumper.SLHADumper(separate_blocks=True)
        slha_text = yaslha.dump(self.slha, dumper=dumper)

        # append trivial comments because some old tools requires a comment on every line,
        # TODO: change the content to meaningful ones
        lines = slha_text.splitlines()
        for i, v in enumerate(lines):
            if len(v) != 1 and v.endswith("#"):
                lines[i] = v + " ..."
        slha_text = "\n".join(lines) + "\n"
        if dumper.config("forbid_last_linebreak"):
            slha_text = slha_text.rstrip()

        if filename is None:
            # print(yaslha.dump(self, dumper=dumper))
            print(slha_text)
        else:
            # yaslha.dump_file(self, filename, dumper=dumper)
            with open(filename, "w") as f:
                f.write(slha_text)
