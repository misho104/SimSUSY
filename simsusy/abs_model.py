import logging
import pathlib
from typing import (  # noqa: F401
    Any,
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import numpy.typing
import yaslha

ComplexMatrix = numpy.typing.NDArray[np.complex_]
RealMatrix = numpy.typing.NDArray[np.float_]
Matrix = numpy.typing.NDArray[Union[np.float_, np.complex_]]
logger = logging.getLogger(__name__)


class AbsModel:
    slha: yaslha.slha.SLHA
    _matrix_cache: Dict[str, RealMatrix]
    dumper: Optional[yaslha.dumper.SLHADumper]

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        if path:
            self.slha = yaslha.parse_file(path)
        else:
            self.slha = yaslha.slha.SLHA()

        self._matrix_cache = dict()
        self.dumper = None

    @property
    def blocks(self) -> yaslha.slha.BlocksDict:
        return self.slha.blocks

    def block(self, block_name):
        # type: (str) -> Union[yaslha.block.Block, yaslha.block.InfoBlock, None]
        return self.slha.blocks.get(block_name, None)

    def get_int(self, *key):
        # type: (Any) -> Optional[int]
        """Possibly get an integer value from the SLHA."""
        return None if (value := self.slha.get(*key)) is None else int(value)

    def get_float(self, *key):
        # type: (Any) -> Optional[float]
        """Possibly get a float value from the SLHA."""
        return None if (value := self.slha.get(*key)) is None else float(value)

    def get_complex(self, block_name, *key):
        # type: (str, Any) -> Optional[complex]
        """Possibly get a value from the SLHA with the complex-number handling.

        The result may be complex or float.
        """
        real = self.get_float(block_name, *key)
        imaginary = self.get_float("IM" + block_name, *key)
        if real is None and imaginary is None:
            return None
        elif imaginary:
            return complex(0 if real is None else real, imaginary)
        else:
            return real

    def mass(self, pid: int) -> Optional[float]:
        return self.get_float("MASS", pid)

    def mass_assert(self, pid: int) -> float:
        return self.get_float_assert("MASS", pid)

    def width(self, pid: int) -> Optional[float]:
        try:
            return self.slha.decays[pid].width
        except KeyError:
            return None

    def br_list(self, pid: int) -> Optional[MutableMapping[Sequence[int], float]]:
        try:
            decay = self.slha.decays[pid]
        except KeyError:
            return None
        return dict(decay.items_br())

    def set_mass(self, key: int, mass: float) -> None:  # just a wrapper
        self.slha["MASS", key] = mass

    def set_matrix(self, block_name, m, diagonal_only=False):
        # type: (str, RealMatrix, bool) -> None
        self._matrix_cache[block_name] = m
        nx, ny = m.shape
        for i in range(0, nx):
            for j in range(0, ny):
                if diagonal_only and i != j:
                    continue
                self.slha[block_name, i + 1, j + 1] = m[i, j]

    def set_complex_matrix(self, block_name, m, diagonal_only=False):
        # type: (str, Matrix, bool) -> None
        im_block_name = "IM" + block_name
        self.set_matrix(block_name, m.real, diagonal_only)

        if not np.all(np.isreal(m)):
            self.set_matrix("IM" + block_name, m.imag, diagonal_only)
        else:
            if im_block_name in self.slha.blocks:
                del self.slha.blocks[im_block_name]
            if im_block_name in self._matrix_cache:
                del self._matrix_cache[im_block_name]

    def get_matrix(self, block_name):
        # type: (str) -> Optional[RealMatrix]
        """Possibly get a real matrix block from the SLHA.

        All the entries are validated to be int or float.
        """
        cache = self._matrix_cache.get(block_name)
        if isinstance(cache, np.ndarray):
            return cache
        block = self.slha.blocks[block_name]
        if not isinstance(block, yaslha.slha.Block):
            logger.warning("The block %s is not found or not matrix-like.", block_name)
            return None
        max_key = [0, 0]  # type: List[int]
        for key in block.keys():
            if (
                isinstance(key, Sequence)
                and len(key) == 2
                and all(isinstance(i, int) for i in key)
            ):
                max_key[0] = max(max_key[0], key[0])
                max_key[1] = max(max_key[1], key[1])
            else:
                logger.warning(f"The block {block_name} is not matrix-like.")
                return None
        matrix = np.zeros(max_key)
        for x in range(0, max_key[0]):
            for y in range(0, max_key[1]):
                entry = block.get((x + 1, y + 1), default=0)
                if isinstance(entry, int) or isinstance(entry, float):
                    matrix[x, y] = entry
        self._matrix_cache[block_name] = matrix
        return self._matrix_cache[block_name]

    def get_complex_matrix(self, block_name):
        # type: (str) -> Optional[Matrix]
        """Possibly get a complex matrix block from the SLHA."""
        re_part = self.get_matrix(block_name)
        if "IM" + block_name in self.slha.blocks:
            im_part = self.get_matrix("IM" + block_name)
        else:
            im_part = None
        if re_part is not None and im_part is not None:
            return re_part + im_part * 1j
        elif im_part is None and re_part is not None:
            return re_part
        elif re_part is None and im_part is not None:
            return im_part * 1j
        else:
            return None

    def remove_block(self, block_name: str) -> None:
        try:
            del self.slha.blocks[block_name]
        except KeyError:
            pass

    def remove_value(self, block_name: str, key: yaslha.line.KeyType) -> None:
        try:
            del self.slha.blocks[block_name][key]  # type: ignore
        except KeyError:
            pass

    def write(self, filename: Optional[str] = None) -> None:
        dumper = self.dumper or yaslha.dumper.SLHADumper(separate_blocks=True)
        slha_text = yaslha.dump(self.slha, dumper=dumper)

        # append trivial comments because some old tools require a comment on each line
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

    def get_int_assert(self, *key, default=None):
        # type: (Any, Optional[int]) -> int
        """Get an integer value with assertion from the SLHA."""
        if (value := self.get_int(*key)) is not None:
            return value
        elif default is not None:
            return default
        raise ValueError(f"{key} is not specified or not an integer.")

    def get_float_assert(self, *key, default=None):
        # type: (Any, Optional[float]) -> float
        """Get a float value with assertion from the SLHA."""
        if (value := self.get_float(*key)) is not None:
            return value
        elif default is not None:
            return default
        raise ValueError(f"{key} is not specified or not a float.")

    def get_complex_assert(self, block_name, *key, default=None):
        # type: (str, Any, Optional[complex]) -> complex
        """Get a complex value with assertion from the SLHA."""
        if (value := self.get_complex(block_name, *key)) is not None:
            return value
        elif default is not None:
            return default
        raise ValueError(f"{key} is not specified or not a (complex) number.")

    def get_matrix_assert(self, block_name, default=None):
        # type: (str, Optional[RealMatrix]) ->RealMatrix
        """Get a real matrix block with assertion from the SLHA."""
        if (value := self.get_matrix(block_name)) is not None:
            # get_matrix already asserts the entries are int or float.
            return value
        if default is not None:
            return default
        raise ValueError(f"{block_name} is not a real matrix block.")

    def get_complex_matrix_assert(self, block_name, default=None):
        # type: (str, Optional[Matrix]) ->Matrix
        """Get a complex matrix block with assertion from the SLHA."""
        if (value := self.get_complex_matrix(block_name)) is not None:
            return value
        if default is not None:
            return default
        raise ValueError(f"{block_name} is not a (complex) matrix block.")
