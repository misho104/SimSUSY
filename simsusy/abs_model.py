"""
Abstract class for a model.

A model corresponds to an SLHA file, i.e., a set of parameters of any model. It
is usually an input file or an output (calculated) file. This class provides
basic I/O functions of an integer, a float, or a matrix, and a complex value or
a complex matrix, where the "IM" convention is assumed.  In the early version,
matrices are cached, but this feature is removed to simplify the codes.
"""

import logging
import pathlib
from multiprocessing.sharedctypes import Value
from typing import (  # noqa: F401
    Any,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    Union,
    overload,
)

import numpy as np
import numpy.typing
import yaslha
import yaslha.block

ComplexMatrix = numpy.typing.NDArray[np.complex_]
RealMatrix = numpy.typing.NDArray[np.float_]
Matrix = numpy.typing.NDArray[Union[np.float_, np.complex_]]
logger = logging.getLogger(__name__)

RAISE_ERROR = object()


class AbsModel:
    """
    Abstract class for models.

    A model represents an SLHA file, i.e., a set of SLHA blocks. This class is
    thus regarded as an extension of the SLHA class.
    """

    slha: yaslha.slha.SLHA
    dumper: Optional[yaslha.dumper.SLHADumper]

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        if path:
            self.slha = yaslha.parse_file(path)
        else:
            self.slha = yaslha.slha.SLHA()
        self.dumper = None

    @property
    def blocks(self) -> yaslha.slha.BlocksDict:
        """Return a list of blocks in the SLHA."""
        return self.slha.blocks

    def block(self, block_name):
        # type: (str) -> Union[yaslha.block.Block, yaslha.block.InfoBlock, None]
        """Return a block in the SLHA."""
        return self.slha.blocks.get(block_name, None)

    @overload
    def get_int(self, *key: Any) -> int:
        ...

    @overload
    def get_int(self, *key: Any, default: int) -> int:
        ...

    @overload
    def get_int(self, *key: Any, default: None) -> Optional[int]:
        ...

    def get_int(self, *key: Any, default: Any = RAISE_ERROR) -> Optional[int]:
        """
        Get an integer value from the SLHA.

        If the value associated to `key` is not found, `default` is returned,
        where if it is unspecified an error is raised.
        """
        if isinstance(value := self.slha.get(*key), (int, float)):
            return round(value)  # use round function to absorb precision error
        elif default is None or isinstance(default, int):
            return default
        raise KeyError(f"{key} is not specified or not an integer.")

    @overload
    def get_float(self, *key: Any) -> float:
        ...

    @overload
    def get_float(self, *key: Any, default: float) -> float:
        ...

    @overload
    def get_float(self, *key: Any, default: None) -> Optional[float]:
        ...

    def get_float(self, *key: Any, default: Any = RAISE_ERROR) -> Optional[float]:
        """
        Get a float value from the SLHA.

        If the value associated to `key` is not found, `default` is returned,
        where if it is unspecified an error is raised.
        """
        if isinstance(value := self.slha.get(*key), (int, float)):
            return value
        elif default is None or isinstance(default, (int, float)):
            return default
        raise KeyError(f"{key} is not specified or not a float.")

    @overload
    def get_complex(self, block_name: str, *key: Any) -> complex:
        ...

    @overload
    def get_complex(self, block_name: str, *key: Any, default: complex) -> complex:
        ...

    @overload
    def get_complex(self, block_name, *key, default):
        # type: (str, Any, None) -> Optional[complex]
        ...

    def get_complex(self, block_name, *key, default=RAISE_ERROR):
        # type: (str, Any, Any) -> Optional[complex]
        """
        Get a complex value from the SLHA, based on "IM"-block convention.

        A complex or float value is returned. If the value associated to `key`
        is not found, `default` is returned, while if it is unspecified an
        error is raised.
        """
        real = self.get_float(block_name, *key, default=None)
        imaginary = self.get_float("IM" + block_name, *key, default=None)
        if imaginary is not None:
            return (0 if real is None else real) + imaginary * 1j
        elif real is not None:
            return real
        elif default is None or isinstance(default, complex):
            return default
        elif isinstance(default, SupportsFloat):
            return float(default)
        raise KeyError(f"{key} is not specified or not a (complex) number.")

    def remove_block(self, block_name: str) -> None:
        """Remove a block if exists."""
        try:
            del self.slha.blocks[block_name]
        except KeyError:
            pass

    def remove_value(self, block_name: str, key: yaslha.line.KeyType) -> None:
        """Remove a value if exists."""
        try:
            del self.slha.blocks[block_name][key]  # type: ignore
        except KeyError:
            pass

    # Matrix manipulation.
    # When setting a real matrix, all the elements are written on the SLHA. Suppression
    # of zero or tiny elements should be done later.
    # Similarly, when setting a complex matrix, all the real parts are written.
    # Meanwhile, the "IM" block will be created only if necessary, and only non-zero
    # elements are set.
    # When reading a matrix, the matrix width and height are determined by the maximal
    # index, respectively. Each element of `get_matrix` is a float, while each of
    # `get_complex_matrix' will be complex or float. If default values are specified,
    # they are returned, which can be None as well. Otherwise an error is raised.

    def set_matrix(self, block_name, m, diagonal_only=False):
        # type: (str, RealMatrix, bool) -> None
        """
        Set a matrix in the SLHA.

        All the elements are written on the SLHA even if zero.
        """
        # prepare the block
        block = self.slha.get(block_name)
        if isinstance(block, yaslha.block.Block):
            to_update = {k: True for k in block.keys()}
        else:
            block = yaslha.block.Block(block_name)
            self.slha.add_block(block)
            to_update = {}
        # set the matrix
        nx, ny = m.shape
        for i in range(0, nx):
            for j in range(0, ny):
                if diagonal_only and i != j:
                    if to_update.get((i + 1, j + 1)):
                        del block[(i + 1, j + 1)]
                        to_update[(i + 1, j + 1)] = False
                else:
                    block[i + 1, j + 1] = m[i, j]
                    to_update[(i + 1, j + 1)] = False
        for k, v in to_update.items():
            if v:
                block[k] = 0.0

    def set_complex_matrix(self, block_name, m, diagonal_only=False):
        # type: (str, Matrix, bool) -> None
        """
        Set a complex matrix in the SLHA.

        All the real and imaginary parts are written even if zero.
        """
        self.set_matrix(block_name, m.real, diagonal_only)
        self.set_matrix("IM" + block_name, m.imag, diagonal_only)

    @overload
    def get_matrix(self, block_name: str) -> RealMatrix:
        ...

    @overload
    def get_matrix(self, block_name: str, default: RealMatrix) -> RealMatrix:
        ...

    @overload
    def get_matrix(self, block_name: str, default: None) -> Optional[RealMatrix]:
        ...

    def get_matrix(self, block_name, default=RAISE_ERROR):
        # type: (str, Any) -> Optional[RealMatrix]
        """
        Get a real matrix from the SLHA.

        The matrix width and height are determined by the maximal index,
        respectively. The results will be a matrix with float elements. If the
        block is not found but `default` is specified as a matrix or None, it
        will be returned.
        """
        block = self.slha.blocks.get(block_name)
        if not isinstance(block, yaslha.block.Block):
            if default is None or (
                isinstance(default, np.ndarray) and np.all(np.isreal(default))
            ):
                return default
            raise KeyError(f"The block {block_name} not found.")

        max_key = [0, 0]  # type: List[int]
        entries = {}  # type: Dict[Tuple[int, int], Union[int, float]]
        for key, value in block.items():
            if (
                isinstance(key, Sequence)
                and len(key) == 2
                and all(isinstance(i, int) and i > 0 for i in key)
                and (isinstance(value, int) or isinstance(value, float))
            ):
                max_key[0] = max(max_key[0], key[0])
                max_key[1] = max(max_key[1], key[1])
                entries[(key[0], key[1])] = value
            else:
                raise ValueError(f"The block {block_name} is not matrix-like.")
        matrix = np.zeros(max_key)
        for key, value in entries.items():
            matrix[key[0] - 1, key[1] - 1] = value
        return matrix

    @overload
    def get_complex_matrix(self, block_name: str) -> Matrix:
        ...

    @overload
    def get_complex_matrix(self, block_name: str, default: Matrix) -> Matrix:
        ...

    @overload
    def get_complex_matrix(self, block_name: str, default: None) -> Optional[Matrix]:
        ...

    def get_complex_matrix(self, block_name, default=RAISE_ERROR):
        # type: (str, Any) -> Optional[Matrix]
        """
        Get a possibly-complex matrix from the SLHA.

        The matrix width and height are determined by the maximal index,
        respectively. The results will be a matrix with float elements. If the
        block is not found but `default` is specified as a matrix or None, it
        will be returned.
        """
        """Possibly get a complex matrix block from the SLHA."""
        re_part = self.get_matrix(block_name, default=None)
        im_part = self.get_matrix("IM" + block_name, default=None)
        if re_part is not None and im_part is not None:
            if default is None or isinstance(default, np.ndarray):
                return default
            raise KeyError(f"The block {block_name} not found.")
        elif im_part is None:
            return re_part
        elif re_part is None:
            return im_part * 1j
        # merge re_part and im_part
        dimensions = re_part.shape, im_part.shape
        if dimensions[0] == dimensions[1]:
            return re_part + im_part * 1j
        result = np.zeros(tuple(max(i) for i in dimensions), np.complex64)
        for index, x in np.ndenumerate(re_part):
            result[index] = x
        for index, x in np.ndenumerate(im_part):
            result[index] += x * 1j
        return result

    def write(self, filename: Optional[str] = None) -> None:
        """Write the SLHA to display or to a file."""
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

    # basic I/O, which are usually valid for OUTPUT models.
    def set_mass(self, pid: int, value: float) -> None:
        """Set mass value."""
        self.slha["MASS", pid] = float(value)

    @overload
    def mass(self, pid: int) -> float:
        ...

    @overload
    def mass(self, pid: int, default: float) -> float:
        ...

    @overload
    def mass(self, pid: int, default: None) -> Optional[float]:
        ...

    def mass(self, pid: int, default: Any = RAISE_ERROR) -> Optional[float]:
        """Return MASS value, which is possibly negative."""
        if (mass := self.get_float("MASS", pid, default=None)) is not None:
            return mass  # stored mass
        elif default is None:
            return default
        elif isinstance(default, SupportsFloat):
            return float(default)
        raise KeyError(f"MASS {pid} not found")

    def width(self, pid: int) -> Optional[float]:
        """Return the width of specified particle, or None."""
        try:
            return self.slha.decays[pid].width
        except KeyError:
            return None

    def br_list(self, pid: int) -> Optional[MutableMapping[Sequence[int], float]]:
        """Return the list of branching ratios of a particle if provided."""
        try:
            decay = self.slha.decays[pid]
        except KeyError:
            return None
        return dict(decay.items_br())
