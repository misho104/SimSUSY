"""Helpers for SLHA objects."""

import tempfile
from typing import Any, Optional

import yaslha.slha


class TempSLHA:
    """Named temporary file for SLHA object."""

    def __init__(self, slha: yaslha.slha.SLHA) -> None:
        self._slha = slha
        self._temp = None  # type: Optional[tempfile._TemporaryFileWrapper[str]]
        pass

    def __enter__(self, **args):
        # type: (Any) -> TempSLHA
        if self._temp:
            raise RuntimeError("TempSLHA is already active.")
        self._temp = tempfile.NamedTemporaryFile(**args).__enter__()
        yaslha.dump_file(self._slha, self._temp.name)
        self._temp.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # type: (Any, Any, Any)->None
        if self._temp:
            self._temp.__exit__(exc_type, exc_value, traceback)
            self._temp = None

    @property
    def path(self) -> Optional[str]:
        """Return the path to the temporary file."""
        return self._temp.name if self._temp else None
