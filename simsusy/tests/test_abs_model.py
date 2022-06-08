"""Unit test for AbsModel class."""

import logging
import pathlib
import unittest
from typing import Optional

import pytest
import yaslha

from simsusy.abs_model import AbsModel

logger = logging.getLogger("test_info")


def assert_almost_equal(actual: Optional[float], expected: float) -> bool:
    """Return if actual is almost equal to expected."""
    return actual is not None and actual == pytest.approx(expected)


class TestAbsModelInitialization(unittest.TestCase):
    """Test the initialization of the AbsModel class."""

    def setUp(self):
        self.working_dir = pathlib.Path(__file__).parent
        self.paths = [
            self.working_dir / "sample.slha",
            self.working_dir / "mssm.slha",
            self.working_dir / "mssm.slha2",
        ]

    def test_init_with_pathlib_path(self):
        for path in self.paths:
            slha = AbsModel(path)
            assert slha.mass(6) is not None
            assert_almost_equal(slha.mass(6), 175)

    def test_init_with_path_string(self):
        for path in self.paths:
            slha = AbsModel(path)
            assert slha.mass(6) is not None
            assert_almost_equal(slha.mass(6), 175)

    def test_init_with_slha_content(self):
        for path in self.paths:
            slha = AbsModel(path)
            assert slha.mass(6) is not None
            assert_almost_equal(slha.mass(6), 175)

    def test_init_with_non_existing_files(self):
        with pytest.raises(FileNotFoundError):
            AbsModel(pathlib.Path("a_not_existing_file"))


class TestAbsModelWithGenericInput(unittest.TestCase):
    """Test AbsModel with a standard input."""

    def setUp(self):
        self.working_dir = pathlib.Path(__file__).parent
        self.slha = AbsModel(self.working_dir / "sample.slha")

    def test_block_with_single_arg(self):
        block = self.slha.block("OneArgBlock")
        assert isinstance(block, yaslha.slha.Block)

        assert block[1] == 10
        assert block[2] == -20
        assert block[3] == 0

        assert block.get(10, default=None) is None
        assert block.get(12345, default=None) is None

        assert_almost_equal(block[11], -1522.2)
        assert_almost_equal(block[12], 250)
        # NOTE: pyslha does not support fortran-type notation 1.000d3 etc.
        # assert_almost_equal(block[13], 0.02)
        # assert_almost_equal(block[14], -0.003)

    def test_block_with_two_args(self):
        block = self.slha.block("DoubleArgBlock")
        assert isinstance(block, yaslha.slha.Block)

        assert block[1, 1] == 1
        assert block[1, 2] == 2
        assert block[2, 1] == 2
        assert block[2, 2] == 4

    def test_block_without_arg(self):
        block = self.slha.block("noargblocka")
        assert isinstance(block, yaslha.slha.Block)
        assert_almost_equal(block.q, 123456.789)
        assert_almost_equal(block[None], 3.1415926535)
        assert_almost_equal(block.get(None, default=-1), 3.1415926535)

        block = self.slha.block("noargblockb")
        assert isinstance(block, yaslha.slha.Block)
        assert_almost_equal(block.q, 123456.789)
        assert block[None] == 0
        assert block.get(None, default=-1) == 0

    def test_block_with_unusual_content(self):
        return NotImplemented  # TODO: do we really have to deal with this?
        block = self.slha.block("unusualcase")
        assert isinstance(block, yaslha.slha.Block)
        assert block[1] == "some calculator returns"
        assert block[2] == "these kind of error messages"
        assert block[3] == "which of course is not expected in slha format."

    def test_get(self):
        assert self.slha.get_int("OneArgBlock", 1) == 10
        assert self.slha.get_int("DoubleArgBlock", (2, 2)) == 4
        assert_almost_equal(self.slha.get_float("noargblocka", None), 3.1415926535)

        assert self.slha.get_int("OneArgBlock", 123456) is None
        assert self.slha.get_int("NotExistingBlock", 1) is None
        assert self.slha.get_int_assert("OneArgBlock", 123456, default=789) == 789
        assert self.slha.get_int_assert("NotExistingBlock", 1, default=789) == 789

    def test_mass(self):
        assert self.slha.mass(6) == 175
        assert self.slha.mass(12345) is None

    def test_width(self):
        assert_almost_equal(self.slha.width(6), 1.45899677)
        assert_almost_equal(self.slha.width(1000021), 13.4988503)
        assert_almost_equal(self.slha.width(1000005), 10.7363639)
        assert self.slha.width(9876543) is None

    def test_br_list(self):
        assert self.slha.br_list(123) is None

        brs = self.slha.br_list(6)
        assert isinstance(brs, dict)
        assert len(brs) == 1
        assert_almost_equal(brs[5, 24], 1)  # key is sorted by pid (negative first!)

        brs = self.slha.br_list(1000021)
        assert isinstance(brs, dict)
        assert len(brs) == 2
        assert_almost_equal(brs[-1, 1000001], 0.0217368689)
        assert_almost_equal(brs[-1000001, 1], 0.0217368689)

        brs = self.slha.br_list(1000005)
        assert isinstance(brs, dict)
        assert len(brs) == 8
        assert_almost_equal(0.001, brs[-3, -2, 1])
        assert_almost_equal(0.002, brs[-3, -2, 1, 4])
        assert_almost_equal(0.003, brs[-3, -2, 1, 4, 5])
        assert_almost_equal(0.004, brs[-3, -2, 1, 4, 5, 6])
