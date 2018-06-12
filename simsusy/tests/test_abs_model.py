from nose.tools import raises, ok_, eq_
from simsusy.abs_model import AbsModel
import pathlib
import logging
import unittest
logger = logging.getLogger('test_info')


class TestAbsModelInitialization(unittest.TestCase):
    def setUp(self):
        self.working_dir = pathlib.Path(__file__).parent
        self.paths = [
            self.working_dir / 'sample.slha',
            self.working_dir / 'mssm.slha',
            self.working_dir / 'mssm.slha2',
        ]

    def test_init_with_pathlib_path(self):
        for path in self.paths:
            slha = AbsModel(path)
            self.assertAlmostEqual(slha.mass(6), 175)

    def test_init_with_path_string(self):
        for path in self.paths:
            slha = AbsModel(str(path))
            self.assertAlmostEqual(slha.mass(6), 175)

    def test_init_with_slha_content(self):
        for path in self.paths:
            slha_content = path.read_text()
            slha = AbsModel(slha_content)
            self.assertAlmostEqual(slha.mass(6), 175)

    @raises(FileNotFoundError)
    def test_init_with_non_existing_files(self):
        AbsModel('a_not_existing_file')

    @raises(IsADirectoryError)
    def test_init_with_directory(self):
        AbsModel(self.working_dir)

    @raises(ValueError)
    def test_init_with_invalid_argument(self):
        AbsModel(['list', 'of', 'string'])


class TestAbsModelWithGenericInput(unittest.TestCase):
    def setUp(self):
        self.working_dir = pathlib.Path(__file__).parent
        self.slha = AbsModel(self.working_dir / 'sample.slha')

    def test_block_with_single_arg(self):
        block = self.slha.block('OneArgBlock')
        ok_(block)

        eq_(block[1], 10)
        eq_(block[2], -20)
        eq_(block[3], 0)

        eq_(block.get(10, default=None), None)
        eq_(block.get(12345, default=None), None)

        self.assertAlmostEqual(block[11], -1522.2)
        self.assertAlmostEqual(block[12], 250)
        # NOTE: pyslha does not support fortran-type notation 1.000d3 etc.
        # self.assertAlmostEqual(block[13], 0.02)
        # self.assertAlmostEqual(block[14], -0.003)

    def test_block_with_two_args(self):
        block = self.slha.block('DoubleArgBlock')
        ok_(block)

        eq_(block[1, 1], 1)
        eq_(block[1, 2], 2)
        eq_(block[2, 1], 2)
        eq_(block[2, 2], 4)

    def test_block_without_arg(self):
        block = self.slha.block('noargblocka')
        ok_(block)
        self.assertAlmostEqual(block.q, 123456.789)
        self.assertAlmostEqual(block[None], 3.1415926535)
        self.assertAlmostEqual(block.get(), 3.1415926535)

        block = self.slha.block('noargblockb')
        ok_(block)
        self.assertAlmostEqual(block.q, 123456.789)
        eq_(block[None], 0)
        eq_(block.get(), 0)

    def test_block_with_unusual_content(self):
        block = self.slha.block('unusualcase')
        ok_(block)
        eq_(block[1], 'some calculator returns')
        eq_(block[2], 'these kind of error messages')
        eq_(block[3], 'which of course is not expected in slha format.')

    def test_get(self):
        eq_(self.slha.get('OneArgBlock', 1), 10)
        eq_(self.slha.get('DoubleArgBlock', (2, 2)), 4)
        self.assertAlmostEqual(self.slha.get('noargblocka', None), 3.1415926535)

        eq_(self.slha.get('OneArgBlock', 123456), None)
        eq_(self.slha.get('NotExistingBlock', 1), None)
        eq_(self.slha.get('OneArgBlock', 123456, 789), 789)
        eq_(self.slha.get('NotExistingBlock', 1, 789), 789)

    def test_mass(self):
        eq_(self.slha.mass(6), 175)
        eq_(self.slha.mass(12345), None)

    def test_width(self):
        self.assertAlmostEqual(self.slha.width(6), 1.45899677)
        self.assertAlmostEqual(self.slha.width(1000021), 13.4988503)
        self.assertAlmostEqual(self.slha.width(1000005), 10.7363639)
        self.assertAlmostEqual(self.slha.width(9876543), None)

    def test_br(self):
        self.assertAlmostEqual(1, self.slha.br(6, 5, 24))
        self.assertAlmostEqual(1, self.slha.br(6, 24, 5))
        self.assertAlmostEqual(0, self.slha.br(6, 5, -24))
        self.assertAlmostEqual(0, self.slha.br(6, -5, -24))

        self.assertAlmostEqual(0.0217368689, self.slha.br(1000021, 1000001, -1))
        self.assertAlmostEqual(0.0217368689, self.slha.br(1000021, 1, -1000001))

        self.assertAlmostEqual(0.001, self.slha.br(1000005, 1, -2, -3))
        self.assertAlmostEqual(0.002, self.slha.br(1000005, 1, -2, -3, 4))
        self.assertAlmostEqual(0.003, self.slha.br(1000005, 1, -2, -3, 4, 5))
        self.assertAlmostEqual(0.004, self.slha.br(1000005, 1, -2, -3, 4, 5, 6))
        self.assertAlmostEqual(0.004, self.slha.br(1000005, 6, 5, 4, 1, -2, -3))
        self.assertAlmostEqual(0, self.slha.br(1000005, 1, -2))

        eq_(self.slha.br(1234567, 8, 9), None)  # NONE for not existing particle
        eq_(self.slha.br(1000005, 8, 9), 0)     # ZERO for not existing decay channel

    def test_br_list(self):
        eq_(self.slha.br_list(123), None)

        brs = self.slha.br_list(6)
        ok_(isinstance(brs, dict))
        eq_(len(brs), 1)
        self.assertAlmostEqual(brs[5, 24], 1)  # key is sorted by pid (negative first!)

        brs = self.slha.br_list(1000021)
        eq_(len(brs), 2)
        self.assertAlmostEqual(brs[-1, 1000001], 0.0217368689)
        self.assertAlmostEqual(brs[-1000001, 1], 0.0217368689)

        brs = self.slha.br_list(1000005)
        eq_(len(brs), 8)
        self.assertAlmostEqual(0.001, brs[-3, -2, 1])
        self.assertAlmostEqual(0.002, brs[-3, -2, 1, 4])
        self.assertAlmostEqual(0.003, brs[-3, -2, 1, 4, 5])
        self.assertAlmostEqual(0.004, brs[-3, -2, 1, 4, 5, 6])
