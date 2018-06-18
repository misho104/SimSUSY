from nose.tools import raises, ok_, eq_  # noqa: F401
from simsusy.mssm.input import MSSMInput
import pathlib
import logging
import unittest
logger = logging.getLogger('test_info')


class TestMSSMInputInitialization(unittest.TestCase):
    def setUp(self):
        self.working_dir = pathlib.Path(__file__).parent
        self.slha1 = self.working_dir / 'mssm.slha.in'
        self.slha2 = self.working_dir / 'mssm.slha2.in'

    def test_init(self):
        ok_(MSSMInput(self.slha1))
