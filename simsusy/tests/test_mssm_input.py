import logging
import pathlib
import unittest

from simsusy.mssm.input import MSSMInput

logger = logging.getLogger("test_info")


class TestMSSMInputInitialization(unittest.TestCase):
    def setUp(self):
        self.working_dir = pathlib.Path(__file__).parent
        self.slha1 = self.working_dir / "mssm.slha.in"
        self.slha2 = self.working_dir / "mssm.slha2.in"

    def test_init(self):
        assert MSSMInput(self.slha1)
