"""Helper classes and methods for general spectrum generators."""


import logging
import os
import pathlib
import shutil
import subprocess
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import colorama
import coloredlogs
import toml
import yaslha
from yaslha.slha import SLHA

from simsusy.contrib.data_formats import GM2CalcOutput, MicromegasOutput
from simsusy.contrib.temp_slha import TempSLHA

# Logger setup
logger = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)8s %(message)s")
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
BLUE = colorama.Fore.BLUE
RESET = colorama.Style.RESET_ALL


def separator(message: str) -> None:
    """Print a message as a separator."""
    print("")
    print("#" * 80)
    print(f"# {message}{' '*max(1, 77 - len(message))}#")
    print("#" * 80)


ConfigurationType = Mapping[str, Mapping[str, str]]
PathLike = Union[str, pathlib.Path]
PathLikeOrSLHA = Union[str, pathlib.Path, SLHA]


class Config:
    """Class that knows how to run other programs."""

    _config_default: ConfigurationType = toml.loads(
        """
[simsusy]
path       = "simsusy"
calculator = "mssm.tree_calculator"
[gm2calc]
path       = ""  # path to gm2calc.x
[sdecay]
path       = ""  # path to SDECAY executable ('run')
[micromegas]
make       = "make"
mssm_dir   = ""  # path to MSSM directory within micrOMEGAs
source     = ""  # path to C-file that should be run by micrOMEGAs
exec_name  = "spec_gen"
[sindarin]
converter  = ""   # path to convert_slha_to_sindarin.py
ufo_model  = ""   # path to UFO model directory
"""
    )

    @staticmethod
    def run_process(command, to_print=True, **kwargs):
        # type: (Sequence[PathLikeOrSLHA], bool, Any) -> Tuple[int, str]
        """
        Run a process and return the exit code and output.

        Arguments in command can be SLHA object, for which temporary files are
        created.
        """
        temporary_files: List[TempSLHA] = []

        def filter_with_temp_slha(x: PathLikeOrSLHA) -> str:
            if isinstance(x, SLHA):
                temporary_files.append(TempSLHA(x))
                temporary_files[-1].__enter__()
                path = temporary_files[-1].path
                assert path
                return path
            else:
                return str(x)

        command_str = [filter_with_temp_slha(x) for x in command]

        # actual run
        logger.info("%s%s%s", BLUE, " ".join(command_str), RESET)
        process = subprocess.Popen(
            command_str,
            stdout=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            **kwargs,
        )
        assert process and process.stdout
        if to_print:
            print(colorama.Style.DIM)
        lines: List[str] = []
        for line in process.stdout:
            if to_print:
                print(line, end="")
            lines.append(line)
        return_code = process.wait()

        # make sure to close all temporary objects...
        for t in temporary_files:
            t.__exit__(None, None, None)

        # post-process
        if to_print:
            print(RESET)
        if return_code != 0:
            logger.error("Run failed with exit code %d.", return_code)
            logger.info(process.__dict__)
            exit()
        return return_code, "".join(lines)

    def __init__(self, config_str: str) -> None:
        self._config: ConfigurationType = toml.loads(config_str)

    def config(self, k1: str, k2: str) -> str:
        """Get a configuration value."""
        if k1 in self._config:
            if k2 in self._config[k1]:
                return self._config[k1][k2]
        if k1 in self._config_default:
            if k2 in self._config_default:
                v = self._config_default[k1][k2]
                logger.info("Use default value %s/%s = %s", k1, k2, v)
                return v
        # We should only accept keys known by _config_default.
        logger.error("Configuration key %s/%s not found.", k1, k2)
        exit(1)

    def config_path(self, k1: str, k2: str = "path") -> Optional[pathlib.Path]:
        """Get a configuration path."""
        if value := self.config(k1, k2):
            return pathlib.Path(value).expanduser()
        else:
            return None

    def run_simsusy(self, input_path, output_path, v1=False):
        # type: (PathLikeOrSLHA, PathLike, bool) -> None
        """Run simsusy."""
        command: List[PathLikeOrSLHA] = [
            self.config_path("simsusy") or "",
            "run",
            self.config("simsusy", "calculator"),
            input_path,
            output_path,
        ]
        if any(not s for s in command):
            logger.warning("Invalid configuration for simsusy.")
            return
        if v1:
            command.insert(3, "--v1")
        self.run_process(command)

    def _setup_micromegas(self) -> Optional[Tuple[pathlib.Path, pathlib.Path]]:
        """Return path to compiled executable of micrOMEGAs."""
        make = self.config("micromegas", "make")
        directory = self.config_path("micromegas", "mssm_dir")
        source = self.config_path("micromegas", "source")
        exec_name = self.config("micromegas", "exec_name")

        if not shutil.which(make):
            logger.error("Make executable '%s' not found.", make)
            return None
        elif not directory or not directory.is_dir():
            logger.error("micrOMEGAs path '%s' not found.", directory)
            return None
        elif not source or not source.is_file():
            logger.error("Source for micrOMEGAs '%s' not found.", source)
            return None

        exec_path = directory / exec_name
        exec_source = exec_path.with_suffix(source.suffix)
        # compile
        logger.info("Copy %s to %s", source, exec_source)
        shutil.copyfile(source, exec_source)
        logger.info("Compile analysis code for micrOMEGAs")
        command: List[PathLike] = [make, "-C", directory, f"main={exec_source.name}"]
        self.run_process(command, to_print=False)

        # check
        if shutil.which(exec_path) is None:
            logger.error("Compiled file %s not found; compile failed?", exec_path)
            return None
        return (directory, exec_path)

    def run_micromegas(self, slha1: PathLikeOrSLHA) -> Optional[MicromegasOutput]:
        """Run micrOMEGAs."""
        path = self._setup_micromegas()
        if not path:
            return None
        # micrOMEGAs needs full path
        if isinstance(slha1, SLHA):
            with TempSLHA(slha1) as f:
                command = [path[1], f.path.expanduser().resolve()]
                _, output = self.run_process(command, cwd=path[0])
        else:
            command = [path[1], pathlib.Path(slha1).resolve()]
            _, output = self.run_process(command, cwd=path[0])
        return MicromegasOutput(output)

    def run_gm2calc(self, slha1: PathLikeOrSLHA) -> Optional[GM2CalcOutput]:
        """Run GM2Calc."""
        gm2calc_path = self.config_path("gm2calc")
        if not gm2calc_path:
            logger.warning("GM2Calc not configured.")
            return None
        _, version = self.run_process([gm2calc_path, "--version"])
        if isinstance(slha1, SLHA):
            slha1_obj = slha1
        else:
            slha1_obj = yaslha.parse_file(slha1)
        slha1_obj["GM2CalcConfig", 0] = 1  # request "DETAILED" output
        with TempSLHA(slha1_obj) as f:
            command: List[PathLike] = [gm2calc_path, f"--slha-input-file={f.path}"]
            _, output = self.run_process(command)
        return GM2CalcOutput(output, version)

    SDECAY_IN = pathlib.Path(__file__).with_name("sdecay.in")
    SDECAY_SLHA = "SD_leshouches.in"
    SDECAY_OUT = "sdecay_slha.out"

    def run_sdecay(self, slha1: PathLikeOrSLHA) -> Optional[SLHA]:
        """Return SLHA object containing SDecay output."""
        sdecay = self.config_path("sdecay")
        if not sdecay or not shutil.which(sdecay):
            logger.warning("SDecay executable '%s' not found.", sdecay)
            return None
        elif not self.SDECAY_IN.is_file():
            logger.warning("SDecay input file '%s' not found.", self.SDECAY_IN)
            return None
        elif self.SDECAY_IN.read_text().find("SDECAY INPUT FILE") == -1:
            logger.error("SDecay input file '%s' seems invalid.", self.SDECAY_IN)
            return None

        if isinstance(slha1, SLHA):
            yaslha.dump_file(slha1, self.SDECAY_SLHA)
        else:
            shutil.copy(slha1, self.SDECAY_SLHA)
        # add dummy block, otherwise SDecay complains.
        with open(self.SDECAY_SLHA, "a") as f:
            f.write("Block DUMMY #\n     1     0.00000000E+00   #\n")
        copied_in = pathlib.Path(".", self.SDECAY_IN.name)
        shutil.copy(self.SDECAY_IN, copied_in)
        self.run_process([sdecay])
        result = yaslha.parse_file(self.SDECAY_OUT)
        os.remove(copied_in)
        os.remove(self.SDECAY_SLHA)
        os.remove(self.SDECAY_OUT)

        # fine-tune the output
        result["DCINFO"].head.pre_comment = ["#"]
        for d in result.decays.values():
            d.head.pre_comment = ["#"]
        return result
