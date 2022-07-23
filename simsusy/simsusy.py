#!/usr/bin/env python

import importlib
import logging
import pathlib
import types
from typing import Any, Dict, KeysView, Optional, ValuesView  # noqa: F401

import click
import coloredlogs

import simsusy.abs_calculator
import simsusy.abs_model

__pkgname__ = "SimSUSY"
__version__ = "0.3.0"
__author__ = "Sho Iwamoto / Misho"
__license__ = "MIT"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
cwd = pathlib.Path(__file__).parent.resolve()


class Calculators:
    """A singleton class to store calculators."""

    def __init__(self) -> None:
        cwd = pathlib.Path(__file__).parent.resolve()
        self.calculators = dict()  # type: Dict[str, pathlib.Path]
        for calculator_file in cwd.glob("*/*_calculator.py"):
            relative_path = calculator_file.resolve().relative_to(cwd)
            module_name = ".".join(relative_path.with_suffix("").parts)
            self.calculators[module_name] = pathlib.Path("simsusy") / relative_path

    @classmethod
    def is_valid(cls, mod: Any) -> bool:
        if not isinstance(mod, types.ModuleType):
            return False
        return (
            issubclass(getattr(mod, "Calculator"), simsusy.abs_calculator.AbsCalculator)
            and issubclass(getattr(mod, "Input"), simsusy.abs_model.AbsModel)
            and issubclass(getattr(mod, "Output"), simsusy.abs_model.AbsModel)
        )

    def __getitem__(self, key: str) -> pathlib.Path:
        return self.calculators.__getitem__(key)

    def get(self, name: str) -> Optional[pathlib.Path]:
        return self.calculators.get(name)

    def guess(self, name: str) -> Optional[pathlib.Path]:
        if name in self.calculators:
            return self.calculators[name]
        if name.count(".") != 1:
            return None
        model, calc = name.lower().split(".")
        candidates = list()
        for i in self.calculators.keys():
            i_model, i_calc = i.lower().split(".")
            if i_model.startswith(model) and i_calc.startswith(calc):
                candidates.append(i)
        if len(candidates) == 1:
            return self.calculators[candidates[0]]
        else:
            if candidates:
                logger.error(f"The calculator specification {name} is ambiguous:")
                for i in candidates:
                    logger.error(f"\t{i}")
            return None

    def keys(self) -> KeysView[str]:
        return self.calculators.keys()

    def values(self) -> ValuesView[pathlib.Path]:
        return self.calculators.values()


@click.group(
    help="Handle the references for high-energy physics",
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.option("--debug", is_flag=True, help="Display debug information for exceptions.")
@click.version_option(__version__, "-V", "--version", prog_name=__pkgname__)
@click.pass_context
# @click.option(
#     "-v", "--verbose", is_flag=True, default=False, help="Show verbose output"
# )
def simsusy_main(context, **kwargs):
    # type: (click.Context, **Dict[str, Any]) -> None
    coloredlogs.install(logger=logging.getLogger(), fmt="%(levelname)8s %(message)s")
    if context.obj is None:
        context.obj = dict()
    context.obj["DEBUG"] = kwargs["debug"] if "debug" in kwargs else False


@simsusy_main.command()
@click.argument("calculator")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False), required=False)
@click.option("--v1", is_flag=True, help="Try to output in SLHA1 format")
@click.pass_context
def run(context, calculator, input, output, v1):
    # type: (click.Context, str, click.Path, Optional[click.Path], bool) -> None
    calculators = Calculators()
    guessed_calculator = calculators.guess(calculator)
    if guessed_calculator is None:
        logger.error(
            f'calculator "{calculator}" not found.\n\nAvailable calculators are:'
        )
        max_length = max(len(name) for name in calculators.keys())
        for name in calculators.keys():
            logger.error(f"\t{name:<{max_length}}\t({calculators[name]})")
        exit(1)
    module_name = ".".join(guessed_calculator.with_suffix("").parts)
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        logger.error(f"Calculator {calculator} cannot be imported.")
        if context.obj["DEBUG"]:
            raise e
        else:
            logger.error("Following exception is caught: " + str(e))
            logger.error("Run with --debug option to see stack trace.")
        exit(1)
    if not calculators.is_valid(mod):
        logger.error(f"Calculator {calculator} imported but invalid.")
        if context.obj["DEBUG"]:
            logger.error(f"Debug information: {mod.__dict__}")
            for i in ["Calculator", "Input", "Output"]:
                logger.error(f"\t{i}\t{getattr(mod, i, '(Not found)')}")
        else:
            logger.error("Run with --debug option to see information.")
        exit(1)
    input_obj = mod.Input(input)
    calc_obj = mod.Calculator(input_obj)
    calc_obj.calculate()
    calc_obj.write_output(output, slha1=v1)
