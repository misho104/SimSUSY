#!/usr/bin/env python

import click
import importlib
import inspect
import pathlib
import types
from logging import basicConfig, getLogger, DEBUG
from typing import Dict  # noqa: F401

import simsusy

__pkgname__ = 'SimSUSY'
__version__ = '0.0.1'
__author__ = 'Sho Iwamoto / Misho'
__license__ = 'MIT'

basicConfig(level=DEBUG)
logger = getLogger(__name__)
cwd = pathlib.Path(__file__).parent.resolve()

calculators = dict()   # type: Dict[str, pathlib.Path]
for calculator_file in cwd.glob('*/*_calculator.py'):
    relative_path = calculator_file.resolve().relative_to(cwd)
    module_name = '.'.join(relative_path.with_suffix('').parts)
    calculators[module_name] = pathlib.Path('simsusy') / relative_path


@click.group(help='Handle the references for high-energy physics',
             context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--debug', is_flag=True, help='Display debug information for exceptions.')
@click.version_option(__version__, '-V', '--version', prog_name=__pkgname__)
@click.pass_context
# @click.option('-v', '--verbose', is_flag=True, default=False, help="Show verbose output")
def simsusy_main(context, **kwargs):
    if context.obj is None:
        context.obj = dict()
    context.obj['DEBUG'] = kwargs['debug'] if 'debug' in kwargs else False


@simsusy_main.command()
@click.argument('calculator')
@click.argument('input', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(dir_okay=False), required=False)
@click.pass_context
def run(context, calculator, input, output):
    if calculator not in calculators:
        logger.error(f'calculator "{calculator}" not found.\n\nAvailable calculators are:')
        max_length = max(len(name) for name in calculators.keys())
        for name, path in calculators.items():
            logger.error(f'\t{name:<{max_length}}\t({path})')
        exit(1)
    try:
        mod = importlib.import_module(f'simsusy.{calculator}')
    except ModuleNotFoundError as e:
        logger.error(f'Calculator {calculator} cannot be imported.\n')
        if context.obj['DEBUG']:
            raise e
        else:
            logger.error(f'Following exception is caught: ' + str(e))
            logger.error(f'Run with --debug option to see stack trace.')
        exit(1)
    if not(isinstance(mod, types.ModuleType) and
           all(inspect.isclass(c) for c in [mod.Calculator, mod.Input, mod.Model]) and
           issubclass(mod.Calculator, simsusy.abs_calculator.AbsCalculator) and
           issubclass(mod.Input, simsusy.abs_model.AbsModel) and
           issubclass(mod.Model, simsusy.abs_model.AbsModel)
           ):
        logger.error(f'Calculator {calculator} imported but invalid.\n')
        if context.obj['DEBUG']:
            logger.error('Debug information:')
            logger.error(f'\tCalculator\t{mod.Calculator}')
            logger.error(f'\tInput\t\t{mod.Input}')
            logger.error(f'\tModel(Output)\t{mod.Model}')
        else:
            logger.error(f'Run with --debug option to see information.')
        exit(1)

    input_obj = mod.Input(input)
    calc_obj = mod.Calculator(input_obj)
    calc_obj.calculate()
    if output:
        calc_obj.output.write(output)
    else:
        calc_obj.output.write()
