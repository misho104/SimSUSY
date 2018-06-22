#!env python
from setuptools import setup
import ast
import pathlib
import re


_version_re = re.compile(r'__version__\s+=\s+(.*)')

root = pathlib.Path('simsusy')
with (root / 'simsusy.py').open('rb')as f:
    version_match = _version_re.search(f.read().decode('utf-8'))
    version = str(ast.literal_eval(version_match.group(1))) if version_match else '?.?.?'

packages = ['simsusy']
for subdir in root.iterdir():
    if subdir.is_dir() and subdir.name[0:4] != 'test' and re.match(r'[a-zA-Z]', subdir.name[0]):
        packages.append(str(subdir).replace('/', '.'))

setup(
    name='simsusy',
    version=version,
    packages=packages,
    install_requires=['click', 'pyslha', 'numpy'],
    entry_points={
        'console_scripts': 'simsusy = simsusy.simsusy:simsusy_main'
    },
    tests_require=['nose', 'coverage', 'mypy', 'flake8'],
    test_suite='nose.collector',
    classifiers=[
        'Environment :: Console',
        'Programming Language :: Python',
    ],
)
