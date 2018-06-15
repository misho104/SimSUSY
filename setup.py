#!env python
from setuptools import setup
import re
import ast


_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('simsusy/simsusy.py', 'rb') as f:
    version_match = _version_re.search(f.read().decode('utf-8'))
    version = str(ast.literal_eval(version_match.group(1))) if version_match else '?.?.?'

setup(
    name='simsusy',
    version=version,
    packages=['simsusy'],
    install_requires=['click', 'pyslha', 'numpy'],
    entry_points={
        'console_scripts': 'generate_spectrum = simsusy.simsusy.generate_spectrum'
    },
    tests_require=['nose', 'coverage', 'mypy', 'flake8'],
    test_suite='nose.collector',
    classifiers=[
        'Environment :: Console',
        'Programming Language :: Python',
    ],
)
