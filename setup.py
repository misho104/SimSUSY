#!env python
from setuptools import setup
import re
import ast


_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('simsusy/simsusy.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(f.read().decode('utf-8')).group(1)))

setup(
    name='simsusy',
    version=version,
    packages=['simsusy'],
    install_requires=['click', 'pyslha', 'numpy'],
    entry_points={
        'console_scripts': 'generate_spectrum = simsusy.simsusy.generate_spectrum'
    },
    tests_require=['nose', 'coverage'],
    test_suite='nose.collector',
    classifiers=[
        'Environment :: Console',
        'Programming Language :: Python',
    ],
)
