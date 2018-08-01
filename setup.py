#!env python
from setuptools import setup
import ast
import pathlib
import re


with (pathlib.Path('simsusy') / 'simsusy.py').open('rb') as f:
    version_match = re.search(r'__version__\s+=\s+(.*)', f.read().decode('utf-8'))
    version = str(ast.literal_eval(version_match.group(1))) if version_match else '0.0.0'

packages = ['simsusy']
for subdir in pathlib.Path('simsusy').iterdir():
    if subdir.is_dir() \
        and (not subdir.name.startswith('test')) \
        and re.match(r'[a-zA-Z]', subdir.name[0]):
        packages.append(str(subdir).replace('/', '.'))

setup(
    name='simsusy',
    version=version,
    author='Sho Iwamoto / Misho',
    author_email='webmaster@misho-web.com',
    url='https://github.com/misho104/simsusy',
    description='TBW...',
    python_requires='>=3.4',
    license='MIT',
    packages=packages,
    zip_safe=False,   # for default_values.json
    package_data={
        'simsusy': [
            'default_values.json',
        ]},
    install_requires=['click', 'yaslha', 'numpy'],
    dependency_links=[
      'git+https://github.com/misho104/yaslha.git#egg=yaslha'
    ],
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
