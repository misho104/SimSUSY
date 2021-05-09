|img_ci| |img_cov| |img_license|

SimSUSY: simple SUSY spectrum calculators
=========================================

A framework to call respective "calculators".

A simple example is:

.. code-block:: shell

   simsusy run -c mssm.tree_calculator input.SLHA

where ``input.slha`` is a proper input file.
Note that **calculator is specified by hand**, or should be specified in ``SIMSUSY`` block of the input file.

Remarks / Policies
------------------

* calculators are stored as ``simsusy/MODEL/NAME_calculator.py`` and called as ``MODEL.NAME_calculator``.
* ``SIMSUSY`` block in input files must be "eaten" by the calculator and absent in the output file.
* ``SIMSUSY`` 0–99 are reserved for SimSUSY program.
* ``SIMSUSY`` 100–999 are for each calculator. However, **these values should not alter physical output** (e.g., "loop order" cannot be specified there!). Instead, prepare another calculator to have different physical output.



License
-------

This code set ("software") is licensed to you under |Apache2|_.
See ``LICENSE`` file and ``NOTICE`` file for further information.




.. |Apache2| replace:: the Apache License, version 2.0
.. _Apache2: https://www.apache.org/licenses/LICENSE-2.0

.. |img_ci| image:: https://api.travis-ci.org/misho104/SimSUSY.svg?branch=master
  :height: 16px

.. |img_cov| image:: https://coveralls.io/repos/github/misho104/SimSUSY/badge.svg?branch=master
  :height: 16px

.. |img_license| image:: https://shields.io/badge/license-Apache--2.0-ff25d1
  :height: 16px
  :alt: This package is licensed under the Apache-2.0 License.
