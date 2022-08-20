|img_ci|_ |img_cov|_ |img_type|_ |img_license|_ |img_black|_

SimSUSY: simple SUSY spectrum calculators
=========================================

A framework to call respective "calculators".

A simple example is:

.. code-block:: shell

   simsusy run mssm.tree_calculator input.slha

where ``input.slha`` is a proper input file and **calculator is specified by hand** as the command-line argument.

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

.. _img_ci: https://github.com/misho104/SimSUSY/actions/workflows/unit-test.yaml
.. |img_ci| image:: https://github.com/misho104/SimSUSY/actions/workflows/unit-test.yaml/badge.svg?branch=master
  :height: 16px

.. _img_cov: https://codecov.io/gh/misho104/SimSUSY
.. |img_cov| image:: https://codecov.io/gh/misho104/SimSUSY/branch/master/graph/badge.svg
  :height: 16px

.. _img_type: https://github.com/misho104/SimSUSY/actions/workflows/type-check.yaml
.. |img_type| image:: https://github.com/misho104/SimSUSY/actions/workflows/type-check.yaml/badge.svg?branch=master
  :height: 16px

.. _img_license: https://github.com/misho104/SimSUSY/blob/master/LICENSE
.. |img_license| image:: https://shields.io/badge/license-Apache--2.0-ff25d1
  :height: 16px
  :alt: This package is licensed under the Apache-2.0 License.

.. _img_black: https://github.com/ambv/black
.. |img_black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :height: 16px
