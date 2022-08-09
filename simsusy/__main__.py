"""Entry point if executed as 'python simsusy'."""

import os
import sys

if __name__ == "__main__":
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)
    sys.path.insert(0, path)
    from simsusy.simsusy import simsusy_main

    sys.exit(simsusy_main())
