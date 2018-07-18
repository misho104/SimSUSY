#!env bash

# TODO: these "tests" should be merged into the other tests....
OUTDIR=simsusy/tests/outputs/
rm $OUTDIR/*
for CALC in tree_calculator mg5_tree_calculator; do
  python simsusy run --v1 mssm.$CALC simsusy/tests/mssm.slha.in  > $OUTDIR/mssm1.$CALC.v1 2> $OUTDIR/mssm1.$CALC.v1.log
  python simsusy run      mssm.$CALC simsusy/tests/mssm.slha.in  > $OUTDIR/mssm1.$CALC.v2 2> $OUTDIR/mssm1.$CALC.v2.log
done
for CALC in tree_calculator; do
  python simsusy run      mssm.$CALC simsusy/tests/mssm.slha2.in > $OUTDIR/mssm2.$CALC.v2 2> $OUTDIR/mssm2.$CALC.log
done
