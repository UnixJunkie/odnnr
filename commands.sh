#!/bin/bash

set -x

# find best model architecture using early stopping
for l in 64 128 256 512 1024 2048; do
    # hidden layers: 1 to 4
    echo odnnr_model --no-plot --train data/chembl1868_good.pFP \
         --NxCV 5 --epochs 100 --early-stop --arch $l
    echo odnnr_model --no-plot --train data/chembl1868_good.pFP \
         --NxCV 5 --epochs 100 --early-stop --arch $l/$l
    echo odnnr_model --no-plot --train data/chembl1868_good.pFP \
         --NxCV 5 --epochs 100 --early-stop --arch $l/$l/$l
    echo odnnr_model --no-plot --train data/chembl1868_good.pFP \
         --NxCV 5 --epochs 100 --early-stop --arch $l/$l/$l/$l
done
