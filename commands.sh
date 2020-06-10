#!/bin/bash

set -x

# find best model using early stopping
for l in 32 64 96 128; do
    # hidden layers: 1 to 5
    echo odnnr_model --no-plot --train data/Boston_regr_train.csv \
         --epochs 100 --early-stop --arch $l
    echo odnnr_model --no-plot --train data/Boston_regr_train.csv \
         --epochs 100 --early-stop --arch $l/$l
    echo odnnr_model --no-plot --train data/Boston_regr_train.csv \
         --epochs 100 --early-stop --arch $l/$l/$l
    echo odnnr_model --no-plot --train data/Boston_regr_train.csv \
         --epochs 100 --early-stop --arch $l/$l/$l/$l
    echo odnnr_model --no-plot --train data/Boston_regr_train.csv \
         --epochs 100 --early-stop --arch $l/$l/$l/$l/$l
done
