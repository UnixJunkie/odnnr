#!/bin/bash

set -x

# |epochs|
for e in `echo 1 2 5 10 20 50 100`; do
    # ./model --train data/Boston_regr_train.csv \
    #         --test data/Boston_regr_test.csv \
    #         --epochs $e --no-plot
    # ./model --no-plot --train data/Boston_regr_train.csv \
    #         --epochs $e --arch 64/64/64/64 --NxCV 10 -np 10
    # hidden layer size
    for l in 32 64 96 128; do
        # number of hidden layers: 1 to 5
        echo ./model --no-plot --train data/Boston_regr_train.csv \
                --epochs $e --arch $l --NxCV 10 -np 10
        echo ./model --no-plot --train data/Boston_regr_train.csv \
                --epochs $e --arch $l/$l --NxCV 10 -np 10
        echo ./model --no-plot --train data/Boston_regr_train.csv \
                --epochs $e --arch $l/$l/$l --NxCV 10 -np 10
        echo ./model --no-plot --train data/Boston_regr_train.csv \
                --epochs $e --arch $l/$l/$l/$l --NxCV 10 -np 10
        echo ./model --no-plot --train data/Boston_regr_train.csv \
                --epochs $e --arch $l/$l/$l/$l/$l --NxCV 10 -np 10
    done
done
