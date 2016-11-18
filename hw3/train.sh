#!/bin/bash

KERAS_BACKEND=theano THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python self_train.py $1 $2
