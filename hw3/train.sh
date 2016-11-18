#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python clustering.py $1 $2
