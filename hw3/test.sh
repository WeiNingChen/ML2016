
#!/bin/bash

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python predict_test_data.py $1 $2 $3
