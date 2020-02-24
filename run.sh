#!/bin/bash

#BSUB -J Example
#BSUB -q dgx
#BSUB -R "span[ptile=1] select [ngpus>0] rusage [ngpus_shared=1]"
#BSUB -n 1
#BSUB -o example.out
#BSUB -e example.err
>example.out
>example.err

export CUDA_VISIBLE_DEVICES=1,2,3,4
export PATH=/seu_share/apps/anaconda3/bin:$PATH
export NCCL_P2P_DISABLE=1

python mm-rnns.py
