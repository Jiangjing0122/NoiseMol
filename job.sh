#!/bin/sh 
# 这个export参数需要修改成自己服务的地址
export PATH=$PATH:/home/jiangj/anaconda3/bin/

mypython=python3

export device=0

CUDA_VISIBLE_DEVICES=$device \
             $mypython main.py 

