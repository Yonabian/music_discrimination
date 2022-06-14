#!/bin/bash 

for ((imgNum=0; imgNum<600; imgNum+=7));
do
nohup python -u wav2spectrum.py -n $imgNum > $imgNum.log&
wait
done