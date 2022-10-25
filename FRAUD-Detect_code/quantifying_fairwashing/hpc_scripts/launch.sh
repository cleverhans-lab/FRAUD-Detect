#!/bin/bash


models=(AdaBoost DNN RF)
klBounds=(0.005 0.01 0.03 0.05 0.07 0.1 0.2)
datasets=(adult_income compas default_credit marketing)


for dataset in "${datasets[@]}" 
do
    for model in "${models[@]}" 
    do
        for kl in ${klBounds[@]}
        do	    
            sbatch main.sh $model $kl $dataset
            sleep 2 
        done
    done
done