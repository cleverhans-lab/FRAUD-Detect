#!/bin/bash
datasets=(adult_income compas default_credit marketing)
models=(AdaBoost DNN RF XgBoost)
rseed=(0 1 2 3 4 5 6 7 8 9)

for dataset in "${datasets[@]}" 
    do
        for r in ${rseed[@]}
            do
                for model in "${models[@]}" 
                    do	
                        FILE="pretrained/${dataset}/${model}_${r}.txt"

                        if [ ! -f $FILE ]; then
                            echo "$FILE not exists."
                        fi

                        
                    done
            done
    done
