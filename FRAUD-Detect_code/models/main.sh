#!/bin/bash
#datasets=(default_credit)
#models=(AdaBoost)
#rseed=(8)

datasets=(adult_income compas default_credit marketing)
models=(AdaBoost DNN RF XgBoost)
rseed=(0 1 2 3 4 5 6 7 8 9)

for dataset in "${datasets[@]}" 
    do
        for r in ${rseed[@]}
            do
                for model in "${models[@]}" 
                    do	
                        # pretraining the black-box model
                        python train_models.py --dataset=$dataset --model_class=$model --nbr_evals=10 --rseed=$r
                        # getting the predictions for the suing group and the test set
                        sleep 2
                        python get_labels.py --dataset=$dataset --model_class=$model --rseed=$r
                    done
                # getting the true labels for the suing group and the test set
                python get_true_labels.py --dataset=$dataset --rseed=$r
            done
    done


