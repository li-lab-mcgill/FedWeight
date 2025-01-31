#!/bin/bash
# eicu
env=${1}
conda_env=${2}
pip_env=${3}

declare -a experiment="mnist"
declare -a all_tasks=("color_mnist")
declare -a all_reweight_lambda=(1.0 3.0 5.0)
declare -a fed_weight_method=fed_weight_method_avg # Other options: fed_weight_method_sgd
declare -a all_target_hospital_id=("target" "0" "1" "2")
declare -a all_algorithm=("unweighted" "weighted")
declare -a test_with_bootstrap=True
declare -a init_global_model="/init_mnist_model.pt"

for target_hospital_id in "${all_target_hospital_id[@]}"
do
    for task in "${all_tasks[@]}"
    do
        for algorithm in "${all_algorithm[@]}"
        do
            for reweight_lambda in "${all_reweight_lambda[@]}"
            do
                sbatch --gres=gpu:1 -c 4 --mem=20G -t 24:00:00 ./fed_weight_eicu.sh $env $experiment $task $reweight_lambda null null null $fed_weight_method $target_hospital_id $algorithm $test_with_bootstrap $init_global_model "$conda_env" "$pip_env"
            done
        done
    done
done