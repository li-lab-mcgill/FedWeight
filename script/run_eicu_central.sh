#!/bin/bash
# eicu
env=${1}
conda_env=${2}
pip_env=${3}

declare -a experiment="eicu_central"
declare -a all_tasks=("death" "ventilator" "sepsis")
declare -a all_focal_alpha=(0.85)
declare -a all_focal_gamma=(2.0)
declare -a batch_size=64
declare -a bias_init_prior_prob=0.85
declare -a fed_weight_method=fed_weight_method_avg # Other options: fed_weight_method_sgd
declare -a all_target_hospital_id=("167" "420" "199" "458" "252")
declare -a total_seed=100
declare -a test_with_bootstrap=True

# Unweighted
for target_hospital_id in "${all_target_hospital_id[@]}"; do
    for task in "${all_tasks[@]}"; do
        for focal_alpha in "${all_focal_alpha[@]}"; do
            for focal_gamma in "${all_focal_gamma[@]}"; do
                estimator="made"
                made_epochs=0
                made_hiddens="1"
                made_num_masks=0
                made_samples=0
                made_resample_every=0
                made_natural_ordering=False
                made_learning_rate=0
                made_weight_decay=0
                vae_hiddens="1"
                vae_learning_rate=0
                topics=0
                sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task null null $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id unweighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
            done
        done
    done
done