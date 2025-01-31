#!/bin/bash
# eicu
env=${1}
conda_env=${2}
pip_env=${3}

declare -a experiment="eicu"
declare -a all_tasks=("death" "ventilator" "sepsis" "length")
declare -a all_estimators=("made" "vae" "vqvae")
declare -a all_vae_hiddens=(\"16,16\")
declare -a all_vae_learning_rate=(0.00005)
declare -a all_topics=(10)
declare -a all_made_epochs=(200)
declare -a all_made_hiddens=(\"268,268\")
declare -a all_made_num_masks=(1)
declare -a all_made_samples=(10)
declare -a all_made_resample_every=(20)
declare -a all_made_natural_ordering=(False)
declare -a all_made_learning_rate=(0.00005)
declare -a all_made_weight_decay=(0.001)
declare -a all_focal_alpha=(0.85)
declare -a all_focal_gamma=(2.0)
declare -a batch_size=64
declare -a bias_init_prior_prob=0.85
declare -a fed_weight_method=fed_weight_method_avg # Other options: fed_weight_method_sgd
declare -a all_target_hospital_id=("167" "420" "199" "458" "252")
declare -a total_seed=100
declare -a test_with_bootstrap=True

# Unweighted
for ((i = 0; i < 10; i++)); do
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
done

# Weighted
 for estimator in "${all_estimators[@]}"; do
     if [ "$estimator" == "lda" ]; then
         for target_hospital_id in "${all_target_hospital_id[@]}"; do
             for task in "${all_tasks[@]}"; do
                 for topics in "${all_topics[@]}"; do
                     for focal_alpha in "${all_focal_alpha[@]}"; do
                         for focal_gamma in "${all_focal_gamma[@]}"; do
                             # LDA
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
#                             sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.1 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                             sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.5 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
                             sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 1.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                             sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 3.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                             sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 5.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
                         done
                     done
                 done
             done
         done
     elif [ "$estimator" == "made" ]; then
         for target_hospital_id in "${all_target_hospital_id[@]}"; do
             for task in "${all_tasks[@]}"; do
                 for made_epochs in "${all_made_epochs[@]}"; do
                     for made_hiddens in "${all_made_hiddens[@]}"; do
                         for made_num_masks in "${all_made_num_masks[@]}"; do
                             for made_samples in "${all_made_samples[@]}"; do
                                 for made_resample_every in "${all_made_resample_every[@]}"; do
                                     for made_natural_ordering in "${all_made_natural_ordering[@]}"; do
                                         for made_learning_rate in "${all_made_learning_rate[@]}"; do
                                             for made_weight_decay in "${all_made_weight_decay[@]}"; do
                                                 for focal_alpha in "${all_focal_alpha[@]}"; do
                                                     for focal_gamma in "${all_focal_gamma[@]}"; do
                                                         # MADE
                                                         vae_hiddens="1"
                                                         vae_learning_rate=0
                                                         topics=0
#                                                         sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.001 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                                         sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.01 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                                         sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.1 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                                         sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.05 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                                         sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.25 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
                                                         sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 1.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                                         sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 3.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                                         sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 5.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
                                                     done
                                                 done
                                             done
                                         done
                                     done
                                 done
                             done
                         done
                     done
                 done
             done
         done
     elif [ "$estimator" == "vae" ]; then
         for target_hospital_id in "${all_target_hospital_id[@]}"; do
             for task in "${all_tasks[@]}"; do
                 for vae_hiddens in "${all_vae_hiddens[@]}"; do
                     for vae_learning_rate in "${all_vae_learning_rate[@]}"; do
                         for focal_alpha in "${all_focal_alpha[@]}"; do
                             for focal_gamma in "${all_focal_gamma[@]}"; do
                                 # VAE
                                 made_epochs=0
                                 made_hiddens="1"
                                 made_num_masks=0
                                 made_samples=0
                                 made_resample_every=0
                                 made_natural_ordering=False
                                 made_learning_rate=0
                                 made_weight_decay=0
                                 topics=0
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.001 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.01 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.1 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.05 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.25 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 1.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 3.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
                             done
                         done
                     done
                 done
             done
         done
     else
         for target_hospital_id in "${all_target_hospital_id[@]}"; do
             for task in "${all_tasks[@]}"; do
                 for vae_hiddens in "${all_vae_hiddens[@]}"; do
                     for vae_learning_rate in "${all_vae_learning_rate[@]}"; do
                         for focal_alpha in "${all_focal_alpha[@]}"; do
                             for focal_gamma in "${all_focal_gamma[@]}"; do
                                 # VQVAE
                                 made_epochs=0
                                 made_hiddens="1"
                                 made_num_masks=0
                                 made_samples=0
                                 made_resample_every=0
                                 made_natural_ordering=False
                                 made_learning_rate=0
                                 made_weight_decay=0
                                 topics=0
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.001 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.01 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.1 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.05 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 0.25 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 1.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
#                                 sbatch --gres=gpu:1 --account=def-liyue_gpu -c 4 --mem=20G -t 12:00:00 ./fed_weight_eicu.sh $env $experiment $task 3.0 1.0 $made_epochs $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $made_learning_rate $made_weight_decay $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id weighted $total_seed $test_with_bootstrap $batch_size $estimator $vae_hiddens $vae_learning_rate $topics "$conda_env" "$pip_env"
                             done
                         done
                     done
                 done
             done
         done
     fi
 done
