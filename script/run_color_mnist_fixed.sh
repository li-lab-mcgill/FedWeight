#!/bin/bash
# eicu
env=${1}
conda_env=${2}
pip_env=${3}

declare -a experiment="color_mnist"
declare -a all_tasks=("color_mnist")
declare -a all_made_hiddens=(\"392\" \"500\")
declare -a all_made_num_masks=(10 1)
declare -a all_made_samples=(10)
declare -a all_made_resample_every=(1 5)
declare -a all_made_natural_ordering=(True False)
declare -a all_focal_alpha=(null)
declare -a all_focal_gamma=(null)
declare -a all_batch_size=(64)
declare -a bias_init_prior_prob=null
declare -a fed_weight_method=fed_weight_method_avg # Other options: fed_weight_method_sgd
declare -a target_hospital_id="target"
declare -a algorithm="both"
declare -a total_seed=3
declare -a test_with_bootstrap=False
declare -a init_global_model=null

for task in "${all_tasks[@]}"
do
	for made_hiddens in "${all_made_hiddens[@]}"
	do
		for made_num_masks in "${all_made_num_masks[@]}"
		do
			for made_samples in "${all_made_samples[@]}"
			do
				for made_resample_every in "${all_made_resample_every[@]}"
				do
					for made_natural_ordering in "${all_made_natural_ordering[@]}"
					do
						for focal_alpha in "${all_focal_alpha[@]}"
						do
							for focal_gamma in "${all_focal_gamma[@]}"
							do
								for batch_size in "${all_batch_size[@]}"
								do
									sbatch --gres=gpu:2 -c 4 --mem=32G -t 72:00:00 ./fed_weight_eicu.sh $env $experiment $task 5.0 1.01 $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id $algorithm $total_seed $test_with_bootstrap $init_global_model $batch_size "$conda_env" "$pip_env"
									sbatch --gres=gpu:2 -c 4 --mem=32G -t 72:00:00 ./fed_weight_eicu.sh $env $experiment $task 3.0 1.01 $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id $algorithm $total_seed $test_with_bootstrap $init_global_model $batch_size "$conda_env" "$pip_env"
									sbatch --gres=gpu:2 -c 4 --mem=32G -t 72:00:00 ./fed_weight_eicu.sh $env $experiment $task 1.0 1.01 $made_hiddens $made_num_masks $made_samples $made_resample_every $made_natural_ordering $focal_alpha $focal_gamma $bias_init_prior_prob $fed_weight_method $target_hospital_id $algorithm $total_seed $test_with_bootstrap $init_global_model $batch_size "$conda_env" "$pip_env"
								done
							done
						done
					done
				done
			done
		done
	done
done