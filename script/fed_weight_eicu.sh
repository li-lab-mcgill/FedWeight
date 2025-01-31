#!/bin/bash
env=${1}
experiment=${2}
task=${3}
reweight_lambda=${4}
reweight_phi=${5}
made_epochs=${6}
made_hiddens=${7}
made_num_masks=${8}
made_samples=${9}
made_resample_every=${10}
made_natural_ordering=${11}
made_learning_rate=${12}
made_weight_decay=${13}
focal_alpha=${14}
focal_gamma=${15}
bias_init_prior_prob=${16}
fed_weight_method=${17}
target_hospital_id=${18}
algorithm=${19}
total_seed=${20}
test_with_bootstrap=${21}
batch_size=${22}
estimator=${23}
vae_hiddens=${24}
vae_learning_rate=${25}
topics=${26}
conda_env=${27}
pip_env=${28}
simulate_original_path=${29}
simulate_x_source_path=${30}
simulate_y_source_path=${31}
simulate_x_target_path=${32}
simulate_y_target_path=${33}
init_global_model=${34}

# 1. Load the required modules
if [[ -z "$conda_env" && -z "$pip_env" ]]; then
    echo "Please specify environment!"
fi

if [[ -n "$conda_env" ]]; then
    echo ${conda_env}
    module load anaconda/3
    conda activate ${conda_env}
fi

if [[ -n "$pip_env" ]]; then
    echo ${pip_env}
    source ../${pip_env}/bin/activate
fi

if [ "$experiment" == "simulation" ]; then

    fl_hiddens=null
    learning_rate=0.0001
    weight_decay=0.0001
    num_rounds=500

    # 2. Run experiments
    # Simulation - MILA
    python3 ../main.py \
        env=${env} \
        experiment=${experiment} \
        experiment.task=${task} \
        experiment.reweight_lambda=${reweight_lambda} \
        experiment.reweight_phi=${reweight_phi} \
        experiment.made_epochs=${made_epochs} \
        experiment.made_hiddens=${made_hiddens} \
        experiment.made_num_masks=${made_num_masks} \
        experiment.made_samples=${made_samples} \
        experiment.made_resample_every=${made_resample_every} \
        experiment.made_natural_ordering=${made_natural_ordering} \
        experiment.made_learning_rate=${made_learning_rate} \
        experiment.made_weight_decay=${made_weight_decay} \
        experiment.focal_alpha=${focal_alpha} \
        experiment.focal_gamma=${focal_gamma} \
        experiment.bias_init_prior_prob=${bias_init_prior_prob} \
        experiment.fed_weight_method=${fed_weight_method} \
        experiment.algorithm=${algorithm} \
        experiment.total_seed=${total_seed} \
        experiment.test_with_bootstrap=${test_with_bootstrap} \
        experiment.simulate_original_path=${simulate_original_path} \
        experiment.simulate_x_source_path=${simulate_x_source_path} \
        experiment.simulate_y_source_path=${simulate_y_source_path} \
        experiment.simulate_x_target_path=${simulate_x_target_path} \
        experiment.simulate_y_target_path=${simulate_y_target_path} \
        experiment.init_global_model=${init_global_model} \
        experiment.target_hospital_id=${target_hospital_id} \
        experiment.batch_size=${batch_size} \
        experiment.fl_hiddens=${fl_hiddens} \
        experiment.learning_rate=${learning_rate} \
        experiment.weight_decay=${weight_decay} \
        experiment.num_rounds=${num_rounds} \
        experiment.density_estimator=${estimator} \
        experiment.vae_hiddens=${vae_hiddens} \
        experiment.vae_learning_rate=${vae_learning_rate} \
        experiment.lda_topics=${topics}

    exit 0

elif [ "$experiment" == "eicu" ]; then

  if [ "$task" == "death" ]; then
      if [ "$algorithm" == 'weighted' ]; then
          fl_hiddens=\"16,16\"
          learning_rate=1e-4
          weight_decay=0.0001
          num_rounds=500
      else
          fl_hiddens=\"16,16\"
          learning_rate=1e-5
          weight_decay=0.0001
          num_rounds=500
      fi
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "ventilator" ]; then
      if [ "$algorithm" == 'weighted' ]; then
          fl_hiddens=\"16\"
          learning_rate=1e-4
          weight_decay=0.0001
          num_rounds=500
      else
          fl_hiddens=\"16\"
          learning_rate=1e-5
          weight_decay=0.0001
          num_rounds=500
      fi
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "sepsis" ]; then
      if [ "$algorithm" == 'weighted' ]; then
          fl_hiddens=\"16\"
          learning_rate=5e-4
          weight_decay=0.0001
          num_rounds=500
      else
          fl_hiddens=\"16\"
          learning_rate=5e-5
          weight_decay=0.0001
          num_rounds=500
      fi
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "length" ]; then
      if [ "$algorithm" == 'weighted' ]; then
          fl_hiddens=\"64,64\"
          learning_rate=5e-5
          weight_decay=0.005
          num_rounds=500
      else
          fl_hiddens=\"64,64\"
          learning_rate=1e-5
          weight_decay=0.005
          num_rounds=500
      fi
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab_los.csv"
      eicu_time_path="/eicu_mimic_lab_time_los.csv"
  fi

elif [ "$experiment" == "eicu_central" ]; then

  if [ "$task" == "death" ]; then
      # FL:
      fl_hiddens=\"16,16\"
      learning_rate=5e-5
      weight_decay=0.0001
      num_rounds=500
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "ventilator" ]; then
      fl_hiddens=\"16\"
      learning_rate=5e-5
      weight_decay=0.0001
      num_rounds=500
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "sepsis" ]; then
      fl_hiddens=\"16\"
      learning_rate=1e-4
      weight_decay=0.0001
      num_rounds=500
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "unsupervised" ]; then
      fl_hiddens=\"268\"
      learning_rate=0.0001
      weight_decay=0.0001
      num_rounds=500
      total_seed=10
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "length" ]; then
      if [ "$target_hospital_id" == 199 ]; then
          fl_hiddens=\"64,64\"
          learning_rate=1e-5
          weight_decay=0.005
          num_rounds=500
      elif [ "$target_hospital_id" == 420 ]; then
          fl_hiddens=\"64,64\"
          learning_rate=0.0005
          weight_decay=0.005
          num_rounds=500
      elif [ "$target_hospital_id" == 458 ]; then
          fl_hiddens=\"16,16\"
          learning_rate=1e-5
          weight_decay=0.005
          num_rounds=500
      else
          fl_hiddens=\"64,64\"
          learning_rate=0.001
          weight_decay=0.005
          num_rounds=500
      fi
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab_los.csv"
      eicu_time_path="/eicu_mimic_lab_time_los.csv"
  fi

elif [ "$experiment" == "mimic" ]; then

  if [ "$task" == "death" ]; then
      # FL:
      if [ "$algorithm" == 'weighted' ]; then
          fl_hiddens=\"64\"
          learning_rate=1e-5
          weight_decay=0.0001
          num_rounds=500
      else
          fl_hiddens=\"64\"
          learning_rate=5e-6
          weight_decay=0.0001
          num_rounds=500
      fi
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
      init_global_model=null
  elif [ "$task" == "ventilator" ]; then
      if [ "$algorithm" == 'weighted' ]; then
          fl_hiddens=\"16\"
          learning_rate=0.0001
          weight_decay=0.0001
          num_rounds=500
      else
          fl_hiddens=\"16\"
          learning_rate=0.00005
          weight_decay=0.0001
          num_rounds=500
      fi
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "sepsis" ]; then
      if [ "$algorithm" == 'weighted' ]; then
          fl_hiddens=\"16\"
          learning_rate=0.0005
          weight_decay=0.0001
          num_rounds=500
      else
          fl_hiddens=\"16\"
          learning_rate=5e-5
          weight_decay=0.0001
          num_rounds=500
      fi
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "length" ]; then
      if [ "$algorithm" == 'weighted' ]; then
          fl_hiddens=\"64\"
          learning_rate=1e-5
          weight_decay=0.0001
          num_rounds=500
      else
          fl_hiddens=\"64\"
          learning_rate=1e-6
          weight_decay=0.0001
          num_rounds=500
      fi
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab_los.csv"
      eicu_time_path="/eicu_mimic_lab_time_los.csv"
  fi
elif [ "$experiment" == "mimic_central" ]; then

  if [ "$task" == "death" ]; then
      # FL:
      fl_hiddens=\"64\"
      learning_rate=0.0005
      weight_decay=0.003
      num_rounds=500
      # Non-FL:
  #    fl_hiddens=\"16\"
  #    learning_rate=0.0001
  #    weight_decay=0.001
  #    num_rounds=500
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "ventilator" ]; then
      fl_hiddens=\"16\"
      learning_rate=0.001
      weight_decay=0.001
      num_rounds=500
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "sepsis" ]; then
      fl_hiddens=\"16\"
      learning_rate=0.0001
      weight_decay=0.003
      num_rounds=500
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab.csv"
      eicu_time_path="/eicu_mimic_lab_time.csv"
  elif [ "$task" == "length" ]; then
      fl_hiddens=\"16,16\"
      learning_rate=1e-5
      weight_decay=0.005
      num_rounds=500
      init_global_model=null
      eicu_path="/eicu_mimic_drug_lab_los.csv"
      eicu_time_path="/eicu_mimic_lab_time_los.csv"
  fi
fi


# 2. Run experiments
python3 ../main.py \
    env=${env} \
    experiment=${experiment} \
    experiment.task=${task} \
    experiment.reweight_lambda=${reweight_lambda} \
    experiment.reweight_phi=${reweight_phi} \
    experiment.made_epochs=${made_epochs} \
    experiment.made_hiddens=${made_hiddens} \
    experiment.made_num_masks=${made_num_masks} \
    experiment.made_samples=${made_samples} \
    experiment.made_resample_every=${made_resample_every} \
    experiment.made_natural_ordering=${made_natural_ordering} \
    experiment.made_learning_rate=${made_learning_rate} \
    experiment.made_weight_decay=${made_weight_decay} \
    experiment.focal_alpha=${focal_alpha} \
    experiment.focal_gamma=${focal_gamma} \
    experiment.bias_init_prior_prob=${bias_init_prior_prob} \
    experiment.fed_weight_method=${fed_weight_method} \
    experiment.algorithm=${algorithm} \
    experiment.total_seed=${total_seed} \
    experiment.test_with_bootstrap=${test_with_bootstrap} \
    experiment.init_global_model=${init_global_model} \
    experiment.target_hospital_id=${target_hospital_id} \
    experiment.batch_size=${batch_size} \
    experiment.fl_hiddens=${fl_hiddens} \
    experiment.learning_rate=${learning_rate} \
    experiment.weight_decay=${weight_decay} \
    experiment.num_rounds=${num_rounds} \
    experiment.density_estimator=${estimator} \
    experiment.vae_hiddens=${vae_hiddens} \
    experiment.vae_learning_rate=${vae_learning_rate} \
    experiment.lda_topics=${topics} \
    experiment.eicu_path=${eicu_path} \
    experiment.eicu_time_path=${eicu_time_path}
