#!/bin/bash
num_topics=${1}

source ../../../fed_weight/venv/bin/activate
python3 fedavg_etm.py --num_topics ${num_topics}