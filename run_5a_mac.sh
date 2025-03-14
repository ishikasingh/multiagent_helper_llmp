#!/bin/bash

# Run each experiment in the background (in parallel)
# echo "Starting experiment 1..."
# ./run_experiments.sh --output-file 5a_termes_1.txt --summary-file 5a_termes_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4037 --time-limit 250 --domains termes --tasks 9,10,11,12 &

# echo "Starting experiment 2..."
# ./run_experiments.sh --output-file 5a_termes_2.txt --summary-file 5a_termes_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4038 --time-limit 250 --domains termes --tasks 14,15,16 &

# echo "Starting experiment 3..."
# ./run_experiments.sh --output-file 5a_termes_3.txt --summary-file 5a_termes_3_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4039 --time-limit 250 --domains termes --tasks 17,18,19,20 &

echo "Starting experiment 4..."
./run_experiments.sh --output-file 5a_barman.txt --summary-file 5a_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4035 --time-limit 250 --domains barman-enabled --tasks 15,17,18,19,20 &

echo "All experiments have been started in the background."
echo "To check their status, use: ps aux | grep helper_script"
echo "To kill all experiments, use: pkill -f 'helper_script_n_agents.py'"
