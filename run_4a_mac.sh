#!/bin/bash

# Run each experiment one at a time (sequentially)
# echo "Starting experiment 1..."
# ./run_experiments.sh --output-file 4a_termes_remaining_p1.txt --summary-file 4a_termes_remaining_p1_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3002 --time-limit 250 --domains termes --tasks 14,15,16,17 &

# echo "Starting experiment 2..."
# ./run_experiments.sh --output-file 4a2_tyreworld.txt --summary-file 4a2_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3007 --time-limit 250 --domains tyreworld --tasks 20 &

echo "Starting experiment 3..."
./run_experiments.sh --output-file 4a2_termes_p1.txt --summary-file 4a2_termes_p1_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3008 --time-limit 250 --domains termes --tasks 14,15,16,17 &

# echo "Starting experiment 4..."
# ./run_experiments.sh --output-file 4a2_termes_p2.txt --summary-file 4a2_termes_p2_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3015 --time-limit 250 --domains termes --tasks 20 &

echo "All experiments have been completed."