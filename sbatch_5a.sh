#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

#Array of experiment configurations with domains - only missing 5-agent tasks
declare -a experiments=(
    # Grippers missing tasks (split into two runs)
    "--output-file 1_17_5a_grippers_1.txt --summary-file 1_17_5a_grippers_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2025 --domains grippers --tasks 7,11,12,15,16"
    "--output-file 1_17_5a_grippers_2.txt --summary-file 1_17_5a_grippers_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2026 --domains grippers --tasks 17,18,19,20"
    
    # Tyreworld missing tasks (split into two runs)
    "--output-file 1_17_5a_tyreworld_1.txt --summary-file 1_17_5a_tyreworld_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2027 --domains tyreworld --tasks 4,6,10,12,15"
    "--output-file 1_17_5a_tyreworld_2.txt --summary-file 1_17_5a_tyreworld_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2028 --domains tyreworld --tasks 16,17,18,19,20"

    "--output-file 1_17_5a_barman.txt --summary-file 1_17_5a_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2029 --domains barman-enabled --tasks 3,5,6,7,8,9,10,11,12"
    "--output-file 1_17_5a_barman_2.txt --summary-file 1_17_5a_barman_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2030 --domains barman-enabled --tasks 13,14,15,16,17,18,19,20"

     "--output-file 1_17_5a_termes.txt --summary-file 1_17_5a_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2031 --domains termes --tasks 1,2,3,4,5,6,7,8,9,10"
    "--output-file 1_17_5a_termes_2.txt --summary-file 1_17_5a_termes_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2032 --domains termes --tasks 11,12,13,14,15,16,17,18,19,20"
)

# Launch each experiment as a separate sbatch job
for exp in "${experiments[@]}"; do
    # Extract run number for job naming
    run_num=$(echo $exp | grep -o 'run [0-9]*' | cut -d' ' -f2)
    
    # Submit the job
    echo "Submitting experiment run $run_num..."
    sbatch --job-name=exp_${run_num} \
           --output=${BASE_DIR}/slurm_exp_${run_num}_%j.out \
           --error=${BASE_DIR}/slurm_exp_${run_num}_%j.err \
           --cpus-per-task=4 \
           --mem=32G \
           --time=8:00:00 \
           --qos=general \
           --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
