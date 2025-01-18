#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

#Array of experiment configurations with domains - only missing 4-agent tasks
declare -a experiments=(
    # Termes missing tasks
    "--output-file 1_17_4a_termes.txt --summary-file 1_17_4a_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 2022 --domains termes --tasks 3,5"
     # Termes missing tasks
    "--output-file 1_17_4a_termes_2.txt --summary-file 1_17_4a_termes_2_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 2027 --domains termes --tasks 13,14,15,16,17,18,19,20"
    
    # Tyreworld missing tasks (split into two runs due to large number)
    "--output-file 1_17_4a_tyreworld_1.txt --summary-file 1_17_4a_tyreworld_1_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 2023 --domains tyreworld --tasks 12,13,14,15"
    "--output-file 1_17_4a_tyreworld_2.txt --summary-file 1_17_4a_tyreworld_2_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 2024 --domains tyreworld --tasks 16,17,18,19,20"
    
    # Barman missing tasks
    "--output-file 1_17_4a_barman.txt --summary-file 1_17_4a_barman_2_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 2026 --domains barman-enabled --tasks 19,20"
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
           --cpus-per-task=2 \
           --mem=32G \
           --time=16:00:00 \
           --qos=general \
           --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
