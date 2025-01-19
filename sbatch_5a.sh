#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

#Array of experiment configurations with domains - only missing 5-agent tasks
declare -a experiments=(
    # Tyreworld missing tasks (split into two runs)
    "--output-file 1_17_5a_tyreworld.txt --summary-file 1_17_5a_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2028 --domains tyreworld --tasks 12,15,17,20"
     "--output-file 1_17_5a_termes.txt --summary-file 1_17_5a_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 2031 --domains termes --tasks 5,6,7,8,9,10,16,17,18,19,20"
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
           --cpus-per-task=8 \
           --mem=32G \
           --time=24:00:00 \
           --qos=general \
           --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
