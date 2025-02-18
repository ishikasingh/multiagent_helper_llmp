#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

declare -a experiments=(
    # Blocksworld tasks
    "--output-file 2_12_2a_blocksworld.txt --summary-file 2_12_2a_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model 'gpt-4o' --run 1030 --time-limit 500 --domains blocksworld"
    
    # Barman tasks
   "--output-file 2_12_2a_barman.txt --summary-file 2_12_2a_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model 'gpt-4o' --run 1031 --time-limit 500 --domains barman-enabled"
    
    # Grippers tasks    
    "--output-file 2_12_2a_grippers.txt --summary-file 2_12_2a_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model 'gpt-4o' --run 1032 --time-limit 500 --domains grippers"

    #Tyreworld tasks
    "--output-file 2_12_2a_tyreworld.txt --summary-file 2_12_2a_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model 'gpt-4o' --run 1033 --time-limit 500 --domains tyreworld"

    # Termes tasks
    "--output-file 2_12_2a_termes.txt --summary-file 2_12_2a_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model 'gpt-4o' --run 1034 --time-limit 500 --domains termes"
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
           --time=16:00:00 \
           --qos=general \
           --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
