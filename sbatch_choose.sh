#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

# Array of experiment configurations (domain, task combinations)
declare -a experiments=(
    # Barman-enabled
    "--output-file 1_20_c_barman-enabled.txt --summary-file 1_20_c_barman-enabled_summary.txt --python-script helper_script_choose_n.py --num-agents 5 --model 'gpt-4o' --run 3000 --domains barman-enabled"
    
    # Blocksworld
    #"--output-file 1_20_c_blocksworld.txt --summary-file 1_20_c_blocksworld_summary.txt --python-script helper_script_choose_n.py --num-agents 5 --model 'gpt-4o' --run 3001 --domains blocksworld"
    
    # Tyreworld
    "--output-file 1_20_c_tyreworld.txt --summary-file 1_20_c_tyreworld_summary.txt --python-script helper_script_choose_n.py --num-agents 5 --model 'gpt-4o' --run 3002 --domains tyreworld"
    
    # Termes
   #"--output-file 1_20_c_termes.txt --summary-file 1_20_c_termes_summary.txt --python-script helper_script_choose_n.py --num-agents 5 --model 'gpt-4o' --run 3003 --domains termes"
    
    # Grippers
    "--output-file 1_20_c_grippers.txt --summary-file 1_20_c_grippers_summary.txt --python-script helper_script_choose_n.py --num-agents 5 --model 'gpt-4o' --run 3004 --domains grippers"
)

# Launch each experiment as a separate sbatch job
for exp in "${experiments[@]}"; do
    # Extract domain name for job naming
    domain=$(echo $exp | grep -o "domains [^ ]*" | cut -d" " -f2)
    
    # Submit the job
    echo "Submitting experiment for domain $domain..."
    sbatch --job-name=choose_${domain} \
           --output=${BASE_DIR}/slurm_choose_${domain}_%j.out \
           --error=${BASE_DIR}/slurm_choose_${domain}_%j.err \
           --cpus-per-task=4 \
           --mem=32G \
           --time=12:00:00 \
           --qos=general \
           --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
