#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

# Array of experiment configurations with domains
declare -a experiments=(
    "--output-file 1_15_2_agent_gpt-4o_termes.txt --summary-file 1_15_2_agent_gpt-4o_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2001 --domains termes --tasks 1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    "--output-file 1_15_2_agent_gpt-4o_blocksworld.txt --summary-file 1_15_2_agent_gpt-4o_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2003 --domains blocksworld --tasks 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
    "--output-file 1_15_2_agent_gpt-4o_grippers.txt --summary-file 1_15_2_agent_gpt-4o_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2000 --domains grippers --tasks 4,5,6,11"

    "--output-file 1_15_3_agent_gpt-4o_termes.txt --summary-file 1_15_3_agent_gpt-4o_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2002 --domains termes --tasks 1,19,20"
    "--output-file 1_15_3_agent_gpt-4o_grippers.txt --summary-file 1_15_3_agent_gpt-4o_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2006 --domains grippers --tasks 1,3,4,5,6,11,12,13,14,15,16,17,18,19,20"
    "--output-file 1_15_3_agent_gpt-4o_blocks.txt --summary-file 1_15_3_agent_gpt-4o_blocks_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2008 --domains blocksworld"
    "--output-file 1_15_3_agent_gpt-4o_tyreworld.txt --summary-file 1_15_3_agent_gpt-4o_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2009 --domains tyreworld --tasks 1,5,7,12,13,14,15,16,17,18,19,20"
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
           --mem=16G \
           --time=8:00:00 \
           --qos=general \
	   --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
