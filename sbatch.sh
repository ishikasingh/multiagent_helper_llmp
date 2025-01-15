#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

# Array of experiment configurations with domains
declare -a experiments=(
    # Termes domain
    "--output-file 1_13_2_agent_gpt-4o_termes.txt --summary-file 1_13_2_agent_gpt-4o_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2001 --domains termes"
    "--output-file 1_13_3_agent_gpt-4o_termes.txt --summary-file 1_13_3_agent_gpt-4o_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2002 --domains termes"
    
    # Barman domain
    "--output-file 1_13_2_agent_gpt-4o_barman.txt --summary-file 1_13_2_agent_gpt-4o_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2003 --domains barman-enabled"
    "--output-file 1_13_3_agent_gpt-4o_barman.txt --summary-file 1_13_3_agent_gpt-4o_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2004 --domains barman-enabled"
    
    # Grippers domain
    "--output-file 1_13_2_agent_gpt-4o_grippers.txt --summary-file 1_13_2_agent_gpt-4o_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2005 --domains grippers"
    "--output-file 1_13_3_agent_gpt-4o_grippers.txt --summary-file 1_13_3_agent_gpt-4o_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2006 --domains grippers"
    
    # Blocksworld domain
    "--output-file 1_13_2_agent_gpt-4o_blocks.txt --summary-file 1_13_2_agent_gpt-4o_blocks_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2007 --domains blocksworld"
    "--output-file 1_13_3_agent_gpt-4o_blocks.txt --summary-file 1_13_3_agent_gpt-4o_blocks_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2008 --domains blocksworld"
    
    # Tyreworld domain (3 agents only)
    "--output-file 1_13_3_agent_gpt-4o_tyreworld.txt --summary-file 1_13_3_agent_gpt-4o_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2009 --domains tyreworld"
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
           --mem=8G \
           --time=8:00:00 \
           --qos=general \
	   --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
