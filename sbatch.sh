#!/bin/bash

# Array of experiment configurations with domains
declare -a experiments=(
    # Termes domain
    "--output-file 1_13_2_agent_gpt-4o_termes.txt --summary-file 1_13_2_agent_gpt-4o_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2001 --domains termes"
    "--output-file 1_13_3_agent_gpt-4o_termes.txt --summary-file 1_13_3_agent_gpt-4o_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2002 --domains termes"
    
    # Barman domain
    "--output-file 1_13_2_agent_gpt-4o_barman.txt --summary-file 1_13_2_agent_gpt-4o_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2005 --domains barman-enabled"
    "--output-file 1_13_3_agent_gpt-4o_barman.txt --summary-file 1_13_3_agent_gpt-4o_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2006 --domains barman-enabled"
    
    # Grippers domain
    "--output-file 1_13_2_agent_gpt-4o_grippers.txt --summary-file 1_13_2_agent_gpt-4o_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2009 --domains grippers"
    "--output-file 1_13_3_agent_gpt-4o_grippers.txt --summary-file 1_13_3_agent_gpt-4o_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2010 --domains grippers"
    
    # Blocksworld domain
    "--output-file 1_13_2_agent_gpt-4o_blocks.txt --summary-file 1_13_2_agent_gpt-4o_blocks_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2013 --domains blocksworld"
    "--output-file 1_13_3_agent_gpt-4o_blocks.txt --summary-file 1_13_3_agent_gpt-4o_blocks_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2014 --domains blocksworld"
)

# Launch each experiment as a separate srun job
for exp in "${experiments[@]}"; do
    # Extract run number for job naming
    run_num=$(echo $exp | grep -o 'run [0-9]*' | cut -d' ' -f2)
    
    # Launch the job using srun
    # Using 2 CPUs, 8GB memory, and 6 hours time limit per job
    echo "Launching experiment run $run_num..."
    srun \
        --job-name="exp_${run_num}" \
        --output="slurm_exp_${run_num}_%j.out" \
        --error="slurm_exp_${run_num}_%j.err" \
        --cpus-per-task=2 \
        --mem=8G \
        --time=8:00:00 \
        --qos general \
        ./run_experiments.sh $exp &
    
    # Add a small delay between job submissions to prevent overwhelming the scheduler
    sleep 2
done

# Wait for all jobs to be submitted
wait

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."