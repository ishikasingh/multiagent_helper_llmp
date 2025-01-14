#!/bin/bash

# Define base directory - replace with your actual path
BASE_DIR="/nfs1/davidbai/multiagent_helper_llmp"
cd $BASE_DIR

# Array of experiment configurations
declare -a experiments=(
    "--output-file 1_13_2_agent_gpt-4o_.txt --summary-file 1_13_2_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2001"
    "--output-file 1_13_3_agent_gpt-4o_.txt --summary-file 1_13_3_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2002"
    "--output-file 1_13_4_agent_gpt-4o_.txt --summary-file 1_13_4_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model \"gpt-4o\" --run 2003"
    "--output-file 1_13_5_agent_gpt-4o_.txt --summary-file 1_13_5_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model \"gpt-4o\" --run 2004"
)

# Activate conda environment
source ~/.bashrc  # Make sure conda is available
conda activate llm_pddl

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
        --time=6:00:00 \
        --nodelist=ink-lucy \
        ./run_experiments.sh $exp &
    
    # Add a small delay between job submissions to prevent overwhelming the scheduler
    sleep 2
done

# Wait for all jobs to be submitted
wait

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."