#!/bin/bash

# Define base directory - replace with your actual path
BASE_DIR="/home/davidbai/multiagent_helper_llmp"
cd $BASE_DIR

# Array of experiment configurations
declare -a experiments=(
    "--output-file 10_23_2_agent_gpt-4o_.txt --summary-file 10_23_2_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2001"
    "--output-file 10_23_3_agent_gpt-4o_.txt --summary-file 10_23_3_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2002"
    "--output-file 10_23_4_agent_gpt-4o_.txt --summary-file 10_23_4_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model \"gpt-4o\" --run 2003"
    "--output-file 10_23_5_agent_gpt-4o_.txt --summary-file 10_23_5_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model \"gpt-4o\" --run 2004"
    "--output-file 10_23_2_agent_o1-mini_.txt --summary-file 10_23_2_agent_o1-mini_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"o1-mini\" --run 2005"
    "--output-file 10_23_3_agent_o1-mini_.txt --summary-file 10_23_3_agent_o1-mini_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"o1-mini\" --run 2006"
    "--output-file 10_23_4_agent_o1-mini_.txt --summary-file 10_23_4_agent_o1-mini_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model \"o1-mini\" --run 2007"
    "--output-file 10_23_5_agent_o1-mini_.txt --summary-file 10_23_5_agent_o1-mini_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model \"o1-mini\" --run 2008"
    "--output-file 10_23_until_gpt-4o_.txt --summary-file 10_23_until_gpt-4o_summary.txt --python-script helper_script_until_n.py --model \"gpt-4o\" --run 2009"
    "--output-file 10_23_until_o1-mini_.txt --summary-file 10_23_until_o1-mini_summary.txt --python-script helper_script_until_n.py --model \"o1-mini\" --run 2010"
    "--output-file 10_23_choose_gpt-4o_.txt --summary-file 10_23_choose_gpt-4o_summary.txt --python-script helper_script_choose_n.py --model \"gpt-4o\" --run 2011"
    "--output-file 10_23_choose_o1-mini_.txt --summary-file 10_23_choose_o1-mini_summary.txt --python-script helper_script_choose_n.py --model \"o1-mini\" --run 2012"
)

# Activate conda environment
source ~/.bashrc  # Make sure conda is available
conda activate llm_pddl

# Launch each experiment as a separate srun job
for exp in "${experiments[@]}"; do
    # Extract run number for job naming
    run_num=$(echo $exp | grep -o 'run [0-9]*' | cut -d' ' -f2)
    
    # Launch the job using srun
    # Using 4 CPUs, 8GB memory, and 3 hours time limit per job
    echo "Launching experiment run $run_num..."
    srun \
        --job-name="exp_${run_num}" \
        --output="slurm_exp_${run_num}_%j.out" \
        --error="slurm_exp_${run_num}_%j.err" \
        --cpus-per-task=4 \
        --mem=8G \
        --time=3:00:00 \
        ./run_experiments.sh $exp &
    
    # Add a small delay between job submissions to prevent overwhelming the scheduler
    sleep 2
done

# Wait for all jobs to be submitted
wait

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."