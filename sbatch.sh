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

# Create a temporary directory for job scripts
TEMP_DIR="job_scripts_$(date +%Y%m%d_%H%M%S)"
mkdir -p $TEMP_DIR

# Launch each experiment as a separate sbatch job
for exp in "${experiments[@]}"; do
    # Extract run number for job naming
    run_num=$(echo $exp | grep -o 'run [0-9]*' | cut -d' ' -f2)
    
    # Create a temporary script for this job
    job_script="$TEMP_DIR/job_${run_num}.sh"
    
    # Write the job script
    cat << EOF > "$job_script"
#!/bin/bash
#SBATCH --job-name=exp_${run_num}
#SBATCH --output=slurm_exp_${run_num}_%j.out
#SBATCH --error=slurm_exp_${run_num}_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --qos=general

# Exit on error
set -e

# Run the experiment
./run_experiments.sh $exp
EOF
    
    # Make the script executable
    chmod +x "$job_script"
    
    # Submit the job
    echo "Submitting experiment run $run_num..."
    sbatch "$job_script"
    
    # Add a longer delay between job submissions
    sleep 5
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
echo "Job scripts are stored in $TEMP_DIR"