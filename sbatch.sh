#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

# Array of experiment configurations with domains
declare -a experiments=(
    "--output-file 1_16_2a_blocksworld.txt --summary-file 1_16_2a_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --run 2003 --domains blocksworld --tasks 10,11"
    "--output-file 1_16_2a_grippers.txt --summary-file 1_16_2a_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --run 2000 --domains grippers --tasks 4,5,6,11"

    "--output-file 1_16_3a_termes.txt --summary-file 1_16_3a_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --run 2002 --domains termes --tasks 1,4"
    "--output-file 1_16_3a_grippers.txt --summary-file 1_16_3a_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --run 2006 --domains grippers --tasks 1,3,4,5,6,11,13,14,15,16,17,18,19,20"
    "--output-file 1_16_3a_blocks.txt --summary-file 1_16_3a_blocks_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --run 2008 --domains blocksworld --tasks 7,10"
    "--output-file 1_16_3a_tyreworld.txt --summary-file 1_16_3a_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --run 2009 --domains tyreworld --tasks 1,16,17,18,19,20"

    "--output-file 1_16_4a_blocksworld.txt --summary-file 1_16_4a_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --run 2010 --domains blocksworld"
    "--output-file 1_16_4a_grippers.txt --summary-file 1_16_4a_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --run 2011 --domains grippers"
    "--output-file 1_16_4a_termes.txt --summary-file 1_16_4a_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --run 2012 --domains termes"
    "--output-file 1_16_4a_tyreworld.txt --summary-file 1_16_4a_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --run 2013 --domains tyreworld"
    "--output-file 1_16_4a_barman.txt --summary-file 1_16_4a_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --run 2014 --domains barman-enabled"

    "--output-file 1_16_5a_blocksworld.txt --summary-file 1_16_5a_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --run 2015 --domains blocksworld"
    "--output-file 1_16_5a_grippers.txt --summary-file 1_16_5a_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --run 2016 --domains grippers"
    "--output-file 1_16_5a_termes.txt --summary-file 1_16_5a_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --run 2017 --domains termes"
    "--output-file 1_16_5a_tyreworld.txt --summary-file 1_16_5a_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --run 2018 --domains tyreworld"
    "--output-file 1_16_5a_barman.txt --summary-file 1_16_5a_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --run 2019 --domains barman-enabled"
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
           --time=10:00:00 \
           --qos=general \
	   --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
