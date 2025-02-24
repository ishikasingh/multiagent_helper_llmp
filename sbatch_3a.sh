#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp_equals_n"

declare -a experiments=(

    # # Blocksworld tasks
    "--output-file 3a1_blocksworld.txt --summary-file 3a1_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2001 --time-limit 334 --domains blocksworld --tasks 18"
    
    # # Barman tasks
    # "--output-file 3a1_barman.txt --summary-file 3a1_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2002 --time-limit 334 --domains barman-enabled --tasks 18,19,20"
    
    # # Grippers tasks
    "--output-file 3a1_grippers.txt --summary-file 3a1_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2003 --time-limit 334 --domains grippers --tasks 8"

    # # Tyreworld tasks
    "--output-file 3a1_tyreworld.txt --summary-file 3a1_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2004 --time-limit 334 --domains tyreworld --tasks 11,12,13,14,15,16,17,18,19,20"

    # # Termes tasks
    "--output-file 3a1_termes.txt --summary-file 3a1_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2005 --time-limit 334 --domains termes --tasks 11,12,13,14,15,16,17,18,19,20"

    # # Blocksworld tasks
    # "--output-file 3a2_blocksworld.txt --summary-file 3a2_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2006 --time-limit 334 --domains blocksworld --tasks 2"
    
    # # Barman tasks
    # "--output-file 3a2_barman.txt --summary-file 3a2_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2007 --time-limit 334 --domains barman-enabled --tasks 18,19,20"
    
    # # Grippers tasks
    # "--output-file 3a2_grippers.txt --summary-file 3a2_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2008 --time-limit 334 --domains grippers --tasks 4,5,11"

    # # Tyreworld tasks
    "--output-file 3a2_tyreworld.txt --summary-file 3a2_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2009 --time-limit 334 --domains tyreworld --tasks 13,14,15,16,17,18,19,20"

    # # Termes tasks
    "--output-file 3a2_termes.txt --summary-file 3a2_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2010 --time-limit 334 --domains termes --tasks 7,8,9,10,11,12,13,14,15,16,17,18,19,20"

    # # Blocksworld tasks
    # "--output-file 3a3_blocksworld.txt --summary-file 3a3_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2011 --time-limit 334 --domains blocksworld --tasks 2"
    
    # # Barman tasks
    # "--output-file 3a3_barman.txt --summary-file 3a3_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2012 --time-limit 334 --domains barman-enabled --tasks 18,19,20"
    
    # # Grippers tasks
    # "--output-file 3a3_grippers.txt --summary-file 3a3_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2013 --time-limit 334 --domains grippers --tasks 4,5,8,11"

    # # Tyreworld tasks
    "--output-file 3a3_tyreworld.txt --summary-file 3a3_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2014 --time-limit 334 --domains tyreworld --tasks 12,13,14,15,16,17,18,19,20"

    # # Termes tasks
    "--output-file 3a3_termes.txt --summary-file 3a3_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model 'gpt-4o' --run 2015 --time-limit 334 --domains termes --tasks 14,15,16,17,18,19,20"

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
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 5
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
