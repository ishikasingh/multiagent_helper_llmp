#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

#Array of experiment configurations with domains - only missing 4-agent tasks
declare -a experiments=(
    # Barman tasks (missing 12-20)
    "--output-file 4a_barman_remaining.txt --summary-file 4a_barman_remaining_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3001 --time-limit 250 --domains barman-enabled --tasks 12 13 14 15 16 17 18 19 20"

    # Termes tasks (missing 7-20)
    "--output-file 4a_termes_remaining.txt --summary-file 4a_termes_remaining_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3002 --time-limit 250 --domains termes --tasks 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

    # Tyreworld tasks (missing 15-20)
    "--output-file 4a_tyreworld_remaining.txt --summary-file 4a_tyreworld_remaining_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3003 --time-limit 250 --domains tyreworld --tasks 15 16 17 18 19 20"
    
    # Blocksworld tasks
    "--output-file 4a2_blocksworld.txt --summary-file 4a2_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3004 --time-limit 250 --domains blocksworld"
    
    # Barman tasks
   "--output-file 4a2_barman.txt --summary-file 4a2_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3005 --time-limit 250 --domains barman-enabled"

    # Grippers tasks
    "--output-file 4a2_grippers.txt --summary-file 4a2_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3006 --time-limit 250 --domains grippers"

    #Tyreworld tasks
    "--output-file 4a2_tyreworld.txt --summary-file 4a2_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3007 --time-limit 250 --domains tyreworld"

    # Termes tasks
    "--output-file 4a2_termes.txt --summary-file 4a2_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3008 --time-limit 250 --domains termes"

     # Blocksworld tasks
    "--output-file 4a3_blocksworld.txt --summary-file 4a3_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3009 --time-limit 250 --domains blocksworld"
    
    # Barman tasks
   "--output-file 4a3_barman.txt --summary-file 4a3_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3010 --time-limit 250 --domains barman-enabled"

    # Grippers tasks
    "--output-file 4a3_grippers.txt --summary-file 4a3_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3011 --time-limit 250 --domains grippers"

    #Tyreworld tasks
    "--output-file 4a3_tyreworld.txt --summary-file 4a3_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3012 --time-limit 250 --domains tyreworld"

    # Termes tasks
    "--output-file 4a3_termes.txt --summary-file 4a3_termes_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model 'gpt-4o' --run 3013 --time-limit 250 --domains termes"

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
           --time=36:00:00 \
           --qos=general \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
