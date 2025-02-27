#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

#Array of experiment configurations with domains - only missing 5-agent tasks
declare -a experiments=(
    # Barman tasks (missing 10-20)
    # "--output-file 5a_barman.txt --summary-file 5a_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4035 --time-limit 250 --domains barman-enabled --tasks 15,17,18,19,20"

    # # # Termes tasks (missing 7-20)
    #     "--output-file 5a_termes_1.txt --summary-file 5a_termes_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4037 --time-limit 250 --domains termes --tasks 9,10,11,12"
    #     "--output-file 5a_termes_2.txt --summary-file 5a_termes_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4038 --time-limit 250 --domains termes --tasks 14,15,16"
    #     "--output-file 5a_termes_3.txt --summary-file 5a_termes_3_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4039 --time-limit 250 --domains termes --tasks 17,18,19,20"

    # # Tyreworld tasks (missing 8-20)
    # "--output-file 5a_tyreworld.txt --summary-file 5a_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4037 --time-limit 250 --domains tyreworld --tasks 20"
    
    # # Blocksworld tasks
    # "--output-file 5a2_blocksworld.txt --summary-file 5a2_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4038 --time-limit 200 --domains blocksworld --tasks 4"
    # # # Barman tasks
    # "--output-file 5a2_barman.txt --summary-file 5a2_barman_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4039 --time-limit 200 --domains barman-enabled --tasks 17,18"
    "--output-file 5a2_barman_2.txt --summary-file 5a2_barman_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4040 --time-limit 200 --domains barman-enabled --tasks 19"
    "--output-file 5a2_barman_2b.txt --summary-file 5a2_barman_2b_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4041 --time-limit 200 --domains barman-enabled --tasks 20"
    
    # # # Grippers tasks
    # # "--output-file 5a2_grippers.txt --summary-file 5a2_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4040 --time-limit 200 --domains grippers --tasks 8"

    # #Tyreworld tasks
    # # "--output-file 5a2_tyreworld.txt --summary-file 5a2_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4041 --time-limit 200 --domains tyreworld --tasks 20"

    # # Termes tasks
    #     "--output-file 5a2_termes_1a.txt --summary-file 5a2_termes_1a_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4042 --time-limit 200 --domains termes --tasks 12,13"
        "--output-file 5a2_termes_1b.txt --summary-file 5a2_termes_1b_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4043 --time-limit 200 --domains termes --tasks 14"
    #     # "--output-file 5a2_termes_2.txt --summary-file 5a2_termes_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4043 --time-limit 200 --domains termes --tasks 15,16,17"
        "--output-file 5a2_termes_3a_1.txt --summary-file 5a2_termes_3a_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4044 --time-limit 200 --domains termes --tasks 18"
        "--output-file 5a2_termes_3a_2.txt --summary-file 5a2_termes_3a_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4045 --time-limit 200 --domains termes --tasks 19"
        "--output-file 5a2_termes_3b.txt --summary-file 5a2_termes_3b_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4046 --time-limit 200 --domains termes --tasks 20"
#
    #  # Blocksworld tasks
    # # "--output-file 5a3_blocksworld.txt --summary-file 5a3_blocksworld_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4043 --time-limit 200 --domains blocksworld --tasks 4"
    
    # # Barman tasks
    #     "--output-file 5a3_barman_1.txt --summary-file 5a3_barman_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4046 --time-limit 200 --domains barman-enabled --tasks 15,16"
    #     "--output-file 5a3_barman_2.txt --summary-file 5a3_barman_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4047 --time-limit 200 --domains barman-enabled --tasks 17,18"
    #     "--output-file 5a3_barman_3.txt --summary-file 5a3_barman_3_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4048 --time-limit 200 --domains barman-enabled --tasks 19,20"
    
    # Grippers tasks
    # "--output-file 5a3_grippers.txt --summary-file 5a3_grippers_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4045 --time-limit 200 --domains grippers --tasks 9,10,11,12,13,14,15,16,17,18,19,20"

    #Tyreworld tasks
    # "--output-file 5a3_tyreworld.txt --summary-file 5a3_tyreworld_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4046 --time-limit 200 --domains tyreworld --tasks 12,13,14,15,16,17,18,19,20"

    # Termes tasks
        # "--output-file 5a3_termes_1a.txt --summary-file 5a3_termes_1a_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4050 --time-limit 200 --domains termes --tasks 9,10"
        # "--output-file 5a3_termes_1b_1.txt --summary-file 5a3_termes_1b_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4051 --time-limit 200 --domains termes --tasks 11"
        # "--output-file 5a3_termes_1b_2.txt --summary-file 5a3_termes_1b_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4052 --time-limit 200 --domains termes --tasks 12"
        "--output-file 5a3_termes_2a_1.txt --summary-file 5a3_termes_2a_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4053 --time-limit 200 --domains termes --tasks 14"
        "--output-file 5a3_termes_2a_2.txt --summary-file 5a3_termes_2a_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4054 --time-limit 200 --domains termes --tasks 15"
        # "--output-file 5a3_termes_2b.txt --summary-file 5a3_termes_2b_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4055 --time-limit 200 --domains termes --tasks 16"
        "--output-file 5a3_termes_3a_1.txt --summary-file 5a3_termes_3a_1_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4056 --time-limit 200 --domains termes --tasks 17"
        "--output-file 5a3_termes_3a_2.txt --summary-file 5a3_termes_3a_2_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4057 --time-limit 200 --domains termes --tasks 18"
        # "--output-file 5a3_termes_3b.txt --summary-file 5a3_termes_3b_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model 'gpt-4o' --run 4055 --time-limit 200 --domains termes --tasks 19,20"
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
           --cpus-per-task=16 \
           --mem=32G \
           --time=24:00:00 \
           --wrap="cd ${BASE_DIR} && ./run_experiments.sh $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
