#!/bin/bash

# Base directory where your code is located
BASE_DIR="/home/davidbai/multiagent_helper_llmp"

# Array of experiment configurations with domains
# declare -a experiments=(
#     # Termes Multi (3-5 agents)
#     "python solve_pddl_MA.py --time-limit 1000 --domain termes-multi"
#     "python solve_pddl_MA.py --time-limit 1000 --domain termes-multi-3"
#     "python solve_pddl_MA.py --time-limit 1000 --domain termes-multi-4"
#     "python solve_pddl_MA.py --time-limit 1000 --domain termes-multi-5"

#     # Blocksworld Multi (3-5 agents)
#     "python solve_pddl_MA.py --time-limit 1000 --domain blocksworld-multi"
#     "python solve_pddl_MA.py --time-limit 1000 --domain blocksworld-multi-3"
#     "python solve_pddl_MA.py --time-limit 1000 --domain blocksworld-multi-4"
#     "python solve_pddl_MA.py --time-limit 1000 --domain blocksworld-multi-5"

#     # Tyreworld (3-5 agents)
#     "python solve_pddl_MA.py --time-limit 1000 --domain tyreworld-multi"
#     "python solve_pddl_MA.py --time-limit 1000 --domain tyreworld-multi-3"
#     "python solve_pddl_MA.py --time-limit 1000 --domain tyreworld-multi-4"
#     "python solve_pddl_MA.py --time-limit 1000 --domain tyreworld-multi-5"

#     # Grippers (3-5 agents)
#     "python solve_pddl_MA.py --time-limit 1000 --domain grippers-multi"
#     "python solve_pddl_MA.py --time-limit 1000 --domain grippers-multi-3"
#     "python solve_pddl_MA.py --time-limit 1000 --domain grippers-multi-4"
#     "python solve_pddl_MA.py --time-limit 1000 --domain grippers-multi-5"

#     # Barman (3-5 agents)
#     "python solve_pddl_MA.py --time-limit 1000 --domain barman-multi"
#     "python solve_pddl_MA.py --time-limit 1000 --domain barman-multi-3"
#     "python solve_pddl_MA.py --time-limit 1000 --domain barman-multi-4"
#     "python solve_pddl_MA.py --time-limit 1000 --domain barman-multi-5"
# )

declare -a experiments=(
    "python solve_pddl_MA.py --time-limit 1000 --domain tyreworld-multi-3"
)

# Launch each experiment as a separate sbatch job
for exp in "${experiments[@]}"; do
    # Extract domain name and number of agents for job naming
    domain=$(echo $exp | grep -o 'domain [^ ]*' | cut -d' ' -f2)
    agents=$(echo $exp | grep -o 'num-agents [0-9]*' | cut -d' ' -f2)
    job_name="${domain}_${agents}agents"
    
    # Submit the job
    echo "Submitting experiment for $job_name..."
    sbatch --job-name=$job_name \
           --output=${BASE_DIR}/slurm_${job_name}_%j.out \
           --error=${BASE_DIR}/slurm_${job_name}_%j.err \
           --cpus-per-task=2 \
           --mem=16G \
           --time=6:00:00 \
           --qos=general \
           --exclude ink-gary \
           --wrap="cd ${BASE_DIR} && $exp"
    
    # Add a delay between job submissions
    sleep 2
done

echo "All jobs have been submitted to the cluster."
echo "Use 'squeue -u $USER' to check their status."
