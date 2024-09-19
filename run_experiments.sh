#!/bin/bash

verbose=false
# blocksworld, barman, grippers, termes, tyreworld
domains=("tyreworld" "barman")  # Add your domains here
num_tasks=20
max_jobs=4  # Adjust this based on your system's capabilities
num_agents=3  # Default number of agents
run_number=100 # Default run number
parallel_execution=false  # Default to sequential execution

# Function to print usage
print_usage() {
    echo "Usage: $0 [-v|--verbose] [-j|--jobs <num_jobs>] [-a|--agents <num_agents>] [-r|--run <run_number>] [-s|--sequential]"
    echo "  -v, --verbose    Enable verbose output"
    echo "  -j, --jobs       Set maximum number of parallel jobs (default: 4)"
    echo "  -a, --agents     Set number of agents (default: 3)"
    echo "  -r, --run        Set run number (default: 102)"
    echo "  -s, --sequential Run experiments sequentially (default: parallel)"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            verbose=true
            shift
            ;;
        -j|--jobs)
            max_jobs=$2
            shift 2
            ;;
        -a|--agents)
            num_agents=$2
            shift 2
            ;;
        -r|--run)
            run_number=$2
            shift 2
            ;;
        -s|--sequential)
            parallel_execution=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Function to run a single experiment
run_experiment() {
    domain=$1
    task_id=$2
    output=$(python helper_script_n_agents.py --run $run_number --domain "$domain" --time-limit 30 --task_id $task_id --num_agents $num_agents)
    
    if $verbose; then
        echo "$output"
    fi

    single_agent_cost=$(echo "$output" | grep '\[singleagent\]' | grep -oP 'cost \K[0-9.]+')
    final_cost=$(echo "$output" | grep 'True' | awk '{print $(NF-1)}')
    
    if (( $(echo "$final_cost < $single_agent_cost" | bc -l) )); then
        echo "SUCCESS:$domain:$task_id:$final_cost:$single_agent_cost"
    else
        echo "FAILURE:$domain:$task_id:$final_cost:$single_agent_cost"
    fi
}

# Function to process results
process_result() {
    local line=$1
    IFS=':' read -r status domain task_id final_cost single_agent_cost <<< "$line"
    
    if [ "$status" == "SUCCESS" ]; then
        ((total_count++))
        ((domain_counts[$domain]++))
        echo "Task $task_id ($domain): Success ($final_cost < $single_agent_cost)"
    else
        echo "Task $task_id ($domain): Failure ($final_cost >= $single_agent_cost)"
    fi
}

# Initialize variables
declare -A domain_counts
total_count=0

# Run experiments
if $parallel_execution; then
    echo "Running experiments in parallel..."
    export -f run_experiment
    export verbose num_agents run_number
    results=$(parallel -j $max_jobs run_experiment {1} {2} ::: "${domains[@]}" ::: $(seq 1 $num_tasks))
    
    echo "$results" | while read -r line; do
        process_result "$line"
    done
else
    echo "Running experiments sequentially..."
    for domain in "${domains[@]}"; do
        for task_id in $(seq 1 $num_tasks); do
            result=$(run_experiment "$domain" "$task_id")
            process_result "$result"
        done
    done
fi

# Print summary
echo -e "\nSummary:"
echo "--------"
for domain in "${domains[@]}"; do
    count=${domain_counts[$domain]:-0}
    echo "$domain: $count successful out of $num_tasks"
done

echo -e "\nTotal number of successful experiments across all domains: $total_count out of $((${#domains[@]} * num_tasks))"
echo "Number of agents used: $num_agents"
echo "Run number: $run_number"
echo "Execution mode: $(if $parallel_execution; then echo "Parallel"; else echo "Sequential"; fi)"