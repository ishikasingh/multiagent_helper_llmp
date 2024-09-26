#!/bin/bash

verbose=false
# blocksworld, barman, grippers, termes, tyreworld
domains=("blocksworld" "grippers" "tyreworld" "barman")  # Add your domains here
tasks=(1 2 5 7 10 12 15 17 20)  # Subset of tasks to execute
max_jobs=4  # Adjust this based on your system's capabilities
run_number=101 # Default run number
parallel_execution=false  # Default to sequential execution
output_file=""  # Initialize output file variable

# Function to print usage
print_usage() {
    echo "Usage: $0 [-v|--verbose] [-j|--jobs <num_jobs>] [-a|--agents <num_agents>] [-r|--run <run_number>] [-s|--sequential] [-o|--output <output_file>]"
    echo "  -v, --verbose    Enable verbose output"
    echo "  -j, --jobs       Set maximum number of parallel jobs (default: 4)"
    echo "  -r, --run        Set run number (default: 100)"
    echo "  -s, --sequential Run experiments sequentially (default: parallel)"
    echo "  -o, --output     Specify output file (required)"
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
        -a|--max_agents)
            max_agents=$2
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
        -o|--output)
            output_file=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Check if output file is specified
if [ -z "$output_file" ]; then
    echo "Error: No output file specified. Use -o or --output to specify an output file."
    print_usage
    exit 1
fi

# Function to run a single experiment
run_experiment() {
    local domain=$1
    local task_id=$2
    local verbose_flag=""
    if $verbose; then
        verbose_flag="--verbose"
    fi

    output=$(python helper_script_until_n.py \
        --run "$run_number" \
        --domain "$domain" \
        --time-limit 20 \
        --task_id "$task_id" \
        --max_agents "$max_agents" \
        )
    
    if $verbose; then
        echo "$output"
    fi

    single_agent_cost=$(echo "$output" | grep '\[singleagent\]' | sed -E 's/.*cost ([0-9.]+).*/\1/')
    final_cost=$(echo "$output" | grep 'True' | awk '{print $(NF-1)}')
    
    if [ -n "$single_agent_cost" ] && [ -n "$final_cost" ]; then
        if (( $(echo "$final_cost < $single_agent_cost" | bc -l) )); then
            echo "SUCCESS: $domain: task $task_id: llm_multi $final_cost: single_agent $single_agent_cost"
        else
            echo "FAILURE: $domain: task $task_id: llm_multi $final_cost: single_agent $single_agent_cost"
        fi
    else
        echo "ERROR:$domain: task $task_id: Unable to parse costs"
    fi
}


# Initialize variables
declare -a domain_counts
total_count=0;

# Function to process results
process_result() {
    local line=$1
    IFS=':' read -r status domain task_id final_cost single_agent_cost <<< "$line"
    
    if [ "$status" == "SUCCESS" ]; then
        ((total_count++))
        index=$(printf '%s\n' "${domains[@]}" | grep -n "^${domain}$" | cut -d: -f1)
        ((index--))
        ((domain_counts[index]++))
        echo "Task $task_id ($domain): Success ($final_cost < $single_agent_cost)"
    else
        echo "Task $task_id ($domain): Failure ($final_cost >= $single_agent_cost)"
    fi
}

# Run experiments

if $parallel_execution; then
    echo "Running experiments in parallel..."
    export -f run_experiment
    export verbose num_agents run_number
    parallel -j $max_jobs run_experiment {1} {2} ::: "${domains[@]}" ::: "${tasks[@]}" > "$output_file"
else
    echo "Running experiments sequentially..."
    {
        for domain in "${domains[@]}"; do
            for task_id in "${tasks[@]}"; do
                result=$(run_experiment "$domain" "$task_id")
                echo "$result"
            done
        done
    } > "$output_file"
fi

echo "Results have been written to $output_file"

# Print summary
echo -e "\nSummary:"
echo "--------"
for i in "${!domains[@]}"; do
    domain="${domains[i]}"
    count=${domain_counts[i]:-0}
    echo "$domain: $count successful out of ${#tasks[@]}"
done

echo -e "\nTotal number of successful experiments across all domains: $total_count out of $((${#domains[@]} * ${#tasks[@]}))"
echo "Number of agents used: $num_agents"
echo "Run number: $run_number"
echo "Execution mode: $(if $parallel_execution; then echo "Parallel"; else echo "Sequential"; fi)"
echo "Output file: $output_file"