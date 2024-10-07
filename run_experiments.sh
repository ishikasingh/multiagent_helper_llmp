#!/bin/bash

# Function to print usage information
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --output-file FILE         Specify the output file (default: experiment_results.txt)"
    echo "  --python-script FILE       Specify the Python script to run"
    echo "  --num-agents N1            number of agents"
    echo "  --run NUMBER               Specify the run number"
}

# Default values
OUTPUT_FILE="experiment_results.txt"
PYTHON_SCRIPT=""
# termes "tyreman" "grippers" "barman"
DOMAINS=("blocksworld")
TIME_LIMIT=100
TASK_IDS=(1 2)
NUM_AGENTS=""
RUN=1

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --python-script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        --num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --run)
            RUN="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Function to run an experiment with given parameters
run_experiment() {
    local domain=$1
    local time_limit=$2
    local task_id=$3
    if [[ "$PYTHON_SCRIPT" == "helper_script_n_agents.py" ]]; then
      local num_agents=$4
      echo "Running experiment: Domain=$domain, Time Limit=$time_limit, Task ID=$task_id, Agents=$NUM_AGENTS"
      python "$PYTHON_SCRIPT" \
          --domain "$domain" \
          --time-limit "$time_limit" \
          --task_id "$task_id" \
          --run "$RUN" \
          --num_agents "$num_agents" \
          >> "$OUTPUT_FILE" 2>&1
    else
      echo "Running experiment: Domain=$domain, Time Limit=$time_limit, Task ID=$task_id"
      python "$PYTHON_SCRIPT" \
          --domain "$domain" \
          --time-limit "$time_limit" \
          --task_id "$task_id" \
          --run "$RUN" \
          >> "$OUTPUT_FILE" 2>&1
    fi
    echo "----------------------------------------" >> "$OUTPUT_FILE"
}

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Iterate through various configurations
for domain in "${DOMAINS[@]}"; do
  for task_id in "${TASK_IDS[@]}"; do
    run_experiment "$domain" "$TIME_LIMIT" "$task_id" "$NUM_AGENTS"
  done
done

echo "All experiments completed. Results are in $OUTPUT_FILE"