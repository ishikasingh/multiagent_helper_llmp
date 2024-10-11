#!/bin/bash

# Define the commands
commands=(
    "./run_experiments.sh --output-file 10_10_2_agent_gpt-4o_.txt --summary-file 10_10_2_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"gpt-4o\" --run 2001"
    "./run_experiments.sh --output-file 10_10_3_agent_gpt-4o_.txt --summary-file 10_10_3_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"gpt-4o\" --run 2002"
    "./run_experiments.sh --output-file 10_10_4_agent_gpt-4o_.txt --summary-file 10_10_4_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model \"gpt-4o\" --run 2003"
    "./run_experiments.sh --output-file 10_10_5_agent_gpt-4o_.txt --summary-file 10_10_5_agent_gpt-4o_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model \"gpt-4o\" --run 2004"
    "./run_experiments.sh --output-file 10_10_2_agent_o1-mini_.txt --summary-file 10_10_2_agent_o1-mini_summary.txt --python-script helper_script_n_agents.py --num-agents 2 --model \"o1-mini\" --run 2005"
    "./run_experiments.sh --output-file 10_10_3_agent_o1-mini_.txt --summary-file 10_10_3_agent_o1-mini_summary.txt --python-script helper_script_n_agents.py --num-agents 3 --model \"o1-mini\" --run 2006"
    "./run_experiments.sh --output-file 10_10_4_agent_o1-mini_.txt --summary-file 10_10_4_agent_o1-mini_summary.txt --python-script helper_script_n_agents.py --num-agents 4 --model \"o1-mini\" --run 2007"
    "./run_experiments.sh --output-file 10_10_5_agent_o1-mini_.txt --summary-file 10_10_5_agent_o1-mini_summary.txt --python-script helper_script_n_agents.py --num-agents 5 --model \"o1-mini\" --run 2008"
    "./run_experiments.sh --output-file 10_10_until_gpt-4o_.txt --summary-file 10_10_until_gpt-4o_summary.txt --python-script helper_script_until_n.py --model \"gpt-4o\" --run 2009"
    "./run_experiments.sh --output-file 10_10_until_o1-mini_.txt --summary-file 10_10_until_o1-mini_summary.txt --python-script helper_script_until_n.py --model \"o1-mini\" --run 2010"
    "./run_experiments.sh --output-file 10_10_choose_gpt-4o_.txt --summary-file 10_10_choose_gpt-4o_summary.txt --python-script helper_script_choose_n.py --model \"gpt-4o\" --run 2011"
    "./run_experiments.sh --output-file 10_10_choose_o1-mini_.txt --summary-file 10_10_choose_o1-mini_summary.txt --python-script helper_script_choose_n.py --model \"o1-mini\" --run 2012"
)

# Base directory and environment activation
base_dir="/data/ishika/david/multiagent_helper_llmp/"
env_activate_cmd="conda activate /data/ishika/david/envs/llm_pddl/"

# Create a new tmux session
session="experiment_session"

# Start the tmux session in detached mode
tmux new-session -d -s $session -n initial_window

# Iterate over the commands and create a new tmux window for each one
for i in "${!commands[@]}"; do
    window_name="window_$i"
    tmux new-window -t $session -n "$window_name"
    cmd="cd $base_dir && $env_activate_cmd && echo 'Running: ${commands[$i]}' && ${commands[$i]}"
    tmux send-keys -t $session:$window_name "$cmd" C-m
done

echo "All tmux windows started and detached."
