for task_id in {1..20}
do
  python helper_script_n_agents.py --run 101 --domain "tyreworld" --time-limit 20 --task_id $task_id --num_agents 3
done