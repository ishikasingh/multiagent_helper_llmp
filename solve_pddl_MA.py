import os
import json
import argparse
from modules.planner import planner

def evaluate_domain(domain_path, time_limit=300):
    domain_name = os.path.basename(domain_path)
    
    # Create Args object with necessary parameters
    class Args:
        def __init__(self):
            self.domain = domain_name
            self.time_limit = time_limit
            self.run = 10000
            self.task_id = None
            self.num_agents = 1  # Set default, adjust if needed
    
    args = Args()
    
    # Get all problem files
    problem_files = [f for f in os.listdir(domain_path) if f.startswith('p') and f.endswith('.pddl')]
    
    # Create domain-specific folder in MA_cache
    domain_cache_path = os.path.join("MA_cache", domain_name)
    os.makedirs(domain_cache_path, exist_ok=True)
    
    for problem_file in sorted(problem_files):
        task_id = int(problem_file[1:-5])  # Extract number from pXX.pddl
        args.task_id = f"{task_id:02d}"  # Update task_id for current problem
        print(f"\nEvaluating {domain_name} problem {args.task_id}")
        
        # Create domain-specific cache directory
        cache_path = os.path.join("experiments_multiagent_help/run_10000", domain_name)
        os.makedirs(cache_path, exist_ok=True)
        
        stats = planner(cache_path, args)
        results = {
            "planner_total_time": stats[0],
            "planner_total_time_opt": stats[1],
            "best_cost": stats[2],
            "planner_search_time_1st_plan": stats[3],
            "first_plan_cost": stats[4]
        }
        print(f"Results for task {args.task_id}:", results)
        
        # Save result file in domain-specific folder
        result_filename = f"p{args.task_id}.json"
        result_path = os.path.join(domain_cache_path, result_filename)
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate PDDL domains')
    parser.add_argument('--time-limit', type=int, default=300,
                      help='Time limit for each problem in seconds (default: 300)')
    parser.add_argument('--domain', type=str, required=True,
                      help='Domain name (e.g., barman-multi-3)')
    
    args = parser.parse_args()
    
    # Construct the full domain path
    domain_path = os.path.join('domains', args.domain)
    
    if not os.path.isdir(domain_path):
        print(f"Error: {domain_path} is not a directory")
        return
    
    print(f"\nProcessing domain: {domain_path}")
    evaluate_domain(domain_path, args.time_limit)

if __name__ == "__main__":
    main()