import os
import json
from collections import defaultdict

def analyze_ma_cache(root_dir="MA_cache", output_file=None):
    # Dictionary to store results for each domain
    results = defaultdict(lambda: {
        "times": [], 
        "costs": [], 
        "tasks": {},
        "timeouts": 0  # Add timeout counter
    })
    
    # Walk through MA_cache directory
    for domain in os.listdir(root_dir):
        domain_path = os.path.join(root_dir, domain)
        
        # Skip if not a directory
        if not os.path.isdir(domain_path):
            continue
            
        # Process each JSON file in the domain directory
        for filename in os.listdir(domain_path):
            if filename.endswith('.json'):
                file_path = os.path.join(domain_path, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    # Extract planning time and best cost
                    planning_time = data.get('planner_total_time', 0)
                    best_cost = data.get('best_cost')
                    task_id = filename.split('.')[0]  # Get task ID from filename
                    
                    # Check for timeout (best_cost is null)
                    if best_cost is None:
                        results[domain]['timeouts'] += 1
                        continue  # Skip adding this to times and costs lists
                    
                    # Store values only for non-timeout cases
                    results[domain]['times'].append(planning_time)
                    results[domain]['costs'].append(best_cost)
                    results[domain]['tasks'][task_id] = {
                        'time': planning_time,
                        'cost': best_cost
                    }
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Write results to file if specified, otherwise print to console
    output = []
    output.append("\nSummary Results:")
    output.append("-" * 50)
    
    # First print summary statistics
    for domain, data in results.items():
        avg_time = sum(data['times']) / len(data['times']) if data['times'] else 0
        avg_cost = sum(data['costs']) / len(data['costs']) if data['costs'] else 0
        total_tasks = len(data['tasks']) + data['timeouts']  # Total including timeouts
        
        output.append(f"\nDomain: {domain}")
        output.append(f"Number of Tasks: {total_tasks}")
        output.append(f"Number of Timeouts: {data['timeouts']}")
        output.append(f"Success Rate: {(total_tasks - data['timeouts']) / total_tasks * 100:.1f}%")
        output.append(f"Average Planning Time (successful only): {avg_time:.2f} seconds")
        output.append(f"Average Solution Cost (successful only): {avg_cost:.2f}")

    # Then print individual task statistics
    output.append("\n\nDetailed Task Results:")
    output.append("-" * 50)
    
    for domain, data in results.items():
        output.append(f"\nDomain: {domain}")
        output.append("Task ID\t\tPlanning Time (s)\tSolution Cost")
        output.append("-" * 50)
        
        # Sort tasks by ID number
        sorted_tasks = sorted(data['tasks'].items(), 
                            key=lambda x: int(''.join(filter(str.isdigit, x[0]))))
        
        for task_id, task_data in sorted_tasks:
            output.append(f"{task_id}\t\t{task_data['time']:.2f}\t\t{task_data['cost']:.2f}")

    output_text = "\n".join(output)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output_text)
    else:
        print(output_text)

if __name__ == "__main__":
    import sys
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    analyze_ma_cache(output_file=output_file)
