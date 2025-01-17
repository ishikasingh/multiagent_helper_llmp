import os
import json
from collections import defaultdict

def analyze_ma_cache(root_dir="MA_cache", output_file=None):
    # Dictionary to store results for each domain
    results = defaultdict(lambda: {"times": [], "costs": []})
    
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
                    best_cost = data.get('best_cost', 0)
                    
                    # Store values
                    results[domain]['times'].append(planning_time)
                    results[domain]['costs'].append(best_cost)
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Write results to file if specified, otherwise print to console
    output = []
    output.append("\nResults:")
    output.append("-" * 50)
    
    for domain, data in results.items():
        avg_time = sum(data['times']) / len(data['times']) if data['times'] else 0
        avg_cost = sum(data['costs']) / len(data['costs']) if data['costs'] else 0
        
        output.append(f"\nDomain: {domain}")
        output.append(f"Average Planning Time: {avg_time:.2f} seconds")
        output.append(f"Average Solution Cost: {avg_cost:.2f}")

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
