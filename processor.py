import re
from collections import defaultdict
import argparse

def parse_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    results = defaultdict(list)
    current_domain = None
    current_task = None
    current_result = {}

    for line in content.split('\n'):
        if line.startswith('[results]'):
            if current_result:
                results[current_domain].append(current_result)
                current_result = {}
            match = re.search(r'\[results\]\[(\w+)\]\[(\d+)\]', line)
            if match:
                current_domain = match.group(1)
                current_task = match.group(2)
                current_result['task'] = current_task
        elif line.startswith('[single_agent]'):
            current_result['single_agent'] = line
        elif line.startswith('[multi_agent]'):
            current_result['multi_agent'] = line
            
            # Check for 'inf' in multi-agent cost
            if 'cost: inf' in line:
                current_result['success'] = False
                current_result['multi_agent_inf'] = True
            else:
                single_cost_match = re.search(r'cost: (\d+\.?\d*)', current_result['single_agent'])
                multi_cost_match = re.search(r'cost: (\d+\.?\d*)', current_result['multi_agent'])
                
                if single_cost_match and multi_cost_match:
                    single_cost = float(single_cost_match.group(1))
                    multi_cost = float(multi_cost_match.group(1))
                    current_result['success'] = multi_cost < single_cost
                else:
                    current_result['success'] = False
                current_result['multi_agent_inf'] = False

    if current_result:
        results[current_domain].append(current_result)

    return results

def calculate_averages(results):
    averages = {}
    for domain, tasks in results.items():
        single_agent_costs = []
        single_agent_times = []
        multi_agent_costs = []
        multi_agent_times = []

        for task in tasks:
            if not task.get('inf', False):
                single_cost_match = re.search(r'cost: (\d+\.?\d*)', task['single_agent'])
                single_time_match = re.search(r'planning time: (\d+\.?\d*)', task['single_agent'])
                multi_cost_match = re.search(r'cost: (\d+\.?\d*)', task['multi_agent'])
                multi_time_match = re.search(r'planning_time: (\d+\.?\d*)', task['multi_agent'])

                if single_cost_match:
                    single_agent_costs.append(float(single_cost_match.group(1)))
                if single_time_match:
                    single_agent_times.append(float(single_time_match.group(1)))
                if multi_cost_match:
                    multi_agent_costs.append(float(multi_cost_match.group(1)))
                if multi_time_match:
                    multi_agent_times.append(float(multi_time_match.group(1)))

        averages[domain] = {
            'single_agent_avg_cost': sum(single_agent_costs) / len(single_agent_costs) if single_agent_costs else None,
            'single_agent_avg_time': sum(single_agent_times) / len(single_agent_times) if single_agent_times else None,
            'multi_agent_avg_cost': sum(multi_agent_costs) / len(multi_agent_costs) if multi_agent_costs else None,
            'multi_agent_avg_time': sum(multi_agent_times) / len(multi_agent_times) if multi_agent_times else None
        }

    return averages

def analyze_results(results, output_file):
    averages = calculate_averages(results)

    with open(output_file, 'w') as f:
        for domain, tasks in results.items():
            successful_tasks = sum(1 for task in tasks if task['success'])
            total_tasks = len(tasks)
            success_rate = successful_tasks / total_tasks * 100

            f.write(f"Domain: {domain}\n")
            f.write(f"Success rate: {success_rate:.2f}% ({successful_tasks}/{total_tasks})\n")
            f.write("Results:\n")
            for task in tasks:
                f.write(f"  [results][{domain}][{task['task']}]\n")
                f.write(f"  {task['single_agent']}\n")
                f.write(f"  {task['multi_agent']}\n")
                f.write(f"  Success: {'No (Inf cost)' if task.get('multi_agent_inf', False) else 'Yes' if task['success'] else 'No'}\n")
                f.write("\n")
            f.write("-" * 40 + "\n")

        f.write("\nDomain Averages:\n")
        for domain, avg in averages.items():
            f.write(f"Domain: {domain}\n")
            f.write("  Single Agent:\n")
            if avg['single_agent_avg_cost'] is not None:
                f.write(f"    Average Cost: {avg['single_agent_avg_cost']:.2f}\n")
            else:
                f.write("    Average Cost: N/A\n")
            if avg['single_agent_avg_time'] is not None:
                f.write(f"    Average Planning Time: {avg['single_agent_avg_time']:.2f} sec\n")
            else:
                f.write("    Average Planning Time: N/A\n")
            f.write("  Multi Agent:\n")
            if avg['multi_agent_avg_cost'] is not None:
                f.write(f"    Average Cost: {avg['multi_agent_avg_cost']:.2f}\n")
            else:
                f.write("    Average Cost: N/A\n")
            if avg['multi_agent_avg_time'] is not None:
                f.write(f"    Average Planning Time: {avg['multi_agent_avg_time']:.2f} sec\n")
            else:
                f.write("    Average Planning Time: N/A\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("input_file", help="Path to the input file containing experiment results")
    parser.add_argument("output_file", help="Path to the output file for analysis results")
    args = parser.parse_args()

    results = parse_results(args.input_file)
    analyze_results(results, args.output_file)
    print(f"Analysis complete. Results written to {args.output_file}")

if __name__ == "__main__":
    main()