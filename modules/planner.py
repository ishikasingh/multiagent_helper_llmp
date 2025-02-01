from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import subprocess
import re
import time
import glob
import modules.utils as utils
import numpy as np
import sys

sys.setrecursionlimit(10000)

FAST_DOWNWARD_ALIAS = "lama"

AGENT_PREDICATES = {
    "barman": ['handempty', 'holding'],
    "barman-enabled": ['handempty', 'holding'],
    "blocksworld": ['arm-empty', 'holding'],
    "grippers": ['at-robby', 'free', 'carry'],
    "termes": ['has-block', 'at'],
    "tyreworld": [],
    "barman-multi": ['handempty', 'holding'],
    "blocksworld-multi": ['arm-empty', 'holding'],
    "termes-multi": ['has-block', 'at'],
    "tyreworld-multi": [],
    "tyreworld-enabled": [],
    "grippers-multi": ['at-robby', 'free', 'carry'],
}

def planner(expt_path, args_, subgoal_idx=-1):

    global args
    args = args_

    domain_pddl_file =  f'./domains/{args.domain}/domain.pddl'
    # if multiagent w single subgoal get subgoal pddl
    if subgoal_idx > 0:
        if subgoal_idx == args.num_agents:
            task_pddl_file_name =  f"./{expt_path}/p{args.task_id}_0.pddl"
            plan_file_name = f"./{expt_path}/p{args.task_id}_0_plan.pddl"
            info = 'multiagent main goal'
        else:
            task_pddl_file_name =  f"./{expt_path}/p{args.task_id}_{subgoal_idx}.pddl"
            plan_file_name = f"./{expt_path}/p{args.task_id}_{subgoal_idx}_plan.pddl"
            info = f'subgoal_{subgoal_idx}'
    # get default pddl > single agent
    else:
        task_pddl_file_name =  f"./domains/{args.domain}/p{args.task_id}.pddl"
        plan_file_name = f"./{expt_path}/p{args.task_id}_plan.pddl"
        info = 'singleagent'
    sas_file_name = plan_file_name + '.sas'
    output_path = plan_file_name + '.out'

    start_time = time.time()

    # run fastforward to plan
    os.system(f"python ./downward/fast-downward.py --alias {FAST_DOWNWARD_ALIAS} " + \
              f"--search-time-limit {args.time_limit} --plan-file {plan_file_name} " + \
              f"--sas-file {sas_file_name} " + \
              f"{domain_pddl_file} {task_pddl_file_name} > {output_path}")
    with open(output_path, "r") as f:
        output = f.read()
    
    if(output.find('Actual search time') == -1):
        print("planner broke")
        print(output)
        
    planner_search_time_1st_plan = float(output.split('Actual search time: ')[1].split('\n')[0].strip()[:-1])
    planner_total_time = float(output.split('Planner time: ')[1].split('\n')[0].strip()[:-1])
    planner_total_time_opt = float(output.split('Actual search time: ')[-1].split('\n')[0].strip()[:-1])
    first_plan_cost = int(output.split('Plan cost: ')[1].split('\n')[0].strip())
    #import ipdb; ipdb.set_trace()
    # collect the least cost plan
    best_cost = 1e10
    best_plan = None

    for fn in glob.glob(f"{plan_file_name}.*"):
        with open(fn, "r") as f:
            plans = f.readlines()
            cost = utils.get_cost(plans[-1])
            if cost < best_cost:
                best_cost = cost
                best_plan = "\n".join([p.strip() for p in plans[:-1]])

    end_time = time.time()
    if best_plan:
        print(f"[info][{info}][{args.domain}] task {args.task_id} takes {planner_total_time} sec, found a plan with cost {best_cost}")
        # print(planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost)
        return planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost
    else:
        print(f"[info][{info}][{args.domain}] task {args.task_id} takes {planner_total_time} sec, no solution found")
        return -1, -1, -1, -1, -1

def validator(expt_path, subgoal_idx=-1):
    print("validating")
    if subgoal_idx >= 0:
        output_path = f"./{expt_path}/p{args.task_id}_{subgoal_idx}_validation.txt"
    else:
        output_path = f"./{expt_path}/p{args.task_id}_validation.txt"
    output_file = open(output_path, "w")

    domain_pddl_file =  f'./domains/{args.domain}/domain.pddl'

    if subgoal_idx >= 0:
        task_pddl_file =  f"./{expt_path}/p{args.task_id}_{subgoal_idx}.pddl"
        # print("validating and getting plan for subgoal", subgoal_idx)
        plan_path = os.path.join(f"./{expt_path}", 
                                f"p{args.task_id}_{subgoal_idx}_plan.pddl" + '.*')
    else:
        task_pddl_file =  f"./{expt_path}/p{args.task_id}.pddl"
        plan_path = os.path.join(f"./{expt_path}", 
                                f"p{args.task_id}_plan.pddl" + '.*')

    best_cost = 10e6
    plan_file = ''
    for fn in glob.glob(plan_path):
        with open(fn, "r") as f:
            plans = f.readlines()
            cost = utils.get_cost(plans[-1])
            if cost < best_cost:
                best_cost = cost
                plan_file = fn
    # print("plan_file", plan_file)
    v_start = time.time()
    result = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_pddl_file, plan_file], stdout=subprocess.PIPE)
    v_end = time.time()
    #print("validated")
    output = result.stdout.decode('utf-8')
    output_file.write(output)
    if "Plan valid" in result.stdout.decode('utf-8'):
        return True, v_end - v_start
    else:
        return False, v_end - v_start

def get_updated_init_conditions(expt_path, validation_filename=None, pddl_problem_filename=None, pddl_problem_filename_edited=None, env_conds_only=True, is_main=False):
    # print("getting updated init conditions")
    # print("validation file", validation_filename)
    # validation_filename = f"./{expt_path}/p{args.task_id}_subgoal_validation.txt" if validation_filename==None else validation_filename
    with open(validation_filename, 'r') as f:
        validation = f.readlines()
    # print(pddl_problem_filename)
    # pddl_problem_filename_ =  f"./domains/{args.domain}/p{args.task_id}.pddl" if pddl_problem_filename==None else pddl_problem_filename
    with open(pddl_problem_filename, 'r') as f:
        pddl_problem = f.read()
    pddl_problem = pddl_problem.split('(:init')
    pddl_problem[1:] = pddl_problem[1].split('(:goal')
    pddl_problem[1] = pddl_problem[1].strip()[:-1] # remove last ')'
    init_conditions  = [cond.strip() for cond in pddl_problem[1].split('\n') if len(cond)>1]
    new_init_conditions = init_conditions.copy()
    for line in validation:
        if any([x in line for x in AGENT_PREDICATES[args.domain]]) and env_conds_only: # skip agent states
            continue
        if 'Adding' in line:
            added_condition = line.split('Adding')[1].strip()
            if added_condition not in new_init_conditions:
                new_init_conditions.append(added_condition)
        if 'Deleting' in line:
            deleted_condition = line.split('Deleting')[1].strip()
            if deleted_condition in new_init_conditions:
                new_init_conditions.remove(deleted_condition)

    pddl_problem[1] = list(new_init_conditions)
    # get new goal for next state, from subgoal we are passing new conditions into
    if is_main: # get original goal from domain descriptor
        with open(f"./domains/{args.domain}/p{args.task_id}.pddl", 'r') as f:
            next_pddl_problem = f.read() 
    else:
        with open(pddl_problem_filename_edited, 'r') as f:
            next_pddl_problem = f.read()
    next_pddl_problem = next_pddl_problem.split('(:init')
    next_pddl_problem[1:] = next_pddl_problem[1].split('(:goal')
    next_pddl_problem[1] = next_pddl_problem[1].strip()[:-1]
    
    pddl_problem = pddl_problem[0] + '(:init\n' + '\n'.join(pddl_problem[1]) + '\n)\n(:goal' + next_pddl_problem[2]

    # print("writing new edited pddl to", pddl_problem_filename_edited)
    with open(pddl_problem_filename_edited, 'w') as f:
        f.write(pddl_problem)

def get_updated_init_conditions_recurse(expt_path, validation_filename=None, pddl_problem_filename=None, pddl_problem_filename_edited=None, env_conds_only=True):
    validation_filename = f"./{expt_path}/p{args.task_id}_subgoal_validation.txt" if validation_filename==None else validation_filename
    with open(validation_filename, 'r') as f:
        validation = f.readlines()
    
    pddl_problem_filename_ =  f"./domains/{args.domain}/p{args.task_id}.pddl" if pddl_problem_filename==None else pddl_problem_filename
    with open(pddl_problem_filename_, 'r') as f:
        pddl_problem = f.read()
    pddl_problem = pddl_problem.split('(:init')
    pddl_problem[1:] = pddl_problem[1].split('(:goal')

    pddl_problem[1] = pddl_problem[1].strip()[:-1] # remove last ')'
    init_conditions  = set([cond.strip() for cond in pddl_problem[1].split('\n') if len(cond)>1])
    new_init_conditions = init_conditions
    for line in validation:
        if any([x in line for x in AGENT_PREDICATES[args.domain]]) and env_conds_only: # skip agent states
            continue
        added_conditions = set([line.split('Adding')[1].strip()]) if 'Adding' in line else set()
        deleted_conditions = set([line.split('Deleting')[1].strip()]) if 'Deleting' in line else set()
        new_init_conditions  = (new_init_conditions | added_conditions) - deleted_conditions

    pddl_problem[1] = list(new_init_conditions)
    pddl_problem = pddl_problem[0] + '(:init\n' + '\n'.join(pddl_problem[1]) + '\n)\n(:goal' + pddl_problem[2]

    pddl_problem_filename = pddl_problem_filename if pddl_problem_filename_edited==None else pddl_problem_filename_edited
    pddl_problem_filename_ =  f"./{expt_path}/p{args.task_id}_edited_init.pddl" if pddl_problem_filename==None else pddl_problem_filename
    with open(pddl_problem_filename_, 'w') as f:
        f.write(pddl_problem)


def validator_simulation_recursive(expt_path, logfile, multi=False):
    domain_pddl_file = f'./domains/{args.domain}/domain.pddl'
    task_pddl_file = f'./domains/{args.domain}/p{args.task_id}.pddl'
    with open(task_pddl_file, 'r') as f:
        task = f.read()

    agent_plans = []
    for i in range(args.num_agents):
        plan_path = os.path.join(f"./{expt_path}", f"p{args.task_id}_{i}_plan.pddl" + '.*')
        best_cost = float('inf')
        best_plan_file = None
        for fn in glob.glob(plan_path):
            with open(fn, "r") as f:
                plans = f.readlines()
                cost = utils.get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    best_plan_file = fn
        if best_plan_file:
            with open(best_plan_file, 'r') as f:
                agent_plans.append(tuple(f.readlines()[:-1]))  # Convert to tuple
        else:
            return float('inf'), False

    print(f"TASK: {args.domain} - {args.run} - {args.task_id}")

    global log_file
    log_file = logfile

    with open(log_file, 'a+') as f:
        f.write(f"TASK: {args.domain} - {args.run} - {args.task_id}\n")

    # Start with agent 1's plan
    initial_plan = [('single', 1, agent_plans[1][i]) for i in range(len(agent_plans[1]))]
    print("initial_plan", initial_plan)

    # First merge plans 2 through n
    for i in range(2, args.num_agents):
        print(f"merging plan {i} into initial_plan")
        next_agent_plan = [('single', i, agent_plans[i][j]) for j in range(len(agent_plans[i]))]
        print("initial_plan", initial_plan)
        print(f"agent_plans[{i}]", next_agent_plan)
        global execution_state
        execution_state = np.full([len(initial_plan) + 1, len(next_agent_plan) + 1, 3], float('inf'))
        plan_length = validator_sim_recursion_function(expt_path, domain_pddl_file, tuple([0] * 2), tuple([tuple(initial_plan), tuple(next_agent_plan)]), tuple([task] * 2))

        success = plan_length < float('inf')
        plan_length = plan_length - 1 if success else float('inf')
        print(plan_length, success)
        
        if not success:
            return float('inf'), False
            
        print("tracing optimal path, building plan object")
        initial_plan = trace_optimal_path(execution_state, (initial_plan, next_agent_plan), log_file)
    #TODO: need to pass in the updated init condition / task variable
    # Finally merge with agent 0's plan
    print("merging main agent's plan into combined plan")
    agent_0_plan = [('single', 0, agent_plans[0][j]) for j in range(len(agent_plans[0]))]
    print("agent_0_plan", agent_0_plan)
    print("initial_plan", initial_plan)
    execution_state = np.full([len(initial_plan) + 1, len(agent_0_plan) + 1, 3], float('inf'))
    plan_length = validator_sim_recursion_function(expt_path, domain_pddl_file, tuple([0] * 2), tuple([tuple(initial_plan), tuple(agent_0_plan)]), tuple([task] * 2))
    
    # Add debugging info before visualization
    print("Final execution state shape:", execution_state.shape)
    print("Number of infinite values:", np.sum(execution_state == float('inf')))
    print("Number of finite values:", np.sum(execution_state != float('inf')))
    
    # visualize_search_tree(execution_state, (initial_plan, agent_0_plan), f"./{expt_path}/search_tree")
    
    success = plan_length < float('inf')
    plan_length = plan_length - 1 if success else float('inf')
    print(plan_length, success)
    
    if success:
        print("tracing optimal path, building final plan")
        initial_plan = trace_optimal_path(execution_state, (initial_plan, agent_0_plan), log_file)
    
    return plan_length, success

#@lru_cache(maxsize=None)
def validator_sim_recursion_function(expt_path,domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute=None):
    num_agents = 2
    if all(indices[i] == len(agent_plans[i]) for i in range(num_agents)):
        return 0

    state_index = indices + (agent_to_execute if agent_to_execute is not None else num_agents,)
    if execution_state[state_index] != float('inf'):
        return execution_state[state_index]

    print("state_index", state_index)
    if agent_to_execute is not None:
        print("executing agent", agent_to_execute)
        result = execute_agent_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute)
    else:
        plans = []
        for i in range(num_agents):
            print("executing agent", i)
            if indices[i] < len(agent_plans[i]):
                plans.append(validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, i))
        
        print("executing all agents")
        plans.append(execute_all_agents_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks))

        result = 1 + min(plans)

    execution_state[state_index] = result
    return result

def flatten_plan(plan):
    out = ""
    for i in range(len(plan)):
        out += str(plan[i][1])
    print("flattened plan")
    print(out)
    return out

def execute_agent_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute):
    current_action = agent_plans[agent_to_execute][indices[agent_to_execute]]
    
    # Setup common paths
    val_path = f"./{expt_path}/agent{agent_to_execute}_val_temp.txt"
    plan_path = f"./{expt_path}/agent{agent_to_execute}_plan_temp.txt"
    task_paths = [f"./{expt_path}/agent{i}_task_temp.txt" for i in range(len(agent_tasks))]

    # Write current tasks
    for i, task in enumerate(agent_tasks):
        with open(task_paths[i], 'w') as f:
            f.write(task)

    # Handle plan flattening
    with open(plan_path, 'w') as f:
        if isinstance(current_action, tuple) and current_action[0] == 'parallel':
            print("unwrapping parallel action")
            current_action = flatten_plan(current_action[1])
            f.write(current_action)
        elif current_action[0] == 'single':
            print("writing single agent plan", current_action[2])
            f.write(current_action[2])
        else:
            print("invalid action", current_action)
            return float('inf')

    # Validate action
    output = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_paths[agent_to_execute], plan_path], capture_output=True, text=True)
    with open(val_path, 'w') as f:
        f.write(output.stdout)

    if 'unsatisfied precondition' not in output.stdout:
        # Log successful action
        with open(log_file, 'a+') as f:
            action_str = current_action if isinstance(current_action, tuple) and current_action[0] == 'parallel' else current_action[2]
            f.write(f"Agent {agent_to_execute}, {indices[agent_to_execute]}, {action_str}\n")

        # Update task states for all agents
        new_task_states = list(agent_tasks).copy()
        for i in range(len(agent_plans)):
            new_task_path = f"./{expt_path}/agent{i}_new_task_temp.txt"
            get_updated_init_conditions_recurse(
                expt_path, 
                validation_filename=val_path,
                pddl_problem_filename=task_paths[i],
                pddl_problem_filename_edited=new_task_path,
                env_conds_only=(i != agent_to_execute)
            )
            with open(new_task_path, 'r') as f:
                new_task_states[i] = f.read()

        # Progress to next state
        new_indices = list(indices)
        new_indices[agent_to_execute] += 1
        return validator_sim_recursion_function(expt_path, domain_pddl_file, tuple(new_indices), agent_plans, tuple(new_task_states))
    else:
        print("unsatisfied precondition encountered for action", current_action)
        return float('inf')

def execute_all_agents_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks):
    val_paths = [f"./{expt_path}/agent{i}_val_temp.txt" for i in range(len(agent_plans))]
    plan_paths = [f"./{expt_path}/agent{i}_plan_temp.txt" for i in range(len(agent_plans))]
    task_paths = [f"./{expt_path}/agent{i}_task_temp.txt" for i in range(len(agent_plans))]
    new_task_paths = [f"./{expt_path}/agent{i}_new_task_temp.txt" for i in range(len(agent_plans))]
    
    all_valid = True
    # Write current tasks
    for i, task in enumerate(agent_tasks):
        with open(task_paths[i], 'w') as f:
            f.write(task)
    
    orders = [[0,1], [1,0]]

    for order in orders:
        print("order", order)
        # For each agent that has a next action
        for i in order:
            if indices[i] < len(agent_plans[i]):
                current_action = agent_plans[i][indices[i]]
                
                if isinstance(current_action, tuple) and current_action[0] == 'parallel':
                    current_action = flatten_plan(current_action[1])
                    with open(plan_paths[i], 'w') as f:
                        f.write(current_action)
                    
                else:
                    with open(plan_paths[i], 'w') as f:
                        f.write(current_action[2])
                    
                output = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_paths[i], plan_paths[i]], capture_output=True, text=True)
                with open(val_paths[i], 'w') as f:
                    f.write(output.stdout)
                
                if 'unsatisfied precondition' in output.stdout:
                    print("unsatisfied precondition encountered for action", current_action)
                    all_valid = False
                    break
                
                # Update environment conditions after each parallel component
                get_updated_init_conditions_recurse(expt_path,
                    validation_filename=val_paths[i],
                    pddl_problem_filename=task_paths[i],
                    pddl_problem_filename_edited=new_task_paths[i],
                    env_conds_only=True)
                    
                # Update other agents' task states
                print("updating other agents' task states")
                for k in range(len(agent_plans)):
                    if k != i:
                        get_updated_init_conditions_recurse(expt_path,
                            validation_filename=val_paths[i],
                            pddl_problem_filename=task_paths[k],
                            pddl_problem_filename_edited=new_task_paths[k],
                            env_conds_only=True)
                        
    if all_valid:
        for i in range(len(agent_plans)):
            if indices[i] < len(agent_plans[i]):
                get_updated_init_conditions_recurse(expt_path,
                    validation_filename=val_paths[i],
                    pddl_problem_filename=task_paths[i],
                    pddl_problem_filename_edited=new_task_paths[i],
                    env_conds_only=False)
                for j in range(len(agent_plans)):
                    if i != j:
                        get_updated_init_conditions_recurse(expt_path,
                            validation_filename=val_paths[i],
                            pddl_problem_filename=task_paths[j],
                            pddl_problem_filename_edited=new_task_paths[j],
                            env_conds_only=True)
                
        
        new_indices = tuple(idx + 1 if idx < len(plan) else idx for idx, plan in zip(indices, agent_plans))
        print("merging plans, all valid")
        new_task_states = []
        for path in task_paths:
            with open(path, 'r') as f:
                new_task_states.append(f.read())
        
        return validator_sim_recursion_function(expt_path, domain_pddl_file, new_indices, agent_plans, tuple(new_task_states))
    else:
        return float('inf')
    
def trace_optimal_path(execution_state, agent_plans, log_file):
    # Start at goal state
    indices = [len(plan) for plan in agent_plans]
    print("indices", indices)
    num_agents = 2
    path = []
    
    while any(idx > 0 for idx in indices):
        current_state = tuple(indices) + (num_agents,)
        
        # Try parallel first
        prev_indices = [idx - 1 if idx > 0 else 0 for idx in indices]
        prev_state = tuple(prev_indices) + (num_agents,)
        
        if prev_state in execution_state and execution_state[prev_state] != float('inf'):
            # Record which agents actually moved
            active_agents = []
            for i in range(num_agents):
                if indices[i] > prev_indices[i]:
                    active_agents.append((i, agent_plans[i][prev_indices[i]][2]))
            
            if len(active_agents) > 1:
                path.append(('parallel', active_agents))
                indices = prev_indices
                continue
        
        # If parallel didn't work, try single agent
        for agent in range(num_agents):
            if indices[agent] > 0:
                test_indices = list(indices)
                test_indices[agent] -= 1
                prev_state = tuple(test_indices) + (agent,)
                
                if prev_state in execution_state and execution_state[prev_state] != float('inf'):
                    path.append(('single', agent, agent_plans[agent][test_indices[agent]][2]))
                    indices = test_indices
                    break
    
    path = list(reversed(path))
    print([action for action in path])
    # Write the path in forward order
    with open(log_file, 'a+') as f:
        f.write("\nOptimal Plan Trace:\n")
        f.write("-" * 50 + "\n")
        
        for action in path:
            if action[0] == 'parallel':
                f.write("Parallel Execution:\n")
                for agent, plan_step in action[1]:
                    f.write(f"  Agent {agent}: {plan_step}")
                f.write("\n")
            else:
                f.write(f"Agent {action[1]}: {action[2]}")
                f.write("\n")
        f.write("-" * 50 + "\n")

    return path

def visualize_search_tree(execution_state, agent_plans, output_path):
    """
    Visualizes the search tree including failed states to show exploration process.
    """
    try:
        import graphviz
    except ImportError:
        print("Please install graphviz: pip install graphviz")
        return

    print(f"Visualizing search tree with shape {execution_state.shape}")
    print(f"Agent plans lengths: {[len(plan) for plan in agent_plans]}")

    dot = graphviz.Digraph(comment='Plan Merge Search Tree')
    dot.attr(rankdir='TB')

    visited = set()
    
    def state_to_str(indices, agent=None):
        """Convert state indices to string representation"""
        base = f"({indices[0]},{indices[1]})"
        if agent is not None:
            base += f"\nAgent:{agent}"
        
        # Add plan steps for better debugging
        if indices[0] < len(agent_plans[0]):
            base += f"\nA0: {agent_plans[0][indices[0]][2].strip()}"
        if indices[1] < len(agent_plans[1]):
            base += f"\nA1: {agent_plans[1][indices[1]][2].strip()}"
        return base

    def add_node(indices, agent=None):
        """Add node to graph if not exists"""
        node_id = f"{indices}_{agent}"
        if node_id not in visited:
            visited.add(node_id)
            
            state_value = execution_state[indices + (agent if agent is not None else 2,)]
            
            # Color coding based on state type
            if all(idx >= len(plan) for idx, plan in zip(indices, agent_plans)):
                color = 'lightgreen'  # Goal state
            elif state_value == float('inf'):
                color = 'lightpink'  # Failed state
            else:
                color = 'lightblue'  # Valid state
            
            label = f"{state_to_str(indices, agent)}\nCost: {state_value:.1f}"
            dot.node(node_id, label, style='filled', fillcolor=color)
        return node_id

    # Add all possible states and transitions
    for i in range(execution_state.shape[0]):
        for j in range(execution_state.shape[1]):
            current_indices = (i, j)
            current_node = add_node(current_indices, 2)  # Add parallel state
            
            # Add possible next states for parallel execution
            if i < len(agent_plans[0]) or j < len(agent_plans[1]):
                next_indices = (
                    min(i + 1, len(agent_plans[0])),
                    min(j + 1, len(agent_plans[1]))
                )
                next_node = add_node(next_indices, 2)
                cost = execution_state[i, j, 2]
                dot.edge(current_node, next_node, 
                        f'parallel\ncost: {cost:.1f}',
                        color='red' if cost == float('inf') else 'black')

            # Add possible next states for individual agents
            for agent in range(2):
                current_node = add_node(current_indices, agent)
                if (agent == 0 and i < len(agent_plans[0])) or \
                   (agent == 1 and j < len(agent_plans[1])):
                    next_indices = list(current_indices)
                    next_indices[agent] += 1
                    next_node = add_node(tuple(next_indices), agent)
                    cost = execution_state[i, j, agent]
                    
                    # Safely get action string
                    try:
                        action = agent_plans[agent][current_indices[agent]]
                        action_str = action[2].strip() if isinstance(action, tuple) and len(action) >= 3 else str(action)
                    except Exception as e:
                        action_str = "ERROR"
                        print(f"Error getting action string: {e}")
                    
                    dot.edge(current_node, next_node,
                            f'A{agent}:{action_str}\ncost: {cost:.1f}',
                            color='red' if cost == float('inf') else 'black')

    # Save the visualization
    try:
        dot.render(output_path, view=True, format='pdf', cleanup=True)
        print(f"Successfully rendered graph to {output_path}.pdf")
    except Exception as e:
        print(f"Error rendering graph: {e}")

    # Print some debugging info about the graph
    print(f"Total nodes created: {len(visited)}")
    print("Sample of agent plans:")
    print(f"Agent 0 first action: {agent_plans[0][0] if len(agent_plans[0]) > 0 else 'None'}")
    print(f"Agent 1 first action: {agent_plans[1][0] if len(agent_plans[1]) > 0 else 'None'}")