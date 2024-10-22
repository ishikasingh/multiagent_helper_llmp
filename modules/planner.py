from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
import subprocess
import re
import time
import glob
import modules.utils as utils
import numpy as np
import itertools

FAST_DOWNWARD_ALIAS = "lama"

AGENT_PREDICATES = {
    "barman": ['handempty', 'holding'],
    "blocksworld": ['arm-empty', 'holding'],
    "grippers": ['at-robby', 'free', 'carry'],
    "termes": ['has-block', 'at'],
    "tyreworld": [],
    "barman-multi": ['handempty', 'holding'],
    "blocksworld-multi": ['arm-empty', 'holding'],
    "termes-multi": ['has-block', 'at'],
    "tyreworld-multi": [],
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
    with open(validation_filename, 'r') as f:
        validation = f.readlines()
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

    with open(pddl_problem_filename_edited, 'w') as f:
        f.write(pddl_problem)

def get_updated_init_conditions_recurse(expt_path, validation_filename=None, pddl_problem_filename=None, pddl_problem_filename_edited=None, env_conds_only=True):
    with open(validation_filename, 'r') as f:
        validation = f.readlines()
    
    pddl_problem_filename_ =  f"./domains/{args.domain}/p{args.task_id}.pddl" if pddl_problem_filename==None else pddl_problem_filename
    with open(pddl_problem_filename_, 'r') as f:
        pddl_problem = f.read()
    pddl_problem = pddl_problem.split('(:init')
    if len(pddl_problem) < 2:
        print(f"Error: Could not find (:init in {pddl_problem_filename_}")
        return
        
    pddl_problem[1:] = pddl_problem[1].split('(:goal')
    if len(pddl_problem) < 3:
        print(f"Error: Could not find (:goal in {pddl_problem_filename_}")
        return

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
                if not plans:
                    print(f"Error: Empty plan file found at {fn}")
                    continue
                cost = utils.get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    best_plan_file = fn
        if best_plan_file:
            with open(best_plan_file, 'r') as f:
                agent_plans.append(tuple(f.readlines()[:-1]))  # Convert to tuple
        else:
            print(f"Error: No valid plan file found for agent {i}")
            return float('inf'), False

    if not agent_plans:
        print("Error: No valid plans found for any agents")
        return float('inf'), False

    print(f"TASK: {args.domain} - {args.run} - {args.task_id}")
    print("AGENT PLANS", agent_plans)
    global log_file
    log_file = logfile

    with open(log_file, 'a+') as f:
        f.write(f"TASK: {args.domain} - {args.run} - {args.task_id}\n")

    global execution_state
    execution_state = np.full([len(plan) + 1 for plan in agent_plans] + [2**args.num_agents], float('inf'))
    print("EXECUTION STATE", execution_state.shape)
    plan_length = validator_sim_recursion_function(expt_path, domain_pddl_file, tuple([0] * args.num_agents), tuple(agent_plans), tuple([task] * args.num_agents))

    success = plan_length < float('inf')
    print(plan_length, success)
    return plan_length, success

@lru_cache(maxsize=None)
def validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_subset: int = 0):
    if not agent_plans:
        print("Error: Empty agent_plans list")
        return float('inf')

    num_agents = len(agent_plans)

    # Check each index is within bounds
    for i, (idx, plan) in enumerate(zip(indices, agent_plans)):
        if idx > len(plan):
            print(f"Error: Index {idx} out of bounds for agent {i}'s plan (length {len(plan)})")
            return float('inf')
    
    for i in range(num_agents):
        if indices[i] > len(agent_plans[i]):
            return float('inf')
    
    if all(indices[i] >= len(agent_plans[i]) for i in range(num_agents)):
        return 0

    # Ensure agent_subset is an integer bitmask
    agent_subset = 0 if agent_subset is None else int(agent_subset)
    
    # Validate agent_subset range
    if agent_subset < 0:
        print(f"Error: Negative agent_subset value: {agent_subset}")
        return float('inf')
    if agent_subset >= 2**num_agents:
        print(f"Error: Agent subset {agent_subset} exceeds maximum value for {num_agents} agents (max: {2**num_agents - 1})")
        return float('inf')

    state_index = indices + (agent_subset,)
    if execution_state[state_index] != float('inf'):
        return execution_state[state_index]

    completed_agents = set(i for i in range(len(agent_plans)) if indices[i] >= len(agent_plans[i]))
    # If all agents have completed their plans, return 0
    if len(completed_agents) == len(agent_plans):
        return 0

    # If agent_subset is 0, generate new subsets
    if agent_subset == 0:
        plans = []
        # Only consider agents that haven't completed their plans
        active_agents = [i for i in range(len(agent_plans)) if i not in completed_agents]
        
        # Generate combinations of active agents
        for r in range(len(active_agents), 0, -1):
            for subset in itertools.combinations(active_agents, r):
                subset_mask = sum(1 << i for i in subset)
                plan_length = validator_sim_recursion_function(
                    expt_path, domain_pddl_file, indices, 
                    agent_plans, agent_tasks, subset_mask
                )
                plans.append(plan_length)
        
        result = 1 + min(plans) if plans else float('inf')
        execution_state[state_index] = result
        return result
    else:
        # Convert bitmask to list of active agent indices
        subset = [i for i in range(num_agents) if (agent_subset & (1 << i))]
        if not subset:
            print(f"Error: Empty agent subset created from bitmask {agent_subset}")
            return float('inf')
        result = execute_agents_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, subset)

    execution_state[state_index] = result
    return result

def execute_agents_action(expt_path, domain_pddl_file, indices, agent_plans, task_states, agent_subset):
    # Ensure agent_subset is an integer if it's passed as a list
    if isinstance(agent_subset, list):
        # Convert list of indices to bitmask
        agent_subset_mask = sum(1 << i for i in agent_subset)
    else:
        agent_subset_mask = agent_subset

    # Now use the bitmask to get agent indices
    agent_indices = [i for i in range(len(agent_plans)) if (agent_subset_mask & (1 << i))]
    
    # Add debug prints
    print("agent_indices:", agent_indices)
    print("indices:", indices)
    print("agent_plans lengths:", [len(plan) for plan in agent_plans])
    
    # Check if any indices are out of range
    if any(i >= len(agent_plans) for i in agent_indices):
        print("Error: Agent index exceeds number of plans")
        return float('inf')
        
    if any(i >= len(indices) for i in agent_indices):
        print("Error: Agent index exceeds length of indices array") 
        return float('inf')

    val_paths = [f"./{expt_path}/agent{i}_val_temp.txt" for i in agent_indices]
    plan_paths = [f"./{expt_path}/agent{i}_plan_temp.txt" for i in agent_indices]
    task_paths = [f"./{expt_path}/agent{i}_task_temp.txt" for i in agent_indices]
    new_task_paths = [f"./{expt_path}/agent{i}_new_task_temp.txt" for i in agent_indices]

    # print("VAL PATHS", val_paths)
    # print("PLAN PATHS", plan_paths)
    # print("TASK PATHS", task_paths)
    # print("NEW TASK PATHS", new_task_paths)

    all_valid = True
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, i in enumerate(agent_indices):
            # Add bounds check before accessing agent_plans[i]
            if i >= len(agent_plans):
                print(f"Error: Agent index {i} exceeds number of plans")
                continue
                
            if i >= len(indices):
                print(f"Error: Agent index {i} exceeds length of indices array")
                continue
                
            # Skip agents that have completed their plans
            if indices[i] >= len(agent_plans[i]):
                continue
                
            # Check task_states bounds
            if i >= len(task_states):
                print(f"Error: Agent index {i} exceeds length of task_states")
                continue
                
            with open(plan_paths[idx], 'w') as f:
                f.write(agent_plans[i][indices[i]])
            with open(task_paths[idx], 'w') as f:
                f.write(task_states[i])
            
            futures.append(executor.submit(subprocess.run, ["./downward/validate", "-v", domain_pddl_file, task_paths[idx], plan_paths[idx]], capture_output=True, text=True))
        
        if not futures:
            return float('inf')
        
        for idx, future in enumerate(futures):
            if idx >= len(val_paths):
                print(f"Error: Index {idx} exceeds length of validation paths")
                continue
                
            output = future.result()
            with open(val_paths[idx], 'w') as f:
                f.write(output.stdout)
            
            if 'unsatisfied precondition' in output.stdout:
                all_valid = False
                break
                
            #print("updating init conditions")
            get_updated_init_conditions_recurse(expt_path, validation_filename=val_paths[idx], pddl_problem_filename=task_paths[idx], pddl_problem_filename_edited=new_task_paths[idx], env_conds_only=False)
            for j in range(len(agent_plans)):
                # Only update init conditions for agents not in the current subset
                 if not (agent_subset_mask & (1 << j)):
                    # print(f"accessing {j} in arrays of size {len(task_paths)} and {len(new_task_paths)}")
                    get_updated_init_conditions_recurse(expt_path, validation_filename=val_paths[idx], pddl_problem_filename=f"./{expt_path}/agent{j}_task_temp.txt", pddl_problem_filename_edited=f"./{expt_path}/agent{j}_new_task_temp.txt")
            #print("done updating init conditions")
            
    if all_valid:
        print("all valid")
        with open(log_file, 'a+') as f:
            for i in agent_indices:
                if i < len(agent_plans) and i < len(indices) and indices[i] < len(agent_plans[i]):
                    f.write(f"Agent {i}, {indices[i]}, {agent_plans[i][indices[i]][:-1]}\n")

        new_indices = list(indices)
        for i in agent_indices:
            if i < len(agent_plans):
                new_indices[i] += 1
                
        new_task_states = list(task_states)
        for idx, i in enumerate(agent_indices):
            if idx < len(new_task_paths) and i < len(new_task_states):
                with open(new_task_paths[idx], 'r') as f:
                    new_task_states[i] = f.read()

        return validator_sim_recursion_function(expt_path, domain_pddl_file, tuple(new_indices), agent_plans, tuple(new_task_states))
    else:
        return float('inf')