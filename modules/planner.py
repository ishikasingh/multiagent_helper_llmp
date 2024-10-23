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

# finds all PDDL files, creates DP array, and calls recursive helper
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
    plan_length = validator_sim_recursion_function(expt_path, domain_pddl_file, tuple([0] * args.num_agents), tuple(agent_plans), tuple([task] * args.num_agents), agent_subset=None)

    success = plan_length < float('inf')
    print(plan_length, success)
    return plan_length, success

# take in a set of indices (agent steps in their plans), DP array, and agent subset
# generate an agent subset if none
# then, execute agents and get result for subset
# base case for 
def validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, task_states, agent_subset):
    if agent_subset is None:
        # generate subsets of agent_subset
        active_agents = get_active_agents(indices, agent_plans)
        # print("ACTIVE AGENTS", active_agents)
        subsets = generate_agent_subsets(active_agents)
        # Sort subsets by size in descending order
        subsets.sort(key=len, reverse=True)
        for subset in subsets:
            plan_length = execute_agents_actions(expt_path, domain_pddl_file, indices, agent_plans, execution_state, task_states, subset, active_agents)
            # add full search later
            if plan_length < float('inf'):
                print(f"Found plan of length {plan_length} for subset {subset}")
                return plan_length
        return float('inf')
    else:
        plan_length = execute_agents_actions(expt_path, domain_pddl_file, indices, agent_plans, execution_state, task_states, agent_subset, active_agents)
        if plan_length < float('inf'):
            print(f"Found plan of length {plan_length} for subset {agent_subset}")
        return plan_length

'''ADD LOGS TO EVERYTHING'''
# take in a set of agents
# through multithreading, run downward validate on all
# if all succeed, then this subset of agents is valid
# update conditions for all agents in and outside of the subset
# increment appropriate indices for agent steps and call validator_sim_recursion_function
def execute_agents_actions(expt_path, domain_pddl_file, indices, agent_plans, execution_state, task_states, agent_subset, agent_set):
    '''Execute agents in agent_subset, update world state, and increment indices'''
    is_valid = True
    for i in agent_subset:
        task_path = f"./{expt_path}/agent{i}_task_temp.txt"
        with open(task_path, 'w') as f:
            f.write(task_states[i])

        plan_path = f"./{expt_path}/agent{i}_plan_temp.txt"
        with open(plan_path, 'w') as f:
            f.write(agent_plans[i][indices[i]])

        val_path = f"./{expt_path}/agent{i}_val_temp.txt"

        print("validating execution of ", agent_plans[i][indices[i]], "for agent", i, "at indices", indices)

        is_valid = is_valid and is_valid_state(domain_pddl_file, task_path, plan_path, val_path)
    print("IS VALID for agents", agent_subset, is_valid)
    if is_valid:
        for i in range(len(agent_subset)):
            update_world_state(expt_path, agent_subset[i], [j for j in agent_set if j not in agent_subset])
        # Increment indices for agents in subset
        new_indices = list(indices)
        for agent in agent_subset:
            if indices[agent] < len(agent_plans[agent]):
                new_indices[agent] += 1
        print("NEW INDICES", new_indices)
        new_task_states = list(task_states)
        for i in range(len(indices)):
            with open(f"./{expt_path}/agent{i}_new_task_temp.txt", 'r') as f:
                new_task_states[i] = f.read()
            with open(f"./{expt_path}/agent{i}_task_temp.txt", 'w') as f:
                f.write(task_states[i])
        return validator_sim_recursion_function(expt_path, domain_pddl_file, tuple(new_indices), agent_plans, new_task_states, None)
    return float('inf')

def is_valid_state(domain_pddl_file, task_path, plan_path, val_path):
    """Check if current state is valid by validating a single agent's plan"""
    print("plan path", plan_path)
    output = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_path, plan_path], capture_output=True, text=True)
    with open(val_path, 'w') as f:
        f.write(output.stdout)
    return 'unsatisfied precondition' not in output.stdout

def get_active_agents(indices, agent_plans):
    """Return list of agents with remaining actions"""
    return [i for i in range(len(indices)) if indices[i] < len(agent_plans[i])]

def update_world_state(expt_path, active_agent: int, inactives: list[int]):
    """Update world state based on validation results"""
    get_updated_init_conditions_recurse(expt_path, validation_filename=f"./{expt_path}/agent{active_agent}_val_temp.txt", pddl_problem_filename=f"./{expt_path}/agent{active_agent}_task_temp.txt", pddl_problem_filename_edited=f"./{expt_path}/agent{active_agent}_new_task_temp.txt", env_conds_only=False)
    for i in inactives:
        get_updated_init_conditions_recurse(expt_path, validation_filename=f"./{expt_path}/agent{i}_val_temp.txt", pddl_problem_filename=f"./{expt_path}/agent{i}_task_temp.txt", pddl_problem_filename_edited=f"./{expt_path}/agent{i}_new_task_temp.txt", env_conds_only=False)

def generate_agent_subsets(active_agents):
    """Generate valid combinations of agents"""
    all_subsets = []
    for r in range(len(active_agents), 0, -1):
        all_subsets.extend(itertools.combinations(active_agents, r))
    return all_subsets