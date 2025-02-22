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
from itertools import permutations

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

# Global cache to store (cost, plan) for every state.
dp_cache = {}

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
        
    try:
        planner_search_time_1st_plan = float(output.split('Actual search time: ')[1].split('\n')[0].strip()[:-1])
        planner_total_time = float(output.split('Planner time: ')[1].split('\n')[0].strip()[:-1])
        planner_total_time_opt = float(output.split('Actual search time: ')[-1].split('\n')[0].strip()[:-1])
        first_plan_cost = int(output.split('Plan cost: ')[1].split('\n')[0].strip())
    except:
        print("planner broke, likely timed out.")
        print(output)
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
    print("Initial Plan (Agent 1):")
    print(plan_tostring(initial_plan))
    print()

    # First merge plans 2 through n
    for i in range(2, args.num_agents):
        print(f"merging plan {i} into initial_plan")
        next_agent_plan = [('single', i, agent_plans[i][j]) for j in range(len(agent_plans[i]))]
        # Clear the dp cache for each merge stage
        global dp_cache
        dp_cache = {}
        cost, merged_plan = validator_sim_recursion_function(expt_path, domain_pddl_file,
                                tuple([0] * 2), (tuple(initial_plan), tuple(next_agent_plan)),
                                tuple([task] * 2))
        success = cost < float('inf')

        print(cost, success)
        if not success:
            print("Merge failed - no valid solution found")
            return float('inf'), False
            
        initial_plan = merged_plan

        # Validate the entire plan as before.
        val_paths = [f"./{expt_path}/agent{i}_val_final_temp.txt" for i in range(args.num_agents)]
        plan_paths = [f"./{expt_path}/agent{i}_plan_final_temp.txt" for i in range(args.num_agents)]
        
        print("verifying all individual plans.")
        for i in range(args.num_agents):
            with open(plan_paths[i], 'w') as f:
                # For each action in the merged plan, determine what to write.
                for action in merged_plan:
                    if action[0] == 'single':
                        f.write(action[2])
                    elif action[0] == 'parallel':
                        # Extract the sub-action corresponding to agent i.
                        sub_action_str = ""
                        for (a_idx, act_str) in action[1]:
                            if a_idx == i:
                                sub_action_str = act_str
                                break
                        f.write(sub_action_str)
            output = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_pddl_file, plan_paths[i]],
                                    capture_output=True, text=True)
            with open(val_paths[i], 'w') as f:
                f.write(output.stdout)
            if 'unsatisfied precondition' in output.stdout:
                print("unsatisfied precondition encountered in merged plan, stopping process.")
                #print(output.stdout)
                return float('inf'), False

    # Finally merge with agent 0's plan
    print("merging main agent's plan into combined plan")
    agent_0_plan = [('single', 0, agent_plans[0][j]) for j in range(len(agent_plans[0]))]
    print("\nMain Agent's Plan:")
    print(plan_tostring(agent_0_plan))
    print("\nCurrent Combined Plan:")
    print(plan_tostring(initial_plan))
    print()
    
    dp_cache = {}
    cost, merged_plan = validator_sim_recursion_function(expt_path, domain_pddl_file,
                                tuple([0] * 2), (tuple(initial_plan), tuple(agent_0_plan)),
                                tuple([task] * 2))
    success = cost < float('inf')
    print(cost, success)
    
    if success:
        val_paths = [f"./{expt_path}/agent{i}_val_final_temp.txt" for i in range(args.num_agents)]
        plan_paths = [f"./{expt_path}/agent{i}_plan_final_temp.txt" for i in range(args.num_agents)]
        
        print("verifying final merged plan constituents.")
        for i in range(args.num_agents):
            with open(plan_paths[i], 'w') as f:
                for action in merged_plan:
                    if action[0] == 'single':
                        f.write(action[2])
                    elif action[0] == 'parallel':
                        # For a parallel step, extract the sub-action for agent i.
                        sub_action_str = ""
                        for (a_idx, act_str) in action[1]:
                            if a_idx == i:
                                sub_action_str = act_str
                                break
                        f.write(sub_action_str)
        for i in range(args.num_agents):
            output = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_pddl_file, plan_paths[i]],
                                    capture_output=True, text=True)
            with open(val_paths[i], 'w') as f:
                f.write(output.stdout)
            if 'Plan invalid' in output.stdout:
                print("Plan invalid, final plan is invalid.")
                return float('inf'), False
        
        print("\nFinal Merged Plan:")
        print(plan_tostring(merged_plan))
        print(f"\nTotal Cost: {cost}")
        initial_plan = merged_plan
    else:
        print("Final merge failed - no valid solution found")
    
    return cost, success

#@lru_cache(maxsize=None)
def validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute=None):
    """
    Recursively computes the optimal plan (as a tuple (cost, plan)) starting from the given state.
    Each action taken (single or parallel) gets recorded in the returned plan.
    """
    num_agents = 2  # used in merging two plans
    state_key = indices + (agent_to_execute if agent_to_execute is not None else num_agents,)
    if state_key in dp_cache:
        return dp_cache[state_key]
    if all(indices[i] == len(agent_plans[i]) for i in range(num_agents)):
        return (0, [])
    if agent_to_execute is not None:
        # print("executing agent", agent_to_execute)
        result = execute_agent_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute)
        dp_cache[state_key] = result
        return result
    else:
        options = []
        for i in range(num_agents):
            if indices[i] < len(agent_plans[i]):
                res = validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, i)
                options.append(res)
        res_all = execute_all_agents_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks)
        options.append(res_all)
        best = min(options, key=lambda x: x[0])
        dp_cache[state_key] = best
        return best

def execute_agent_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute):
    """
    Executes the next action for a single agent.
    When the action is parallel, instead of flattening the whole parallel step,
    we extract the sub-action corresponding to this agent.
    The validated action is then recorded and the agent's task state updated.
    """
    # Check if the agent has finished its plan.
    if indices[agent_to_execute] >= len(agent_plans[agent_to_execute]):
        # Agent is finished; simply return the cost/plan for the current state.
        return validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute=None)
    
    current_action = agent_plans[agent_to_execute][indices[agent_to_execute]]
    
    # Setup common file paths.
    val_path = f"./{expt_path}/agent{agent_to_execute}_val_temp.txt"
    plan_path = f"./{expt_path}/agent{agent_to_execute}_plan_temp.txt"
    task_paths = [f"./{expt_path}/agent{i}_task_temp.txt" for i in range(len(agent_tasks))]
    
    # Write the current tasks.
    for i, task in enumerate(agent_tasks):
        with open(task_paths[i], 'w') as f:
            f.write(task)
    
    with open(plan_path, 'w') as f:
        if isinstance(current_action, tuple) and current_action[0] == 'parallel':
            # Instead of flattening, find the sub-action for this agent.
            found = False
            for (a_idx, act_str) in current_action[1]:
                if a_idx == agent_to_execute:
                    current_action_str = act_str
                    found = True
                    break
            if not found:
                return (float('inf'), [])
            f.write(current_action_str)
        elif current_action[0] == 'single':
            current_action_str = current_action[2]
            f.write(current_action_str)
        else:
            return (float('inf'), [])
    
    output = subprocess.run(["./downward/validate", "-v", domain_pddl_file,
                              task_paths[agent_to_execute], plan_path],
                              capture_output=True, text=True)
    with open(val_path, 'w') as f:
        f.write(output.stdout)
    
    if 'unsatisfied precondition' in output.stdout:
        print(f"[Agent {agent_to_execute}] Unsatisfied preconditions encountered:")
        return (float('inf'), [])
    
    with open(log_file, 'a+') as f:
        f.write(f"Agent {agent_to_execute}, {indices[agent_to_execute]}, {current_action_str}\n")
    
    # Update the task file for every agent.
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
    
    # Progress to the next state for this agent.
    new_indices = list(indices)
    new_indices[agent_to_execute] += 1
    child_cost, child_plan = validator_sim_recursion_function(
                                expt_path, domain_pddl_file,
                                tuple(new_indices), agent_plans, tuple(new_task_states),
                                agent_to_execute=None)
    return (child_cost + 1, [('single', agent_to_execute, current_action_str)] + child_plan)

def execute_all_agents_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks):
    """
    Rather than flattening any parallel steps here, we pass control to execute_subset_agents for all agents.
    (Given that in our DP recursion, merging is always done pairwise, this subset is typically all agents.)
    """
    subset = list(range(len(agent_plans)))
    return execute_subset_agents(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, subset)

def execute_subset_agents(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, subset, sub_actions=None):
    """
    Processes a subset of agent actions concurrently without flattening a parallel step.
    If sub_actions is provided, this dict maps agent index -> the (parallel) action string.
    Otherwise, for each agent in the subset the function extracts its current action from agent_plans:
      - If the action is 'single', its action string is used.
      - If the action is 'parallel', we extract the sub-action for that agent.
    
    After validating each agent's action concurrently, we update the tasks and indices and invoke the DP recursion.
    """
    # First, filter out agents that have finished their plan.
    active_subset = [i for i in subset if indices[i] < len(agent_plans[i])]
    if not active_subset:
        # If no agents remain active, simply delegate back to recursion.
        return validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute=None)
    if len(active_subset) == 1:
        # Only one active agent remains; delegate to single-agent execution.
        return execute_agent_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, active_subset[0])
    
    subset = active_subset
    best_option = (float('inf'), None)  # (cost, plan)
    from itertools import permutations
    # Try all orderings of the active subset.
    for order in permutations(subset):
        temp_all_valid = True
        parallel_actions = []   # collect list of (agent_index, action_str) tuples
        task_paths = {}
        plan_paths = {}
        val_paths = {}
        new_task_paths = {}
        # Write current task files for agents in the subset.
        for i in subset:
            task_paths[i] = f"./{expt_path}/agent{i}_task_temp.txt"
            plan_paths[i] = f"./{expt_path}/agent{i}_plan_temp.txt"
            val_paths[i]  = f"./{expt_path}/agent{i}_val_temp.txt"
            new_task_paths[i] = f"./{expt_path}/agent{i}_new_task_temp.txt"
            with open(task_paths[i], 'w') as f:
                f.write(agent_tasks[i])
        # Process agents in the current permutation order.
        for i in order:
            # Now that we filtered the subset, each agent i should have a next action.
            if sub_actions is not None and i in sub_actions:
                action_str = sub_actions[i]
            else:
                current_action = agent_plans[i][indices[i]]
                if current_action[0] == 'single':
                    action_str = current_action[2]
                elif current_action[0] == 'parallel':
                    # Instead of flattening, extract the sub-action corresponding to this agent.
                    found = False
                    for (a_idx, act_str) in current_action[1]:
                        if a_idx == i:
                            action_str = act_str
                            found = True
                            break
                    if not found:
                        action_str = ""
                else:
                    temp_all_valid = False
                    break
            with open(plan_paths[i], 'w') as f:
                f.write(action_str)
            output = subprocess.run(["./downward/validate", "-v", domain_pddl_file,
                                     task_paths[i], plan_paths[i]],
                                     capture_output=True, text=True)
            with open(val_paths[i], 'w') as f:
                f.write(output.stdout)
            if 'unsatisfied precondition' in output.stdout:
                print(f"[Subset Agent {i}] Unsatisfied preconditions encountered:")
                temp_all_valid = False
                break
            parallel_actions.append((i, action_str))
            # Update the agent's task state using the validator output.
            get_updated_init_conditions(expt_path,
                                        validation_filename=val_paths[i],
                                        pddl_problem_filename=task_paths[i],
                                        pddl_problem_filename_edited=new_task_paths[i],
                                        env_conds_only=False)
            for j in subset:
                get_updated_init_conditions_recurse(expt_path,
                                                    validation_filename=val_paths[i],
                                                    pddl_problem_filename=task_paths[j],
                                                    pddl_problem_filename_edited=new_task_paths[j],
                                                    env_conds_only=True)
        if not temp_all_valid:
            continue
        new_task_states = list(agent_tasks)
        for i in subset:
            with open(new_task_paths[i], 'r') as f:
                new_task_states[i] = f.read()
        # Update indices only for agents in this subset.
        new_indices = list(indices)
        for i in subset:
            if indices[i] < len(agent_plans[i]):
                new_indices[i] += 1
        # Call the DP recursion with the updated indices and task states.
        child_cost, child_plan = validator_sim_recursion_function(
                                        expt_path, domain_pddl_file, tuple(new_indices),
                                        agent_plans, tuple(new_task_states),
                                        agent_to_execute=None)
        total_cost = child_cost + 1
        if total_cost < best_option[0]:
            best_option = (total_cost, [('parallel', parallel_actions)] + child_plan)
        if best_option[0] < float('inf'):
            break  # choose the first valid ordering.
    if best_option[0] == float('inf'):
         return float('inf'), []
    return best_option

def flatten_plan(plan):
    out = ""
    for i in range(len(plan)):
        out += str(plan[i][1])
    # print("flattened plan")
    # print(out)
    return out

def plan_tostring(plan):
    lines = []
    step_number = 1

    for action in plan:
        action_type = action[0]

        if action_type == 'single':
            # action also holds an agent index and an action string.
            agent_index = action[1]
            action_str = action[2].strip()
            # Check if the action string contains multiple steps (separated by newlines)
            subactions = [line.strip() for line in action_str.split('\n') if line.strip()]
            if len(subactions) > 1:
                lines.append(f"{step_number} (Parallel, derived from single action with multiple steps):")
                for subaction in subactions:
                    lines.append(f"  Agent {agent_index}: {subaction}")
            else:
                lines.append(f"{step_number} (Single): Agent {agent_index}: {subactions[0]}")
        elif action_type == 'parallel':
            # action[1] is expected to be a list of (agent_index, action_string) tuples.
            lines.append(f"{step_number} (Parallel):")
            for agent_index, action_str in action[1]:
                lines.append(f"  Agent {agent_index}: {action_str.strip()}")
        else:
            lines.append(f"{step_number}: Unknown action format: {action}")

        step_number += 1

    return "\n".join(lines)
