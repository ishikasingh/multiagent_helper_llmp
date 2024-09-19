import argparse
import glob
import json
import os
from dotenv import load_dotenv
import random
import sys
import time
import re
import subprocess
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import copy

load_dotenv()
client = OpenAI()

# import alfworld.agents
# from alfworld.info import ALFWORLD_DATA
# from alfworld.env.thor_env import ThorEnv
# from alfworld.env.thor_env_multiagent import ThorEnvMultiAgent
# from alfworld.agents.detector.mrcnn import load_pretrained_model
# from alfworld.agents.controller import OracleAgent, OracleAStarAgent, MaskRCNNAgent, MaskRCNNAStarAgent
# from alfworld.gen.planner.ff_planner_handler import parse_action_arg

FAST_DOWNWARD_ALIAS = "lama"

DOMAINS = [ ## .nl not changed for multi excpet gripper, since planner doesnt use it
    "barman",
    "blocksworld",
    # "floortile",
    # "storage",
    "termes",
    "tyreworld",
    "grippers",
    "barman-multi",
    "blocksworld-multi",
    "termes-multi",
    "tyreworld-multi",
    "grippers-multi",
]

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

def get_cost(x):
    splitted = x.split()
    counter = 0
    found = False
    cost = 1e5
    for i, xx in enumerate(splitted):
        if xx == "cost":
            counter = i
            found = True
            break
    if found:
        cost = float(splitted[counter+2])
    return cost

# class Planner:
#     def __init__(self, domain, expt_path, args):
#         self.domain = domain
#         self.


def query(prompt_text, system_text=None, use_chatgpt=False):
    server_flag = 0
    server_cnt = 0
    # import ipdb; ipdb.set_trace()
    while server_cnt < 10:
        try:
            if use_chatgpt: # currently, we will always use chatgpt
                response = client.chat.completions.create(model="gpt-4",
                temperature=0.1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt_text},
                ])
                result_text = response.choices[0].message.content
            else:
                response =  client.completions.create(model="text-davinci-003",
                prompt=prompt_text,
                temperature=0.0,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0)
                result_text = response.choices[0].text
            server_flag = 1
            if server_flag:
                break
        except Exception as e:
            server_cnt += 1
            print(e)
    # print(result_text)
    return result_text


def planner(expt_path, args, subgoal_idx=-1, time_limit=200):
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
        print("reattempting planner")
        os.system(f"python ./downward/fast-downward.py --alias {FAST_DOWNWARD_ALIAS} " + \
              f"--search-time-limit {args.time_limit} --plan-file {plan_file_name} " + \
              f"--sas-file {sas_file_name} " + \
              f"{domain_pddl_file} {task_pddl_file_name} > {output_path}")
        with open(output_path, "r") as f:
            output = f.read()
        
    planner_search_time_1st_plan = float(output.split('Actual search time: ')[1].split('\n')[0].strip()[:-1])
    print("successfully listed")
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
            cost = get_cost(plans[-1])
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

def validator(expt_path, args, subgoal_idx=-1):
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
            cost = get_cost(plans[-1])
            if cost < best_cost:
                best_cost = cost
                plan_file = fn
    # print("plan_file", plan_file)
    result = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_pddl_file, plan_file], stdout=subprocess.PIPE)
    #print("validated")
    output = result.stdout.decode('utf-8')
    output_file.write(output)
    if "Plan valid" in result.stdout.decode('utf-8'):
        return True
    else:
        return False

def get_updated_init_conditions(expt_path, args, validation_filename=None, pddl_problem_filename=None, pddl_problem_filename_edited=None, env_conds_only=True, is_main=False):
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

def get_updated_init_conditions_recurse(expt_path, args, validation_filename=None, pddl_problem_filename=None, pddl_problem_filename_edited=None, env_conds_only=True):
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

def get_pddl_goal(expt_path, args, helper_subgoal, log_file):
    # taken from (LLM+P), create the problem PDDL given the context
    context_nl_filename = f"./domains/{args.domain}/p{args.task_id}.nl"
    with open(context_nl_filename, 'r') as f:
        context_nl = f.read()
    #natural language plan
    context_nl = 'Your goal is to' + context_nl.split('Your goal is to')[-1]
    #print("context_nl", context_nl)
    context_pddl_filename = f"./domains/{args.domain}/p{args.task_id}.pddl"
    with open(context_pddl_filename, 'r') as f:
        context_pddl = f.read()
    context_pddl_init, context_pddl_goal = context_pddl.split('(:goal')
    context_pddl_goal = f'(:goal\n{context_pddl_goal.strip()[:-1]}'

    pddl_problem_filename_arr = []
    
    pddl_problem_filename = f""
    system_text = 'I want you to solve planning problems. Provide me with the PDDL goal that describes the new planning goal directly without further explanations. Make sure to provide only non-conflicting, necessary, and final goal conditions mentioned in the given goal.'

    start=time.time()

    for i in range(1, args.num_agents):

        pddl_problem_filename = f"./{expt_path}/p{args.task_id}_{i}.pddl"

        prompt_text = f"The PDDL problem and its initial conditions are given as: \n{context_pddl_init.strip()} \n\n" + \
                    f"An example planning goal for this problem:  \n{context_nl.strip()} \n\n\n" + \
                    f"The PDDL goal for the example planning goal:  \n{context_pddl_goal.strip()} \n\n\n" + \
                    f"New planning goal for the same problem:\n Your goal is: {helper_subgoal[i-1].strip()} \n\n" + \
                    f'The PDDL goal for the new planning goal:\n'

        #print(f"agent {i} prompt \n", prompt_text)
        # import ipdb; ipdb.set_trace()
        # print("\n natural language to pddl prompt \n", prompt_text)
        pddl_goal = query(prompt_text, system_text=system_text, use_chatgpt=True)
        # import ipdb; ipdb.set_trace()
        # remove undefined goal conditions using domain predicate list
        if args.domain == 'tyreworld':
            pddl_goal = pddl_goal.replace('(empty hands)', '').replace('(empty-hand)', '').replace('(empty-hands)', '')
        with open(log_file, 'a+') as f:
            f.write(f"\n\n{pddl_goal}")
        with open(pddl_problem_filename, 'w') as f:
            f.write(context_pddl_init + pddl_goal + ')')

        pddl_problem_filename_arr.append(pddl_problem_filename)
    
    end = time.time()-start
    return pddl_problem_filename_arr, end

# deleted get_pddl_expert , may need to restore this for benchmarking
# deleted get_helper_subgoal > asks for pddl directly from helper, instead of breaking it down

def get_helper_subgoal_without_plan(expt_path, args, log_file):

    system_text = '''Main agent: agent0
                    Helper agent: ''' 

    for i in range(1,args.num_agents):
        system_text += f' agent{i}, '
    
    system_text += '\n'

    system_text += ''' Your goal is to generate goals for agents such that they can be executed in parallel to decrease plan execution length. Generate only one clearly stated small independent subgoal for each helper agent to help main agent complete the given task.
      The subgoal should not rely on any main agent actions, and should be executable by the helper agents independently without waiting for any main agent actions.
     The subgoal should be clearly stated with unambiguous terminology. Do not use actions like assist or help. Generate actions that the helper agents can do independently, 
     based on the given steps for completing the task. The main agent should be able to continue working on the remaining task while each of the helper agents is completing its small subgoal.
     Do not overtake the full sequence of actions. Remember, the helper agents are only assisting the main agent and act agnostically to the main agent.'''
    
    # print("system_text \n", system_text, "\n")
    
    # default barman prompts
    prompt_text = f'''Example  domain scenario:
    You have 1 shaker with 3 levels, 3 shot glasses, 3 dispensers for 3 ingredients. 
    The shaker and shot glasses are clean, empty, and on the table. Your left and right hands are empty. 
    The first ingredient of cocktail1 is ingredient3. The second ingredient of cocktail1 is ingredient1. 
    Your goal is to make 1 cocktail. 
    shot1 contains cocktail1. 

    agent0 takes the following steps to complete the above task:
    grasp right shot2
    fill-shot shot2 ingredient1 right left dispenser1
    pour-shot-to-clean-shaker shot2 ingredient1 shaker1 right l0 l1
    clean-shot shot2 ingredient1 right left
    fill-shot shot2 ingredient3 right left dispenser3
    grasp left shaker1
    pour-shot-to-used-shaker shot2 ingredient3 shaker1 right l1 l2
    leave right shot2
    shake cocktail1 ingredient3 ingredient1 shaker1 left right
    pour-shaker-to-shot cocktail1 shot1 left shaker1 l2 l1

    Now we have a new problem defined in this domain for which we don't have access to the single agent plan:
    You have 1 shaker with 3 levels, 3 shot glasses, 3 dispensers for 3 ingredients. 
    The shaker and shot glasses are clean, empty, and on the table. Your left and right hands are empty. 
    The first ingredient of cocktail1 is ingredient3. The second ingredient of cocktail1 is ingredient1. 
    The first ingredient of cocktail2 is ingredient1. The second ingredient of cocktail2 is ingredient2. 
    Your goal is to make 2 cocktails. 
    shot1 contains cocktail1. shot2 contains cocktail2. 

    A possible agent1 subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - 
    agent1 subgoals: It can help in filling ingredient1 in a shot glass, then pour it in shaker1, while agent0 prepares other cocktail ingredients using other objects. In this way, agent1 would not need to wait for agent0 and it can complete its goal independently. agent1 should also release all objects that the main agent might need for its own actions. Therefore, agent1's clearly stated (with object names) complete and final goal condition is: shaker1 contains ingredient1 and all hands are empty.
    A possible agent2 subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - 
    agent2 subgoals: It can help in filling ingredient3 in a shot glass, while agent0 and agent1 prepare other cocktail ingredients using other objects. In this way, agent2 would not need to wait for agent1 and agent0 and it can complete its goal independently. agent2 should also release all objects that the main agent might need for its own actions. Therefore, agent2's clearly stated (with object names) complete and final goal condition is: shotglass3 contains ingredient3 and all hands are empty.
    '''
    # This pattern continues until {args.num_agents - 1} subgoals are generated, or until it is unnecessary to generate more agents.
    # get natural language descriptions of current domain task
    scenario_filename =  f"./domains/{args.domain}/p{args.task_id}.nl"
    with open(scenario_filename, 'r') as f:
        current_scenario = f.read()
    if args.domain != 'barman':
        scenario_filename =  f"./domains/{args.domain}/p_example.nl"
        with open(scenario_filename, 'r') as f:
            current_scenario_example = f.read()

        singleagent_example_plan_filename = f"./domains/{args.domain}/p_example.sol"

        with open(singleagent_example_plan_filename, 'r') as f:
            singleagent_plan = f.read()
        singleagent_plan = singleagent_plan.split(';')[0]
        current_prompt_text = '\n\nCurrent domain scenario:\n'
        current_prompt_text += f'{current_scenario_example.strip()}\n\n'
        current_prompt_text += 'agent0 takes the following steps to complete the above task:\n'
        current_prompt_text += f'{singleagent_plan.strip()}\n\n'
        current_prompt_text += 'Now we have a new problem defined in this domain for which we don\'t have access to the single agent plan:\n'
    else:
        current_prompt_text = '\n\nNow we have another new problem defined in this domain for which we don\'t have access to the single agent plan:\n'
    current_prompt_text += f'{current_scenario.strip()}\n\n'
    current_prompt_text += f'Return only one clearly stated subgoal condition for one and only one agent without explanation or steps. A possible subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - \n'

    prompt_text = prompt_text + current_prompt_text
    # helper_subgoal = 'Fetch the intact tyre from the boot, inflate the intact tyre, and put on the intact tyre on the hub.'
    # print("\n prompt_text for helper_sg w/o plan \n",prompt_text)
    # import ipdb; ipdb.set_trace()
    #print("prompt text\n", prompt_text)
    start = time.time()
    all_subgoals = []
    #helper_subgoal = query(prompt_text, system_text=system_text, use_chatgpt=True)
    for i in range(1,args.num_agents):
        prompt_text += f"\n agent{i} subgoal:"
        #print(f"querying for agent {i}")
        helper_subgoal = query(prompt_text, system_text=system_text, use_chatgpt=True)
        #print(helper_subgoal, "\n")
        prompt_text += helper_subgoal
        all_subgoals.append(helper_subgoal)
    end = time.time()-start

    #print(prompt_text)
    #print("cumulative subgoals", all_subgoals,"\n")

    with open(log_file, 'a+') as f: f.write(f"\n\n{current_prompt_text} {helper_subgoal}")
    helper_subgoal = helper_subgoal.split('final goal condition is:')[-1].strip()
    return all_subgoals, end


def validator_simulation_recursive(expt_path, args, log_file, multi=False):
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
                cost = get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    best_plan_file = fn
        if best_plan_file:
            with open(best_plan_file, 'r') as f:
                agent_plans.append(tuple(f.readlines()[:-1]))  # Convert to tuple
        else:
            return float('inf'), False

    print(f"TASK: {args.domain} - {args.run} - {args.task_id}")
    with open(log_file, 'a+') as f:
        f.write(f"TASK: {args.domain} - {args.run} - {args.task_id}\n")

    global execution_state
    execution_state = np.full([len(plan) + 1 for plan in agent_plans] + [args.num_agents + 1], float('inf'))

    plan_length = validator_sim_recursion_function(expt_path, domain_pddl_file, tuple([0] * args.num_agents), tuple(agent_plans), tuple([task] * args.num_agents))

    success = plan_length < float('inf')
    print(plan_length, success)
    return plan_length, success

@lru_cache(maxsize=None)
def validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute=None):
    num_agents = len(agent_plans)
    
    if all(indices[i] == len(agent_plans[i]) for i in range(num_agents)):
        return 0

    state_index = indices + (agent_to_execute if agent_to_execute is not None else num_agents,)
    if execution_state[state_index] != float('inf'):
        return execution_state[state_index]

    if agent_to_execute is not None:
        result = execute_agent_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, agent_to_execute)
    else:
        plans = []
        for i in range(num_agents):
            if indices[i] < len(agent_plans[i]):
                plans.append(validator_sim_recursion_function(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks, i))
        
        plans.append(execute_all_agents_action(expt_path, domain_pddl_file, indices, agent_plans, agent_tasks))

        result = 1 + min(plans)

    execution_state[state_index] = result
    return result

def execute_agent_action(expt_path, domain_pddl_file, indices, agent_plans, task_states, agent_index):
    val_path = f"./{expt_path}/agent{agent_index}_val_temp.txt"
    plan_path = f"./{expt_path}/agent{agent_index}_plan_temp.txt"
    task_paths = [f"./{expt_path}/agent{i}_task_temp.txt" for i in range(len(agent_plans))]

    with open(plan_path, 'w') as f:
        f.write(agent_plans[agent_index][indices[agent_index]])
    for i, task in enumerate(task_states):
        with open(task_paths[i], 'w') as f:
            f.write(task)

    output = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_paths[agent_index], plan_path], capture_output=True, text=True)
    with open(val_path, 'w') as f:
        f.write(output.stdout)

    if 'unsatisfied precondition' not in output.stdout:
        with open(log_file, 'a+') as f:
            f.write(f"Agent {agent_index}, {indices[agent_index]}, {agent_plans[agent_index][indices[agent_index]][:-1]}\n")

        new_task_states = list(task_states).copy()
        for i in range(len(agent_plans)):
            new_task_path = f"./{expt_path}/agent{i}_new_task_temp.txt"
            get_updated_init_conditions_recurse(expt_path, args, validation_filename=val_path, pddl_problem_filename=task_paths[i], pddl_problem_filename_edited=new_task_path, env_conds_only=(i != agent_index))
            with open(new_task_path, 'r') as f:
                new_task_states[i] = f.read()

        new_indices = list(indices)
        new_indices[agent_index] += 1
        return validator_sim_recursion_function(expt_path, domain_pddl_file, tuple(new_indices), agent_plans, tuple(new_task_states))
    else:
        return float('inf')

def execute_all_agents_action(expt_path, domain_pddl_file, indices, agent_plans, task_states):
    val_paths = [f"./{expt_path}/agent{i}_val_temp.txt" for i in range(len(agent_plans))]
    plan_paths = [f"./{expt_path}/agent{i}_plan_temp.txt" for i in range(len(agent_plans))]
    task_paths = [f"./{expt_path}/agent{i}_task_temp.txt" for i in range(len(agent_plans))]
    new_task_paths = [f"./{expt_path}/agent{i}_new_task_temp.txt" for i in range(len(agent_plans))]

    all_valid = True
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(agent_plans)):
            if indices[i] < len(agent_plans[i]):
                with open(plan_paths[i], 'w') as f:
                    f.write(agent_plans[i][indices[i]])
                with open(task_paths[i], 'w') as f:
                    f.write(task_states[i])
                
                futures.append(executor.submit(subprocess.run, ["./downward/validate", "-v", domain_pddl_file, task_paths[i], plan_paths[i]], capture_output=True, text=True))

        for i, future in enumerate(futures):
            output = future.result()
            with open(val_paths[i], 'w') as f:
                f.write(output.stdout)
            
            if 'unsatisfied precondition' in output.stdout:
                all_valid = False
                break

            get_updated_init_conditions_recurse(expt_path, args, validation_filename=val_paths[i], pddl_problem_filename=task_paths[i], pddl_problem_filename_edited=new_task_paths[i], env_conds_only=False)
            for j in range(len(agent_plans)):
                if i != j:
                    get_updated_init_conditions_recurse(expt_path, args, validation_filename=val_paths[i], pddl_problem_filename=task_paths[j], pddl_problem_filename_edited=new_task_paths[j])

    if all_valid:
        with open(log_file, 'a+') as f:
            for i in range(len(agent_plans)):
                if indices[i] < len(agent_plans[i]):
                    f.write(f"Agent {i}, {indices[i]}, {agent_plans[i][indices[i]][:-1]}\n")

        new_indices = tuple(idx + 1 if idx < len(plan) else idx for idx, plan in zip(indices, agent_plans))
        new_task_states = []
        for path in new_task_paths:
            with open(path, 'r') as f:
                new_task_states.append(f.read())

        return validator_sim_recursion_function(expt_path, domain_pddl_file, new_indices, agent_plans, tuple(new_task_states))
    else:
        return float('inf')

if __name__ == "__main__":
    # parse arguments, define domain, and create experiment folder
    parser = argparse.ArgumentParser(description="LLM-multiagent-helper")
    parser.add_argument('--domain', type=str, choices=DOMAINS, default="tyreworld")
    parser.add_argument('--time-limit', type=int, default=200)
    parser.add_argument('--task_id', type=str)
    parser.add_argument('--experiment_folder', type=str, default='experiments_multiagent_help')
    parser.add_argument('--human_eval', type=bool, default=False)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--num_agents', type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.experiment_folder):
        os.mkdir(args.experiment_folder)
    base_path = os.path.join(args.experiment_folder, f'run{args.run}')
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    # for domain in DOMAINS:

    singleagent_planning_time = []
    singleagent_planning_time_opt = []
    singleagent_cost = []
    singleagent_planning_time_1st = []
    singleagent_cost_1st = []

    multiagent_helper_planning_time = []
    multiagent_helper_planning_time_opt = []
    multiagent_helper_cost = []
    multiagent_helper_planning_time_1st = []
    multiagent_helper_cost_1st = []
    multiagent_helper_success = []

    LLM_text_sg_time = []
    LLM_pddl_sg_time = []

    multiagent_main_planning_time = []
    multiagent_main_planning_time_opt = []
    multiagent_main_cost = []
    multiagent_main_planning_time_1st = []
    multiagent_main_cost_1st = []
    multiagent_main_success = []

    overall_plan_length = []

    # args.domain = domain
    # print("task domain", args.domain)
    path = os.path.join(base_path, args.domain)
    if not os.path.exists(path):
        os.mkdir(path) 
    log_file = os.path.join(base_path, f'helper_logs_{args.domain}_exec_lens.log')
    with open(log_file, 'w') as f: f.write(f"start_eval\n")

    human_eval_task_ids = [1, 6, 11, 16]

    # if task is not provided, evaluate all tasks 1-20
    task_list = [args.task_id] if args.task_id != None else range(1, 21)

    for task_id in task_list: # 1, 21
        args.task_id = str(task_id)

        if args.human_eval: #human eval
            args.task_id = str(human_eval_task_ids[task_id])

        args.task_id = f'0{args.task_id}' if len(args.task_id)==1 else args.task_id

        # normal planning and same for multi-agent planning
        try:
            # hard coded time as 10? Shouldn't this be args.time_limit, ask Ishika
            planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner(path, args, time_limit=10)
            singleagent_planning_time.append(planner_total_time)
            singleagent_planning_time_opt.append(planner_total_time_opt)
            singleagent_cost.append(best_cost)
            singleagent_planning_time_1st.append(planner_search_time_1st_plan)
            singleagent_cost_1st.append(first_plan_cost)
        except:
            singleagent_planning_time.append(args.time_limit)
            singleagent_planning_time_opt.append(args.time_limit)
            singleagent_cost.append(1e6)
            singleagent_planning_time_1st.append(args.time_limit)
            singleagent_cost_1st.append(1e6)

        # half split baseline
        # plan_length, success = validator_simulation(path, args, log_file, half_split=True)
        # with open(log_file, 'a+') as f: f.write(f"plan_length {plan_length}\n") # {singleagent_cost[-1]}\n")
        # overall_plan_length.append(plan_length)
        # multiagent_main_success.append(success)

        # # # multiagent pddl
        # plan_length, success = validator_simulation_recursive(path, args, log_file, multi=True)
        # with open(log_file, 'a+') as f: f.write(f"plan_length {plan_length}\n") # {singleagent_cost[-1]}\n")
        # overall_plan_length.append(plan_length)
        # multiagent_main_success.append(success)

        # helper_subgoal, t1 = get_helper_subgoal_without_plan(path, args, log_file)
        # _, t2 = get_pddl_goal(path, args, helper_subgoal, log_file)
        try:
            subgoal_array, t1 = get_helper_subgoal_without_plan(path, args, log_file)
            # add check for validity of all goals
            print(subgoal_array)
            # # helper_subgoal = "xyz"
            goal_files, t2 = get_pddl_goal(path, args, subgoal_array, log_file)
            print(goal_files)

        except Exception as e:
            print("LLM generation failed, ", e)
        
        # handle all subgoals and init conditions
        # edited init starts at  0 for original, then 1 for post-first subgoal, etc ...
        # subgoal 1 used original pddl domain, then subgoal 2 uses edited_init_1, 3 uses edited_init_2, etc ...
        for i in range(1,args.num_agents):
            print(f"agent{i}")
            try:
                planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner(path, args, subgoal_idx=i)
                print("planner successful")
                success = validator(path, args, subgoal_idx=i)

                LLM_text_sg_time.append(t1)
                LLM_pddl_sg_time.append(t2)
                multiagent_helper_planning_time.append(planner_total_time)
                multiagent_helper_planning_time_opt.append(planner_total_time_opt)
                multiagent_helper_cost.append(best_cost)
                multiagent_helper_planning_time_1st.append(planner_search_time_1st_plan)
                multiagent_helper_cost_1st.append(first_plan_cost)
                multiagent_helper_success.append(success)
                # print(LLM_text_sg_time)
                # print(LLM_pddl_sg_time)
                # print(multiagent_helper_planning_time)
                # print(multiagent_helper_planning_time_opt)
                # print(multiagent_helper_cost)
                # print(multiagent_helper_planning_time_1st)
                # print(multiagent_helper_cost_1st)
                # print(multiagent_helper_success)
            except Exception as e:
                LLM_text_sg_time.append(-1)
                LLM_pddl_sg_time.append(-1)
                multiagent_helper_planning_time.append(-1)
                multiagent_helper_planning_time_opt.append(-1)
                multiagent_helper_cost.append(-1)
                multiagent_helper_planning_time_1st.append(-1)
                multiagent_helper_cost_1st.append(-1)
                multiagent_helper_success.append(0)
                print(e)
                with open(log_file, 'a+') as f:
                    f.write(f"\n\nError: {e}")

            init_problem = f"./{path}/p{args.task_id}_{i}.pddl"
            init_problem_out = f"./{path}/p{args.task_id}_{i+1}.pddl"
            if i == 1:
                init_problem = f"./domains/{args.domain}/p{args.task_id}.pddl"
            main_goal = False
            if args.num_agents == i+1: 
                main_goal = True 
                init_problem_out = f"./{path}/p{args.task_id}_0.pddl"
            get_updated_init_conditions(path, args, validation_filename=f"./{path}/p{args.task_id}_{i}_validation.txt", pddl_problem_filename=init_problem, pddl_problem_filename_edited=init_problem_out,is_main=main_goal)

        # handle main agent
        try:
            # init conditions should be good from last iter of subgoal loop
            # add goal to main agent 
            planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner(path, args, subgoal_idx=args.num_agents)
            # print("running validator_sim_recursion_function")
            plan_length, success = validator_simulation_recursive(path, args, log_file)
            with open(log_file, 'a+') as f: f.write(f"plan_length {plan_length})\n") # {singleagent_cost[-1]}\n")
            multiagent_main_planning_time.append(planner_total_time)
            multiagent_main_planning_time_opt.append(planner_total_time_opt)
            multiagent_main_cost.append(best_cost)
            multiagent_main_planning_time_1st.append(planner_search_time_1st_plan)
            multiagent_main_cost_1st.append(first_plan_cost)
            multiagent_main_success.append(success)
            overall_plan_length.append(plan_length)
        except Exception as e:
            multiagent_main_planning_time.append(-1)
            multiagent_main_planning_time_opt.append(-1)
            multiagent_main_cost.append(-1)
            multiagent_main_planning_time_1st.append(-1)
            multiagent_main_cost_1st.append(-1)
            multiagent_main_success.append(0)
            overall_plan_length.append(-1)
            print(e)
            with open(log_file, 'a+') as f:
                f.write(f"\n\nError: {e}")


        # print(f"singleagent_planning_time = {singleagent_planning_time}")
        # print(f"singleagent_planning_time_opt = {singleagent_planning_time_opt}")
        # print(f"singleagent_cost = {singleagent_cost}")
        # print(f"singleagent_planning_time_1st = {singleagent_planning_time_1st}")
        # print(f"singleagent_cost_1st = {singleagent_cost_1st}")
        # print(f"LLM_text_sg_time = {LLM_text_sg_time}")
        # print(f"LLM_pddl_sg_time = {LLM_pddl_sg_time}")
        # print(f"multiagent_helper_planning_time = {multiagent_helper_planning_time}")
        # print(f"multiagent_helper_planning_time_opt = {multiagent_helper_planning_time_opt}")
        # print(f"multiagent_helper_cost = {multiagent_helper_cost}")
        # print(f"multiagent_helper_planning_time_1st = {multiagent_helper_planning_time_1st}")
        # print(f"multiagent_helper_cost_1st = {multiagent_helper_cost_1st}")
        # print(f"multiagent_helper_success = {multiagent_helper_success}")
        # print(f"multiagent_main_planning_time = {multiagent_main_planning_time}")
        # print(f"multiagent_main_planning_time_opt = {multiagent_main_planning_time_opt}")
        # print(f"multiagent_main_cost = {multiagent_main_cost}")
        # print(f"multiagent_main_planning_time_1st = {multiagent_main_planning_time_1st}")
        # print(f"multiagent_main_cost_1st = {multiagent_main_cost_1st}")
        # print(f"multiagent_main_success = {multiagent_main_success}")
        # print(f"overall_plan_length = {overall_plan_length}")

        with open(log_file, 'a+') as f:
            f.write("\n\n" +\
                f"singleagent_planning_time = {singleagent_planning_time}\n" + \
                    f"singleagent_planning_time_opt = {singleagent_planning_time_opt}\n" + \
                f"singleagent_cost = {singleagent_cost}\n" + \
                f"singleagent_planning_time_1st = {singleagent_planning_time_1st}\n" + \
                f"singleagent_cost_1st = {singleagent_cost_1st}\n" + \
                    f"LLM_text_sg_time = {LLM_text_sg_time}\n" + \
                    f"LLM_pddl_sg_time = {LLM_pddl_sg_time}\n" + \
                f"multiagent_helper_planning_time = {multiagent_helper_planning_time}\n" + \
                    f"multiagent_helper_planning_time_opt = {multiagent_helper_planning_time_opt}\n" + \
                f"multiagent_helper_cost = {multiagent_helper_cost}\n" + \
                f"multiagent_helper_planning_time_1st = {multiagent_helper_planning_time_1st}\n" + \
                f"multiagent_helper_cost_1st = {multiagent_helper_cost_1st}\n" + \
                f"multiagent_helper_success = {multiagent_helper_success}\n" + \
                f"multiagent_main_planning_time = {multiagent_main_planning_time}\n" + \
                        f"multiagent_main_planning_time_opt = {multiagent_main_planning_time_opt}\n" + \
                f"multiagent_main_cost = {multiagent_main_cost}\n" + \
                f"multiagent_main_planning_time_1st = {multiagent_main_planning_time_1st}\n" + \
                f"multiagent_main_cost_1st = {multiagent_main_cost_1st}\n" + \
                f"multiagent_main_success = {multiagent_main_success}\n" + \
                    f"overall_plan_length = {overall_plan_length}\n"
            )
    # import ipdb; ipdb.set_trace()
    # sr, gcr, total_planner_time, planner_steps_max =  evaluate()