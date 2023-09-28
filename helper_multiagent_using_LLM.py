import argparse
import glob
import json
import os
import random
import sys
import time
import re
import subprocess

import openai

import alfworld.agents
from alfworld.info import ALFWORLD_DATA
from alfworld.env.thor_env import ThorEnv
from alfworld.env.thor_env_multiagent import ThorEnvMultiAgent
from alfworld.agents.detector.mrcnn import load_pretrained_model
from alfworld.agents.controller import OracleAgent, OracleAStarAgent, MaskRCNNAgent, MaskRCNNAStarAgent
from alfworld.gen.planner.ff_planner_handler import parse_action_arg



FAST_DOWNWARD_ALIAS = "lama"

DOMAINS = [
    "barman",
    "blocksworld",
    "floortile",
    "grippers",
    "storage",
    "termes",
    "tyreworld",
    "alfred",
]

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
    openai.api_key = "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"
    while server_cnt < 10:
        try:
            if use_chatgpt: # currently, we will always use chatgpt
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    temperature=0.0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": prompt_text},
                    ],
                )
                result_text = response['choices'][0]['message']['content']
            else:
                response =  openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt_text,
                    temperature=0.0,
                    max_tokens=1024,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                result_text = response['choices'][0]['text']
            server_flag = 1
            if server_flag:
                break
        except Exception as e:
            server_cnt += 1
            print(e)
    return result_text


def planner(expt_path, args, subgoal=False, edited_init=False):
    domain_pddl_file =  f'./domains/{args.domain}/domain.pddl'

    if subgoal:
        task_pddl_file_name =  f"./{expt_path}/p{args.task_id}_subgoal.pddl"
        plan_file_name = f"./{expt_path}/p{args.task_id}_subgoal_plan.pddl"
    elif edited_init:
        task_pddl_file_name =  f"./{expt_path}/p{args.task_id}_edited_init.pddl"
        plan_file_name = f"./{expt_path}/p{args.task_id}_edited_init_plan.pddl"
    else:
        task_pddl_file_name =  f"./domains/{args.domain}/p{args.task_id}.pddl"
        plan_file_name = f"./{expt_path}/p{args.task_id}_plan.pddl"
    sas_file_name = plan_file_name + '.sas'

    start_time = time.time()

    # run fastforward to plan
    os.system(f"python ./downward/fast-downward.py --alias {FAST_DOWNWARD_ALIAS} " + \
              f"--search-time-limit {args.time_limit} --plan-file {plan_file_name} " + \
              f"--sas-file {sas_file_name} " + \
              f"{domain_pddl_file} {task_pddl_file_name}")

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
        print(f"[info] task {args.task_id} takes {end_time - start_time} sec, found a plan with cost {best_cost}")
        return end_time - start_time, best_cost
    else:
        print(f"[info] task {args.task_id} takes {end_time - start_time} sec, no solution found")


def validator(expt_path, args, subgoal=False, edited_init=False):
    if subgoal:
        output_path = f"./{expt_path}/p{args.task_id}_subgoal_validation.txt"
    elif edited_init:
        output_path = f"./{expt_path}/p{args.task_id}_edited_init_validation.txt"
    else:
        output_path = f"./{expt_path}/p{args.task_id}_validation.txt"
    output_file = open(output_path, "w")

    domain_pddl_file =  f'./domains/{args.domain}/domain.pddl'
    

    if subgoal:
        task_pddl_file =  f"./{expt_path}/p{args.task_id}_subgoal.pddl"
        plan_path = os.path.join(f"./{expt_path}", 
                                f"p{args.task_id}_subgoal_plan.pddl" + '.*')
    elif edited_init:
        task_pddl_file =  f"./{expt_path}/p{args.task_id}_edited_init.pddl"
        plan_path = os.path.join(f"./{expt_path}", 
                                f"p{args.task_id}_edited_init_plan.pddl" + '.*')
    else:
        task_pddl_file =  f"./{expt_path}/p{args.task_id}.pddl"
        plan_path = os.path.join(f"./{expt_path}", 
                                f"p{args.task_id}_plan.pddl" + '.*')
    
    plan_files = glob.glob(plan_path)
    plan_file = [plan for plan in plan_files if not plan.endswith('sas')][-1]
    result = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_pddl_file, plan_file], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    output_file.write(output)
    if "Plan valid" in result.stdout.decode('utf-8'):
        return True
    else:
        return False

def get_updated_init_conditions(expt_path, args):
    validation_filename = f"./{expt_path}/p{args.task_id}_subgoal_validation.txt"
    with open(validation_filename, 'r') as f:
        validation = f.readlines()
    
    pddl_problem_filename =  f"./domains/{args.domain}/p{args.task_id}.pddl"
    with open(pddl_problem_filename, 'r') as f:
        pddl_problem = f.read()
    pddl_problem = pddl_problem.split('(:init')
    pddl_problem[1:] = pddl_problem[1].split('(:goal')

    
    init_conditions  = set([cond.strip() for cond in pddl_problem[1].split('\n') if len(cond)>1])
    added_conditions = set([line.split('Adding')[1].strip() for line in validation if 'Adding' in line])
    deleted_conditions = set([line.split('Deleting')[1].strip() for line in validation if 'Deleting' in line])
    new_init_conditions  = list((init_conditions | added_conditions) - deleted_conditions)

    pddl_problem[1] = new_init_conditions
    pddl_problem = pddl_problem[0] + '(:init\n' + '\n'.join(pddl_problem[1]) + '\n)\n(:goal' + pddl_problem[2]

    pddl_problem_filename =  f"./{expt_path}/p{args.task_id}_edited_init.pddl"
    with open(pddl_problem_filename, 'w') as f:
        f.write(pddl_problem)

def get_pddl_problem(expt_path, args, helper_subgoal=None):
    # taken from (LLM+P), create the problem PDDL given the context
    context_nl_filename = f"./domains/{args.domain}/p_example.nl"
    with open(context_nl_filename, 'r') as f:
        context_nl = f.read()
    context_pddl_filename = f"./domains/{args.domain}/p_example.pddl"
    with open(context_pddl_filename, 'r') as f:
        context_pddl = f.read()
    task_nl_filename = f"./domains/{args.domain}/p{args.task_id}.nl"
    with open(task_nl_filename, 'r') as f:
        task_nl = f.read()
    
    pddl_problem_filename = f"./{expt_path}/p{args.task_id}.pddl"
    if helper_subgoal != None:
        task_nl = task_nl.split('Your goal is to ')[0] + 'Your goal is to ' + helper_subgoal
        pddl_problem_filename = f"./{expt_path}/p{args.task_id}_subgoal.pddl"
    
    if os.path.exists(pddl_problem_filename):
        return pddl_problem_filename
    
    system_text = f"I want you to solve planning problems. "
    prompt_text = f"An example planning problem is: \n {context_nl} \n" + \
                f"The problem PDDL file to this problem is: \n {context_pddl} \n" + \
                f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                f"Provide me with the problem PDDL file that describes " + \
                f"the new planning problem directly without further explanations? Only return the PDDL file. Do not return anything else."
    pddl_problem = query(prompt_text, system_text=system_text, use_chatgpt=True)
    import ipdb; ipdb.set_trace()

    with open(pddl_problem_filename, 'w') as f:
        f.write(pddl_problem)
    return pddl_problem_filename

def get_pddl_goal(expt_path, args, helper_subgoal=None):
    # taken from (LLM+P), create the problem PDDL given the context
    context_nl_filename = f"./domains/{args.domain}/p{args.task_id}.nl"
    with open(context_nl_filename, 'r') as f:
        context_nl = f.read()
    context_nl = 'Your goal is to' + context_nl.split('Your goal is to')[-1]
    context_pddl_filename = f"./domains/{args.domain}/p{args.task_id}.pddl"
    with open(context_pddl_filename, 'r') as f:
        context_pddl = f.read()
    context_pddl_init, context_pddl_goal = context_pddl.split('(:goal')
    context_pddl_goal = f'(:goal\n{context_pddl_goal.strip()[:-1]}'
    # task_nl_filename = f"./domains/{args.domain}/p{args.task_id}.nl"
    # with open(task_nl_filename, 'r') as f:
    #     task_nl = f.read()
    
    # pddl_problem_filename =  f"./domains/{args.domain}/p{args.task_id}.pddl"
    # with open(pddl_problem_filename, 'r') as f:
    #     pddl_problem = f.read()
    # pddl_problem = pddl_problem.split('(:goal')[0]

    # if helper_subgoal != None:
    #     task_nl = task_nl.split('Your goal is to ')[0] + 'Your goal is to ' + helper_subgoal
    pddl_problem_filename = f"./{expt_path}/p{args.task_id}_subgoal.pddl"
    
    # if os.path.exists(pddl_problem_filename):
    #     return pddl_problem_filename
    
    system_text = 'I want you to solve planning problems. Provide me with the PDDL goal that describes the new planning goal directly without further explanations. Make sure to provide only necessary and final goal conditions mentioned in the given goal.'
    prompt_text = f"The PDDL problem and its initial conditions are given as: \n{context_pddl_init.strip()} \n\n" + \
                f"An example planning goal for this problem:  \n{context_nl.strip()} \n\n\n" + \
                f"The PDDL goal for the example planning goal:  \n{context_pddl_goal.strip()} \n\n\n" + \
                f"New planning goal for the same problem: \n{helper_subgoal.strip()} \n\n" + \
                f'The PDDL goal for the new planning goal:\n'

    import ipdb; ipdb.set_trace()
    pddl_goal = query(prompt_text, system_text=system_text, use_chatgpt=True)
    
    with open(pddl_problem_filename, 'w') as f:
        f.write(context_pddl_init + pddl_goal + ')')
    return pddl_problem_filename
    

def get_helper_subgoal(expt_path, args):
    system_text = '''Main agent: agent0
Helper agent: agent1

Generate a clearly stated small independent subgoal for helper agent to help main agent complete the given task, in parallel. The subgoal should not rely on any main agent actions, and should be executable by the helper agent independently without waiting for any main agent actions. The subgoal should be clearly state with unambiguous terminology. Do not use actions like assist or help. Generate actions that the helper agent can do independently, based on the given steps for completing the task, while main agents completes the main task alongside. Do not overtake the full sequence of actions. Remember, the helper agent is only assisting the main agent.
    '''
    prompt_text = '''Example  scenario:
You have 1 shaker with 3 levels, 3 shot glasses, 3 dispensers for 3 ingredients. 
The shaker and shot glasses are clean, empty, and on the table. Your left and right hands are empty. 
The first ingredient of cocktail1 is ingredient3. The second ingredient of cocktail1 is ingredient1. 
Your goal is to make 1 cocktail. 
shot1 contains cocktail1. 

agent0 takes the following steps to complete the above task:
Grasp shot2 with right hand, fill shot2 with ingredient1 from dispenser1 using right and left hands, pour shot2 with ingredient1 into clean shaker1 using right hand, clean shot2 with right and left hands, fill shot2 with ingredient3 from dispenser3 using right and left hands, grasp shaker1 with left hand, pour shot2 with ingredient3 into used shaker1 using right hand, leave shot2 with right hand, shake cocktail1 with ingredient3 and ingredient1 in shaker1 using left and right hands, pour shaker1 with cocktail1 into shot1 using left hand.

agent1 subgoals: put ingredient1 in shaker1 
'''
    scenario_filename =  f"./domains/{args.domain}/p{args.task_id}.nl"
    with open(scenario_filename, 'r') as f:
        current_scenario = f.read()
    singleagent_plan_filename =  f"./experiments/run1/results/llm_ic_pddl/{args.domain}/p{args.task_id}.pddl"
    with open(singleagent_plan_filename, 'r') as f:
        singleagent_plan = f.read()
    current_prompt_text = '\n\nCurrent Scenario:\n'
    current_prompt_text += f'{current_scenario.strip()}\n\n'
    current_prompt_text += 'agent0 takes the following steps to complete the above task:\n'
    current_prompt_text += f'{singleagent_plan.strip()}\n\n'
    current_prompt_text += 'agent1 subgoals: '
    
    prompt_text = prompt_text + current_prompt_text
    # helper_subgoal = 'Fetch the intact tyre from the boot, inflate the intact tyre, and put on the intact tyre on the hub.'
    import ipdb; ipdb.set_trace()
    helper_subgoal = query(prompt_text, system_text=system_text, use_chatgpt=True)
    
    return helper_subgoal




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-domain-fixer")
    parser.add_argument('--domain', type=str, choices=DOMAINS, default="tyreworld")
    parser.add_argument('--time-limit', type=int, default=10)
    parser.add_argument('--task_id', type=str, default='1')
    parser.add_argument('--experiment_folder', type=str, default='experiments_multiagent_help')
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()

    singleagent_planning_time = []
    singleagent_cost = []

    multiagent_helper_planning_time = []
    multiagent_helper_cost = []
    multiagent_helper_success = []

    multiagent_main_planning_time = []
    multiagent_main_cost = []
    multiagent_main_success = []

    for domain in DOMAINS:
        args.domain = domain
        if not os.path.exists(args.experiment_folder):
            os.mkdir(args.experiment_folder)
        path = os.path.join(args.experiment_folder, f'run{args.run}')
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, args.domain)
        if not os.path.exists(path):
            os.mkdir(path)   
        args.task_id = f'0{args.task_id}' if len(args.task_id)==1 else args.task_id
        
        # normal planning
        # planning_time, cost = planner(path, args)
        # singleagent_planning_time.append(planning_time)
        # singleagent_cost.append(cost)

        helper_subgoal = get_helper_subgoal(path, args)
        # get_pddl_problem(path, args)
        get_pddl_goal(path, args, helper_subgoal=helper_subgoal)
        planning_time, cost = planner(path, args, subgoal=True)
        success = validator(path, args, subgoal=True)
        multiagent_helper_planning_time.append(planning_time)
        multiagent_helper_cost.append(cost)
        multiagent_helper_success.append(success)

        get_updated_init_conditions(path, args)
        planning_time, cost = planner(path, args, edited_init=True)
        success = validator(path, args, edited_init=True)
        multiagent_main_planning_time.append(planning_time)
        multiagent_main_cost.append(cost)
        multiagent_main_success.append(success)

        import ipdb; ipdb.set_trace()

    print(singleagent_planning_time, singleagent_cost, multiagent_helper_planning_time,
    multiagent_helper_cost, multiagent_helper_success, multiagent_main_planning_time,
    multiagent_main_cost, multiagent_main_success)
    # sr, gcr, total_planner_time, planner_steps_max =  evaluate()