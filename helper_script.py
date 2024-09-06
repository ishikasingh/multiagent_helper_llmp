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
import numpy as np

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
    openai.api_key = "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"
    while server_cnt < 10:
        try:
            if use_chatgpt: # currently, we will always use chatgpt
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    temperature=0.1,
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


def planner(expt_path, args, subgoal=False, edited_init=False, time_limit=200):
    domain_pddl_file =  f'./domains/{args.domain}/domain.pddl'

    if subgoal:
        task_pddl_file_name =  f"./{expt_path}/p{args.task_id}_subgoal.pddl"
        plan_file_name = f"./{expt_path}/p{args.task_id}_subgoal_plan.pddl"
        info = 'subgoal'
    elif edited_init:
        task_pddl_file_name =  f"./{expt_path}/p{args.task_id}_edited_init.pddl"
        plan_file_name = f"./{expt_path}/p{args.task_id}_edited_init_plan.pddl"
        info = 'edited_init'
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
    planner_search_time_1st_plan = float(output.split('Actual search time: ')[1].split('\n')[0].strip()[:-1])
    planner_total_time = float(output.split('Planner time: ')[1].split('\n')[0].strip()[:-1])
    planner_total_time_opt = float(output.split('Actual search time: ')[-1].split('\n')[0].strip()[:-1])
    first_plan_cost = int(output.split('Plan cost: ')[1].split('\n')[0].strip())
    # import ipdb; ipdb.set_trace()
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
    
    best_cost = 10e6
    for fn in glob.glob(plan_path):
        with open(fn, "r") as f:
            plans = f.readlines()
            cost = get_cost(plans[-1])
            if cost < best_cost:
                best_cost = cost
                plan_file = fn
    print(plan_file)
    result = subprocess.run(["./downward/validate", "-v", domain_pddl_file, task_pddl_file, plan_file], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    output_file.write(output)
    if "Plan valid" in result.stdout.decode('utf-8'):
        return True
    else:
        return False

def get_updated_init_conditions(expt_path, args, validation_filename=None, pddl_problem_filename=None, pddl_problem_filename_edited=None, env_conds_only=True):
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

def get_pddl_goal(expt_path, args, helper_subgoal, log_file):
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

    pddl_problem_filename = f"./{expt_path}/p{args.task_id}_subgoal.pddl"
    
    
    system_text = 'I want you to solve planning problems. Provide me with the PDDL goal that describes the new planning goal directly without further explanations. Make sure to provide only non-conflicting, necessary, and final goal conditions mentioned in the given goal.'
    prompt_text = f"The PDDL problem and its initial conditions are given as: \n{context_pddl_init.strip()} \n\n" + \
                f"An example planning goal for this problem:  \n{context_nl.strip()} \n\n\n" + \
                f"The PDDL goal for the example planning goal:  \n{context_pddl_goal.strip()} \n\n\n" + \
                f"New planning goal for the same problem:\n Your goal is: {helper_subgoal.strip()} \n\n" + \
                f'The PDDL goal for the new planning goal:\n'

    # import ipdb; ipdb.set_trace()
    start=time.time()
    pddl_goal = query(prompt_text, system_text=system_text, use_chatgpt=True)
    end = time.time()-start
    print(prompt_text)
    import ipdb; ipdb.set_trace()
    # remove undefined goal conditions using domain predicate list
    if args.domain == 'tyreworld':
        pddl_goal = pddl_goal.replace('(empty hands)', '').replace('(empty-hand)', '').replace('(empty-hands)', '')
    with open(log_file, 'a+') as f:
        f.write(f"\n\n{pddl_goal}")
    with open(pddl_problem_filename, 'w') as f:
        f.write(context_pddl_init + pddl_goal + ')')
    return pddl_problem_filename, end
    
def get_pddl_goal_expert(expt_path, args, log_file):
    # # taken from (LLM+P), create the problem PDDL given the context
    # context_nl_filename = f"./domains/{args.domain}/p{args.task_id}.nl"
    # with open(context_nl_filename, 'r') as f:
    #     context_nl = f.read()
    # context_nl = 'Your goal is to' + context_nl.split('Your goal is to')[-1]
    context_pddl_filename = f"./domains/{args.domain}/p{args.task_id}.pddl"
    with open(context_pddl_filename, 'r') as f:
        context_pddl = f.read()
    context_pddl_init, context_pddl_goal = context_pddl.split('(:goal')
    # context_pddl_goal = f'(:goal\n{context_pddl_goal.strip()[:-1]}'

    pddl_problem_filename = f"./{expt_path}/p{args.task_id}_subgoal.pddl"

    expert_pddl_goals_filename = f"./experiments_multiagent_help/Human Topline - Bill.txt"
    with open(expert_pddl_goals_filename, 'r') as f:
        expert_pddl_goals = f.read()

    expert_pddl_goals = ''.join(expert_pddl_goals.split(args.domain)[1:])
    task_idx = human_eval_task_ids.index(int(args.task_id)) + 1

    expert_pddl_goal = expert_pddl_goals.split("####### Please provide answer here ######")[task_idx]
    pddl_goal = '(:goal' + expert_pddl_goal.split('#')[0].strip() + ')'
    print('expert_pddl_goal task idx ', task_idx)
    
    # system_text = 'I want you to solve planning problems. Provide me with the PDDL goal that describes the new planning goal directly without further explanations. Make sure to provide only non-conflicting, necessary, and final goal conditions mentioned in the given goal.'
    # prompt_text = f"The PDDL problem and its initial conditions are given as: \n{context_pddl_init.strip()} \n\n" + \
    #             f"An example planning goal for this problem:  \n{context_nl.strip()} \n\n\n" + \
    #             f"The PDDL goal for the example planning goal:  \n{context_pddl_goal.strip()} \n\n\n" + \
    #             f"New planning goal for the same problem:\n Your goal is: {helper_subgoal.strip()} \n\n" + \
    #             f'The PDDL goal for the new planning goal:\n'
    


    # start=time.time()
    # pddl_goal = query(prompt_text, system_text=system_text, use_chatgpt=True)
    # end = time.time()-start
    with open(log_file, 'a+') as f:
        f.write(f"\n\n{pddl_goal}")
    with open(pddl_problem_filename, 'w') as f:
        f.write(context_pddl_init + pddl_goal + ')')
    return pddl_problem_filename, 0

def get_helper_subgoal(expt_path, args, log_file):
#     system_text = '''Main agent: agent0
# Helper agent: agent1

# Generate one clearly stated small independent subgoal for helper agent to help main agent complete the given task. The subgoal should not rely on any main agent actions, and should be executable by the helper agent independently without waiting for any main agent actions. The subgoal should be clearly state with unambiguous terminology. Do not use actions like assist or help. Generate actions that the helper agent can do independently, based on the given steps for completing the task. The main agent should be able to continue working on the remaining task while the helper agent is completing its small subgoal. Do not overtake the full sequence of actions. Remember, the helper agent is only assisting the main agent and acts agnostically to the main agent.
#     '''
# agent1 subgoals: It can help in filling ingredient1 in shot2, then pour it in shaker1, while agent0 prepares other cocktail ingredients using other objects. In this way, agent1 would not need to wait for agent0 and it can complete its goal independently. Therefore, agent1's clearly stated (with object names) complete and final goal condition is: shaker1 contains ingredient1

# Generate a clearly stated independent subgoal for helper agent to help main agent complete roughly half of the given task. The subgoal should not rely on any main agent actions, and should be executable by the helper agent independently without waiting for any main agent actions. The subgoal should be clearly state with unambiguous terminology. Do not use actions like assist or help. Generate actions that the helper agent can do independently, based on the given steps for completing the task. The main agent should be able to continue working on the remaining task while the helper agent is completing its subgoal. Help main agent with half of the task but do not overtake the full sequence of actions. Remember, the helper agent is only assisting the main agent and acts agnostically to the main agent.
# agent1 subgoals: It can help in filling ingredient1 and ingredient3 in shaker1, and agent0 can then perform the remaining steps for making the cocktail. In this way, agent1 would not need to wait for agent0 and it can complete its goal independently, while completing roughly half of the task. Therefore, agent1's clearly stated (with object names) complete and final goal condition is: shaker1 contains ingredient1 and ingredient3

# Grasp shot2 with right hand, fill shot2 with ingredient1 from dispenser1 using right and left hands, pour shot2 with ingredient1 into clean shaker1 using right hand, clean shot2 with right and left hands, fill shot2 with ingredient3 from dispenser3 using right and left hands, grasp shaker1 with left hand, pour shot2 with ingredient3 into used shaker1 using right hand, leave shot2 with right hand, shake cocktail1 with ingredient3 and ingredient1 in shaker1 using left and right hands, pour shaker1 with cocktail1 into shot1 using left hand.
#  agent1 can use shot3 for filling ingredient1, so agent2 can continue using shot2 for other ingredients, and shot1 remains clean which needs to contain the final cocktail, so we save an additional step there.



    system_text = '''Main agent: agent0
Helper agent: agent1

Generate one clearly stated small independent subgoal for helper agent to help main agent complete the given task. The subgoal should not rely on any main agent actions, and should be executable by the helper agent independently without waiting for any main agent actions. The subgoal should be clearly state with unambiguous terminology. Do not use actions like assist or help. Generate actions that the helper agent can do independently, based on the given steps for completing the task. The main agent should be able to continue working on the remaining task while the helper agent is completing its small subgoal. Do not overtake the full sequence of actions. Remember, the helper agent is only assisting the main agent and acts agnostically to the main agent.
'''
    prompt_text = '''Example  scenario:
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

agent1 subgoals: It can help in filling ingredient1 in a shot glass, then pour it in shaker1, while agent0 prepares other cocktail ingredients using other objects. In this way, agent1 would not need to wait for agent0 and it can complete its goal independently. agent1 should also release all objects that the main agent might need for its own actions. Therefore, agent1's clearly stated (with object names) complete and final goal condition is: shaker1 contains ingredient1 and all hands are empty.
'''

    scenario_filename =  f"./domains/{args.domain}/p{args.task_id}.nl"
    with open(scenario_filename, 'r') as f:
        current_scenario = f.read()

    # singleagent_plan_filename =  f"./experiments/run1/results/llm_ic_pddl/{args.domain}/p{args.task_id}.pddl"
    # expt_path_singleagent_plan = expt_path.split('run')[0] + 'run1' + expt_path.split('run')[1][2:]
    expt_path_singleagent_plan = expt_path
    plan_path = os.path.join(f"./{expt_path_singleagent_plan}", f"p{args.task_id}_plan.pddl" + '.*')
    # import ipdb; ipdb.set_trace()
    best_cost = 10e6
    for fn in glob.glob(plan_path):
        with open(fn, "r") as f:
            plans = f.readlines()
            cost = get_cost(plans[-1])
            if cost < best_cost:
                best_cost = cost
                singleagent_plan_filename = fn
    # singleagent_plan_filename = [plan for plan in plan_files if not plan.endswith('sas') and not plan.endswith('out')][-1]

    with open(singleagent_plan_filename, 'r') as f:
        singleagent_plan = f.read()
    singleagent_plan = singleagent_plan.split(';')[0]
    current_prompt_text = '\n\nCurrent Scenario:\n'
    current_prompt_text += f'{current_scenario.strip()}\n\n'
    current_prompt_text += 'agent0 takes the following steps to complete the above task:\n'
    current_prompt_text += f'{singleagent_plan.strip()}\n\n'
    current_prompt_text += 'agent1 subgoals: '
    
    prompt_text = prompt_text + current_prompt_text
    # helper_subgoal = 'Fetch the intact tyre from the boot, inflate the intact tyre, and put on the intact tyre on the hub.'
    import ipdb; ipdb.set_trace()
    start = time.time()
    helper_subgoal = query(prompt_text, system_text=system_text, use_chatgpt=True)
    end = time.time()-start
    with open(log_file, 'a+') as f: f.write(f"\n\n{current_prompt_text} {helper_subgoal}")
    helper_subgoal = helper_subgoal.split('final goal condition is:')[-1].strip()
    return helper_subgoal, end


def get_helper_subgoal_without_plan(expt_path, args, log_file):

    system_text = '''Main agent: agent0
Helper agent: agent1

Generate one clearly stated small independent subgoal for helper agent to help main agent complete the given task. The subgoal should not rely on any main agent actions, and should be executable by the helper agent independently without waiting for any main agent actions. The subgoal should be clearly state with unambiguous terminology. Do not use actions like assist or help. Generate actions that the helper agent can do independently, based on the given steps for completing the task. The main agent should be able to continue working on the remaining task while the helper agent is completing its small subgoal. Do not overtake the full sequence of actions. Remember, the helper agent is only assisting the main agent and acts agnostically to the main agent.
'''

    prompt_text = '''Example  domain scenario:
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

Now we have a new problem defined in this domain for which we don't have access to the signle agent plan:
You have 1 shaker with 3 levels, 3 shot glasses, 3 dispensers for 3 ingredients. 
The shaker and shot glasses are clean, empty, and on the table. Your left and right hands are empty. 
The first ingredient of cocktail1 is ingredient3. The second ingredient of cocktail1 is ingredient1. 
The first ingredient of cocktail2 is ingredient1. The second ingredient of cocktail2 is ingredient2. 
Your goal is to make 2 cocktails. 
shot1 contains cocktail1. shot2 contains cocktail2. 

A possible agent1 subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - 
agent1 subgoals: It can help in filling ingredient1 in a shot glass, then pour it in shaker1, while agent0 prepares other cocktail ingredients using other objects. In this way, agent1 would not need to wait for agent0 and it can complete its goal independently. agent1 should also release all objects that the main agent might need for its own actions. Therefore, agent1's clearly stated (with object names) complete and final goal condition is: shaker1 contains ingredient1 and all hands are empty.
'''

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
        current_prompt_text += 'Now we have a new problem defined in this domain for which we don\'t have access to the signle agent plan:\n'
    else:
        current_prompt_text = '\n\nNow we have another new problem defined in this domain for which we don\'t have access to the signle agent plan:\n'
    current_prompt_text += f'{current_scenario.strip()}\n\n'
    current_prompt_text += 'A possible agent1 subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - \nagent1 subgoals: '
    
    prompt_text = prompt_text + current_prompt_text
    # helper_subgoal = 'Fetch the intact tyre from the boot, inflate the intact tyre, and put on the intact tyre on the hub.'
    print(current_prompt_text)
    import ipdb; ipdb.set_trace()
    start = time.time()
    helper_subgoal = query(prompt_text, system_text=system_text, use_chatgpt=True)
    end = time.time()-start

    with open(log_file, 'a+') as f: f.write(f"\n\n{current_prompt_text} {helper_subgoal}")
    helper_subgoal = helper_subgoal.split('final goal condition is:')[-1].strip()
    return helper_subgoal, end



def validator_simulation_recurssive(expt_path, args, log_file, multi=False, half_split=False):

    # get relevant files for validation
    domain_pddl_file =  f'./domains/{args.domain}/domain.pddl'
    task_pddl_file =  f'./domains/{args.domain}/p{args.task_id}.pddl' # since we need actual init conds
    with open(task_pddl_file, 'r') as f: task = f.read()
    
    if multi:
        plan_main = []
        plan_helper = []
        # task_pddl_file_helper =  f"./{expt_path}/p{args.task_id}_subgoal.pddl"
        # with open(task_pddl_file_helper, 'r') as f: task_helper = f.read()
        plan_path = os.path.join(f"./{expt_path}", f"p{args.task_id}_plan.pddl" + '.*')
        best_cost = 10e6
        for fn in glob.glob(plan_path):
            with open(fn, "r") as f:
                plans = f.readlines()
                cost = get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    plan_file = fn
        with open(plan_file, 'r') as f: plan_all = f.readlines()[:-1]
        for i in plan_all:
            if 'robot1' in i:
                plan_main.append(i)
            else:
                plan_helper.append(i)
    
    elif half_split:
        # task_pddl_file_helper =  f"./{expt_path}/p{args.task_id}_subgoal.pddl"
        # with open(task_pddl_file_helper, 'r') as f: task_helper = f.read()
        plan_path = os.path.join(f"./{expt_path}", f"p{args.task_id}_plan.pddl" + '.*')
        best_cost = 10e6
        for fn in glob.glob(plan_path):
            with open(fn, "r") as f:
                plans = f.readlines()
                cost = get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    plan_file = fn
        with open(plan_file, 'r') as f: plan_all = f.readlines()[:-1]
        half_len = int(len(plan_all)/2)
        plan_helper = plan_all[:half_len]
        plan_main = plan_all[half_len:]

        
    else:
        # task_pddl_file_helper =  f"./{expt_path}/p{args.task_id}_subgoal.pddl"
        # with open(task_pddl_file_helper, 'r') as f: task_helper = f.read()
        plan_path = os.path.join(f"./{expt_path}", f"p{args.task_id}_subgoal_plan.pddl" + '.*')
        best_cost = 10e6
        for fn in glob.glob(plan_path):
            with open(fn, "r") as f:
                plans = f.readlines()
                cost = get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    plan_file = fn

        with open(plan_file, 'r') as f: plan_helper = f.readlines()[:-1]
        if '.pddl.sas' in plan_file or '.pddl.out' in plan_file:
            return 1e6, False

        # task_pddl_file_main =  f"./{expt_path}/p{args.task_id}.pddl" # since we need actual init conds
        # with open(task_pddl_file_main, 'r') as f: task_main = f.read()
        plan_path = os.path.join(f"./{expt_path}", f"p{args.task_id}_edited_init_plan.pddl" + '.*')
        best_cost = 10e6
        for fn in glob.glob(plan_path):
            with open(fn, "r") as f:
                plans = f.readlines()
                cost = get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    plan_file = fn
        with open(plan_file, 'r') as f: plan_main = f.readlines()[:-1]
        if '.pddl.sas' in plan_file or '.pddl.out' in plan_file:
            return 1e6, False




    # make temp val files
    # task_path = f"./{expt_path}/p{args.task_id}_task_temp.txt"
    # with open(task_path, 'w') as f: f.write(task)    
    val_path_helper = f"./{expt_path}/helper_val_temp.txt"
    val_path_main = f"./{expt_path}/main_val_temp.txt"
    plan_path_helper = f"./{expt_path}/helper_plan_temp.txt"
    plan_path_main = f"./{expt_path}/main_plan_temp.txt"

    task_path_helper = f"./{expt_path}/helper_task_temp.txt"
    task_path_main = f"./{expt_path}/main_task_temp.txt"

    # os.remove(val_path_helper); os.remove(val_path_main)
    # os.remove(plan_path_helper); os.remove(plan_path_main)
    # os.remove(task_path_helper); os.remove(task_path_main)

    with open(task_path_helper, 'w') as f: f.write(task)
    with open(task_path_main, 'w') as f: f.write(task)

    with open(val_path_helper, 'w') as f: f.write('')
    with open(val_path_main, 'w') as f: f.write('')
    with open(plan_path_helper, 'w') as f: f.write('')
    with open(plan_path_main, 'w') as f: f.write('')

    i=0; j=0
    task_helper = task
    task_main = task

    print(f"TASK: {args.domain} - {args.run} - ", args.task_id)
    with open(log_file, 'a+') as f: f.write(f"TASK: {args.domain} - {args.run} - {args.task_id}\n")

    print(len(plan_helper), len(plan_main), len(plan_helper) + len(plan_main))
    
    global execution_state
    execution_state = np.zeros((len(plan_helper)+1, len(plan_main)+1, 3)) + 1e6

    if i == len(plan_helper):
        plans = [len(plan_main)]
    elif j == len(plan_main):
        plans = [len(plan_helper)]
    else:
        plans = [validator_sim_recurssion_function(expt_path, domain_pddl_file, i, j, plan_helper, plan_main, task_helper, task_main, agent='helper'),
                validator_sim_recurssion_function(expt_path, domain_pddl_file, i, j, plan_helper, plan_main, task_helper, task_main, agent='main'),
                validator_sim_recurssion_function(expt_path, domain_pddl_file, i, j, plan_helper, plan_main, task_helper, task_main, agent='both')]

    # plans = validator_sim_recurssion_function(expt_path, domain_pddl_file, i, j, plan_helper, plan_main, task_helper, task_main)
    print(plans)

    # if plan_length[0] == len(plan_helper):
    #     plan_length = plan_length[1:]
    # elif plan_length[1] == len(plan_main):
    #     plan_length = plan_length[:1] + plan_length[2:]
    
    plan_length = min(plans)
    success=False if plan_length >= 1e10 else True
    # import ipdb; ipdb.set_trace()
        
    # if "Plan valid" in output.stdout.decode('utf-8'):
    #     success =  True
    # else:
    #     # import ipdb; ipdb.set_trace()
    #     output = output.stdout.decode('utf-8')
    #     with open(log_file, 'a+') as f: f.write(f'\nparallel exec goal not reached: {plan_helper[i-1]}, {plan_main[j-1]} - {output}\n')
    #     success =  False
    # import ipdb; ipdb.set_trace()
    print(plan_length, success)
    return plan_length, success



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-multiagent-helper")
    parser.add_argument('--domain', type=str, choices=DOMAINS, default="tyreworld")
    parser.add_argument('--time-limit', type=int, default=200)
    parser.add_argument('--task_id', type=str)
    parser.add_argument('--experiment_folder', type=str, default='experiments_multiagent_help')
    parser.add_argument('--human_eval', type=bool, default=False)
    parser.add_argument('--run', type=int, default=1)
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
    print(args.domain)
    path = os.path.join(base_path, args.domain)
    if not os.path.exists(path):
        os.mkdir(path) 
    log_file = os.path.join(base_path, f'helper_logs_{args.domain}_exec_lens.log')
    with open(log_file, 'w') as f: f.write(f"start_eval\n")
    
    human_eval_task_ids = [1, 6, 11, 16]

    task_list = [args.task_id] if args.task_id != None else range(1, 21)

    for task_id in task_list: # 1, 21
        args.task_id = str(task_id)

        if args.human_eval: #human eval
            args.task_id = str(human_eval_task_ids[task_id])

        args.task_id = f'0{args.task_id}' if len(args.task_id)==1 else args.task_id
        
        # normal planning and same for multi-agent planning
        try:
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
        # plan_length, success = validator_simulation_recurssive(path, args, log_file, multi=True)
        # with open(log_file, 'a+') as f: f.write(f"plan_length {plan_length}\n") # {singleagent_cost[-1]}\n")
        # overall_plan_length.append(plan_length)
        # multiagent_main_success.append(success)
        
        # helper_subgoal, t1 = get_helper_subgoal_without_plan(path, args, log_file)
        # _, t2 = get_pddl_goal(path, args, helper_subgoal, log_file)
        try:
            helper_subgoal, t1 = get_helper_subgoal_without_plan(path, args, log_file)
            # # helper_subgoal = "xyz"
            _, t2 = get_pddl_goal(path, args, helper_subgoal, log_file)
            planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner(path, args, subgoal=True)
            success = validator(path, args, subgoal=True)

            LLM_text_sg_time.append(t1)
            LLM_pddl_sg_time.append(t2)
            multiagent_helper_planning_time.append(planner_total_time)
            multiagent_helper_planning_time_opt.append(planner_total_time_opt)
            multiagent_helper_cost.append(best_cost)
            multiagent_helper_planning_time_1st.append(planner_search_time_1st_plan)
            multiagent_helper_cost_1st.append(first_plan_cost)
            multiagent_helper_success.append(success)
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

        # _, t2 = get_pddl_goal_expert(path, args, log_file)
        # try:
        #     # helper_subgoal, t1 = get_helper_subgoal_without_plan(path, args, log_file)
        #     # # helper_subgoal = "xyz"
        #     _, _ = get_pddl_goal_expert(path, args, log_file)
        #     planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner(path, args, subgoal=True)
        #     success = validator(path, args, subgoal=True)

        #     LLM_text_sg_time.append(-1)
        #     LLM_pddl_sg_time.append(-1)
        #     multiagent_helper_planning_time.append(planner_total_time)
        #     multiagent_helper_planning_time_opt.append(planner_total_time_opt)
        #     multiagent_helper_cost.append(best_cost)
        #     multiagent_helper_planning_time_1st.append(planner_search_time_1st_plan)
        #     multiagent_helper_cost_1st.append(first_plan_cost)
        #     multiagent_helper_success.append(success)
        # except Exception as e:
        #     LLM_text_sg_time.append(-1)
        #     LLM_pddl_sg_time.append(-1)
        #     multiagent_helper_planning_time.append(-1)
        #     multiagent_helper_planning_time_opt.append(-1)
        #     multiagent_helper_cost.append(-1)
        #     multiagent_helper_planning_time_1st.append(-1)
        #     multiagent_helper_cost_1st.append(-1)
        #     multiagent_helper_success.append(0)
        #     print(e)
        #     with open(log_file, 'a+') as f:
        #         f.write(f"\n\nError: {e}")


        try:
            get_updated_init_conditions(path, args)
            planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner(path, args, edited_init=True)
            plan_length, success = validator_simulation_recurssive(path, args, log_file)
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