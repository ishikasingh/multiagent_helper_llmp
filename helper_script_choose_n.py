import argparse
import glob
import json
import os
import random
import sys
import time
import numpy as np
import copy
import modules.planner as planner
import modules.utils as utils
import modules.generator as generator
import re

def get_helper_subgoal_without_plan(expt_path, args, log_file, agent_text):

    system_text = '''Main agent: agent0
                    Helper agent: ''' 

    for i in range(1,args.num_agents):
        system_text += f' agent{i}, '
    
    system_text += '\n'

    system_text += ''' Your goal is to generate goals for agents such that they can be executed in parallel to decrease plan execution length. Generate only one clearly stated small independent subgoal for each helper agent to help the main agent complete the given task. The subgoal must be executable by a helper agent completely independently without waiting for any main agent actions to change predicates. The subgoal SHOULD NOT be interwoven with other generated subgoals or the main task, but rather run uninterrupted from inception time in PARALLEL with other subgoals.  
    The subgoal should be clearly stated with unambiguous terminology. Do not use actions like assist or help, only actions CLEARLY DEFINED IN THE DOMAIN. The main goal will be augmented based on the generated subgoals, but will run in parallel with them. Do not overtake the full sequence of actions. Remember, the helper agents are only assisting the main agent and act agnostically to the main agent. If there are no subgoals needed, return "none".
    '''
    
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
    '''
    
    if args.domain == 'tyreworld' or args.domain == 'grippers':
        prompt_text += '''
        A possible agent1 subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - 
        agent1 subgoals: It can independently prepare cocktail1 using shot3 and shaker1. The agent will grasp shot3, fill it with ingredients, pour to shaker, shake the cocktail, and pour it to the target glass. All actions can be done while agent0 works with other containers. Therefore, agent1's clearly stated (with object names) complete and final goal condition is: contains shot1 cocktail1.
        A possible agent2 subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - 
        agent2 subgoals: It can independently prepare cocktail2 using shot3 and shaker2. The agent will grasp shot3, fill it with ingredients, pour to shaker, shake the cocktail, and pour it to the target glass. All actions can be done while other agents work with other containers. Therefore, agent2's clearly stated (with object names) complete and final goal condition is: contains shot2 cocktail2.
        '''
    else:
        prompt_text += '''
        A possible agent1 subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - 
        agent1 subgoals: It can independently prepare cocktail1 using shot3 and shaker1. The agent will grasp shot3, fill it with ingredients, pour to shaker, shake the cocktail, and pour it to the target glass. All actions can be done while agent0 works with other containers. Therefore, agent1's clearly stated (with object names) complete and final goal condition is: contains shot1 cocktail1 and all hands are empty.
        A possible agent2 subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - 
        agent2 subgoals: It can independently prepare cocktail2 using shot3 and shaker2. The agent will grasp shot3, fill it with ingredients, pour to shaker, shake the cocktail, and pour it to the target glass. All actions can be done while other agents work with other containers. Therefore, agent2's clearly stated (with object names) complete and final goal condition is: contains shot2 cocktail2 and all hands are empty.
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
    current_prompt_text += f'Here is the number of, and REASONING for, agents: {agent_text}. THIS REASONING MAY CONTAIN SUBGOALS, BUT THESE WILL NOT BE CONSIDERED. Your task is to either restate them or generate a new subgoal based off the reasoning so we can generate PDDL plans based off these goals.\n'
    current_prompt_text += f'Return only one clearly stated subgoal condition for one and only one agent without explanation or steps. A possible subgoal looking at how the domain works based on the plan example provided for another task in this domain could be - \n'

    prompt_text = prompt_text + current_prompt_text
    # helper_subgoal = 'Fetch the intact tyre from the boot, inflate the intact tyre, and put on the intact tyre on the hub.'
    # print("\n prompt_text for helper_sg w/o plan \n",prompt_text)
    # import ipdb; ipdb.set_trace()
    #print("prompt text\n", prompt_text)
    start = time.time()
    all_subgoals = []
    valid_subgoals = 0
    
    helper_subgoal = "none"
    for i in range(1, args.num_agents):
        prompt_text += f"\n agent{i} subgoal:"
        # print(f"querying for subgoal {i}")
        helper_subgoal = generator.query(prompt_text, system_text=system_text, model=args.model)
        prompt_text += helper_subgoal
        
        if helper_subgoal == "none":
            print(f"LLM generated no subgoal for agent {i}, reducing number of agents")
            args.num_agents = i
            print(f"args.num_agents: {args.num_agents}")
            break
            
        all_subgoals.append(helper_subgoal)
        valid_subgoals += 1

    end = time.time()-start

    with open(log_file, 'a+') as f: 
        f.write(f"\n\n{current_prompt_text} {helper_subgoal}")
        f.write(f"\nReduced to {valid_subgoals + 1} agents")
        
    helper_subgoal = helper_subgoal.split('final goal condition is:')[-1].strip()
    return all_subgoals, end

def get_helper_subgoal_manual(expt_path, args, log_file):
    all_subgoals = []
    print(f"Entering manual subgoal generation for {args.domain} task {args.task_id}, with {args.num_agents-1} subgoals")
    for i in range(1, args.num_agents):
        while True:
            subgoal = input(f"Enter subgoal for agent {i}: ")
            if subgoal == "":
                print(f"Subgoal for agent {i} is empty, please enter a valid subgoal")
                continue
            break
        all_subgoals.append(subgoal)
    return all_subgoals, 0

def get_cached_results(args):
    cache_file = f"./SA_cache/{args.domain}_{args.task_id}.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            return (
                cache['planner_total_time'],
                cache['planner_total_time_opt'], 
                cache['best_cost'],
                cache['planner_search_time_1st_plan'],
                cache['first_plan_cost']
            )
    return None

def cache_results(args, results):
    cache_file = f"./SA_cache/{args.domain}_{args.task_id}.json"
    cache = {
        'planner_total_time': results[0],
        'planner_total_time_opt': results[1],
        'best_cost': results[2],
        'planner_search_time_1st_plan': results[3],
        'first_plan_cost': results[4]
    }
    with open(cache_file, 'w') as f:
        json.dump(cache, f)

def choose_n_agents(expt_path, args, log_file, max_agents):
    # system prompt
    system_text = '''
    Given a planning problem description, determine how many agents would be optimal to solve it efficiently.

    Each agent will be assigned a single clear goal that they can work towards independently. The planner will generate individual PDDL plans for each agent's goal.

    Consider that while multiple agents can work in parallel, they may interfere with and wait for each other if their goals are not truly independent. This behavior will be punished as the overall plan length will increase. We want to minimize plan length, as well as planning length. In most cases, planning length increases with more agents when the subgoals are not atomic.
    
    All agents will start at the same time. Do NOT pay attention to whether there is one robot in the domain. We are capable of scaling to multiple robots or agents, so do not let this constrain your decisions.

    Important: Include your final number recommendation in square brackets at the end, like this: [2]
    '''
    
    prompt_text = '''Example domain scenario: You have 3 blocks. b2 is on top of b3. b3 is on top of b1. b1 is on the table. b2 is clear. Your arm is empty. 
    Your goal is to move the blocks. b3 should be on top of b2. b1 should be on top of b3.  \n'''

    prompt_text += '''In this example, the optimal number of agents is 2. Anymore has no effect or increases aggregated plan length as only one agent can work on stacking the
     blocks at a time. The characteristics of blocksworlds domain mean that only one block can be stacked at a time to achieve a final goal. More agents will only wait for actions to finish, increasing the overall plan length. [2]'''
    
    scenario_filename =  f"./domains/{args.domain}/p{args.task_id}.nl"
    with open(scenario_filename, 'r') as f:
        current_scenario = f.read()
    
    prompt_text += f'''The task is: \n {current_scenario.strip()}. \n Return the optimal number of agents, greater than or equal to 1. Be sure to clearly explain your reasoning and discuss how you consider critical paths in the main goal. Remember, every agent has a SINGULAR goal. Do not propose a general role for these agents but rather a singular thing they can accomplish to reduce execution length. End your response with the number in square brackets, like [3]. \n'''

    start = time.time()

    try:
        num_agents_text = generator.query(prompt_text, system_text=system_text, model=args.model)
        match = re.search(r'\[(\d+)\]', num_agents_text)
        if match:
            num_agents = int(match.group(1))
            print(f"num_agents extracted from regex: {num_agents}")
            if num_agents > max_agents or num_agents < 1:
                raise ValueError("invalid number range")
        else:
            raise ValueError("No bracketed number found")
    except ValueError as e:
        print(f"Error: {e}")
        num_agents = max_agents

    end = time.time()-start
    return num_agents_text, num_agents, end

if __name__ == "__main__":
    # parse arguments, define domain, and create experiment folder
    parser = argparse.ArgumentParser(description="LLM-multiagent-helper")
    parser.add_argument('--domain', type=str, choices=utils.DOMAINS, default="tyreworld")
    parser.add_argument('--time-limit', type=int, default=200)
    parser.add_argument('--task_id', type=str)
    parser.add_argument('--experiment_folder', type=str, default='/run/user/1188')
    parser.add_argument('--human_eval', type=bool, default=False)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--manual', type=bool, default=False)
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

        # Determine optimal number of agents if not manually specified
        if not args.manual:
            max_agents = 5  # Set a reasonable maximum number of agents
            num_agents_text, optimal_agents, choose_time = choose_n_agents(path, args, log_file, max_agents)
            if optimal_agents == 0:
                optimal_agents = 1
            args.num_agents = optimal_agents
            print(f"args.num_agents: {args.num_agents}")
            with open(log_file, 'a+') as f:
                f.write(f"\nAgent Selection Analysis:\n{num_agents_text}\nChosen number of agents: {optimal_agents}\nSelection time: {choose_time}s\n")

        # normal planning and same for multi-agent planning
        # cache single agent results for better experiment runtime
        try:
            cached_results = get_cached_results(args)
            if cached_results:
                print("Using cached single agent results")
                planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = cached_results
            else:
                print("No cached single agent results found, running planner")
                planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner.planner(path, args)
                cache_results(args, (planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost))
            
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

        try:
            if args.manual == True:
                subgoal_array, t1 = get_helper_subgoal_manual(path, args, log_file)
            else:
                subgoal_array, t1 = get_helper_subgoal_without_plan(path, args, log_file, f"{num_agents_text}. There will be {args.num_agents} agents.")
            # add check for validity of all goals
            print(subgoal_array)
            # # helper_subgoal = "xyz"
            goal_files, t2 = generator.get_pddl_goal(path, args, subgoal_array, log_file)
            print(goal_files)
            LLM_text_sg_time.append(t1)
            LLM_pddl_sg_time.append(t2)

        except Exception as e:
            print("LLM generation failed, ", e)
            
        # If num_agents was reduced to 1, print single agent results and skip remaining processing
        if args.num_agents == 1:
            print("Number of agents reduced to 1, using single agent results")
            
            # Print results including LLM generation time overhead
            print(f"[results][{args.domain}][{args.task_id}]")
            print(f"[single_agent][planning time: {singleagent_planning_time[0]}][cost: {singleagent_cost[0]}]")
            print(f"[multi_agent][planning_time: {singleagent_planning_time[0] + t1 + t2}][cost: {singleagent_cost[0]}][agents: 1][optimization time 0]")
            
            with open(log_file, 'a+') as f:
                f.write("\nReduced to single agent, using cached results")
            continue
            
        # handle all subgoals and init conditions
        # edited init starts at  0 for original, then 1 for post-first subgoal, etc ...
        # subgoal 1 used original pddl domain, then subgoal 2 uses edited_init_1, 3 uses edited_init_2, etc ...
        for i in range(1,args.num_agents):
            print(f"agent{i}")
            try:
                planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner.planner(path, args, subgoal_idx=i)
                print("planner successful")
                success, validator_time = planner.validator(path, subgoal_idx=i)
                print("validator time", validator_time)
                multiagent_helper_planning_time.append(planner_total_time)
                multiagent_helper_planning_time_opt.append(planner_total_time_opt)
                multiagent_helper_planning_time.append(validator_time)
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
                LLM_text_sg_time.append(0)
                LLM_pddl_sg_time.append(0)
                multiagent_helper_planning_time.append(0)
                multiagent_helper_planning_time_opt.append(0)
                multiagent_helper_cost.append(0)
                multiagent_helper_planning_time_1st.append(0)
                multiagent_helper_cost_1st.append(0)
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
            planner.get_updated_init_conditions(path, validation_filename=f"./{path}/p{args.task_id}_{i}_validation.txt", pddl_problem_filename=init_problem, pddl_problem_filename_edited=init_problem_out,is_main=main_goal)

        # handle main agent
        try:
            # init conditions should be good from last iter of subgoal loop
            # add goal to main agent 
            planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner.planner(path, args, subgoal_idx=args.num_agents)
            # print("running validator_sim_recursion_function")
            dp_start = time.time()
            #print("starting dp")
            plan_length, success = planner.validator_simulation_recursive(path, log_file)
            dp_end = time.time()
            #print("dp planning time", dp_end - dp_start)
            with open(log_file, 'a+') as f: f.write(f"plan_length {plan_length})\n") # {singleagent_cost[-1]}\n")
            multiagent_main_planning_time.append(planner_total_time)
            multiagent_main_planning_time_opt.append(planner_total_time_opt)
            multiagent_main_cost.append(best_cost)
            multiagent_main_planning_time_1st.append(planner_search_time_1st_plan)
            multiagent_main_cost_1st.append(first_plan_cost)
            multiagent_main_success.append(success)
            overall_plan_length.append(plan_length)
            print(f"[results][{args.domain}][{args.task_id}]")
            print(f"[single_agent][planning time: {singleagent_planning_time[0]}][cost: {singleagent_cost[0]}]")
            print(f"[multi_agent][planning_time: {LLM_pddl_sg_time[0]+LLM_text_sg_time[0]+np.sum(multiagent_helper_planning_time)+multiagent_main_planning_time[0]}][cost: {float(overall_plan_length[0])}][agents: {args.num_agents}][optimization time {dp_end - dp_start}]")
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