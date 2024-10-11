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
        helper_subgoal = generator.query(prompt_text, system_text=system_text, model=args.model)
        #print(helper_subgoal, "\n")
        prompt_text += helper_subgoal
        all_subgoals.append(helper_subgoal)
    end = time.time()-start

    #print(prompt_text)
    #print("cumulative subgoals", all_subgoals,"\n")

    with open(log_file, 'a+') as f: f.write(f"\n\n{current_prompt_text} {helper_subgoal}")
    helper_subgoal = helper_subgoal.split('final goal condition is:')[-1].strip()
    return all_subgoals, end


if __name__ == "__main__":
    # parse arguments, define domain, and create experiment folder
    parser = argparse.ArgumentParser(description="LLM-multiagent-helper")
    parser.add_argument('--domain', type=str, choices=utils.DOMAINS, default="tyreworld")
    parser.add_argument('--time-limit', type=int, default=200)
    parser.add_argument('--task_id', type=str)
    parser.add_argument('--experiment_folder', type=str, default='experiments_multiagent_help')
    parser.add_argument('--human_eval', type=bool, default=False)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--model', type=str, default='gpt-4o')
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
            planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner.planner(path, args)
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
            subgoal_array, t1 = get_helper_subgoal_without_plan(path, args, log_file)
            # add check for validity of all goals
            print(subgoal_array)
            # # helper_subgoal = "xyz"
            goal_files, t2 = generator.get_pddl_goal(path, args, subgoal_array, log_file)
            # print(goal_files)
            LLM_text_sg_time.append(t1)
            LLM_pddl_sg_time.append(t2)

        except Exception as e:
            print("LLM generation failed, ", e)
        
        # handle all subgoals and init conditions
        # edited init starts at  0 for original, then 1 for post-first subgoal, etc ...
        # subgoal 1 used original pddl domain, then subgoal 2 uses edited_init_1, 3 uses edited_init_2, etc ...
        for i in range(1,args.num_agents):
            # print(f"agent{i}")
            try:
                planner_total_time, planner_total_time_opt, best_cost, planner_search_time_1st_plan, first_plan_cost = planner.planner(path, args, subgoal_idx=i)
                #print("planner successful")
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