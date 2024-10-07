from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()
client = OpenAI()

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
    return result_text

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
            pddl_goal = pddl_goal.replace('(empty hands)', '').replace('(empty-hand)', '').replace('(empty-hands)', '').replace('empty hand', '')
        with open(log_file, 'a+') as f:
            f.write(f"\n\n{pddl_goal}")
        with open(pddl_problem_filename, 'w') as f:
            f.write(context_pddl_init + pddl_goal + ')')

        pddl_problem_filename_arr.append(pddl_problem_filename)
    
    end = time.time()-start
    return pddl_problem_filename_arr, end