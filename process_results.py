
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length):
    x = np.array([x for x in singleagent_planning_time if x!=-1]).mean(), \
    np.array([x for x in singleagent_planning_time_opt if x!=-1]).mean(), \
    np.array([x for x in singleagent_planning_time_1st if x!=-1]).mean(), \
    np.array([x for x in singleagent_cost if x!=-1]).mean(), \
    np.array([x for x in singleagent_cost_1st if x!=-1]).mean(), \
    np.array([x for x in multiagent_main_success if x!=-1]).sum(), \
    np.array([x for x in overall_plan_length if x!=-1]).mean()
    singleagent.append(x)

    singleagent_planning_time = np.array([x for x in singleagent_planning_time])
    singleagent_cost = np.array([x for x in singleagent_cost])
    overall_plan_length = np.array([x for x in overall_plan_length])
    return singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length

def get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length):
    x = np.array([x for x in singleagent_planning_time if x!=-1]).mean(), \
    np.array([x for x in singleagent_planning_time_opt if x!=-1]).mean(), \
    np.array([x for x in singleagent_planning_time_1st if x!=-1]).mean(), \
    np.array([x for x in singleagent_cost if x!=-1]).mean(), \
    np.array([x for x in singleagent_cost_1st if x!=-1]).mean(), \
    np.array([x for x in singleagent_planning_time if x==-1]).sum()
    singleagent.append(x)

    multiagent_main_planning_time = [multiagent_main_planning_time[x] if multiagent_helper_planning_time[x]!=-1 else singleagent_planning_time[x] for x in range(len(multiagent_main_planning_time))]
    multiagent_main_cost = [multiagent_main_cost[x] if multiagent_helper_cost[x]!=-1 else singleagent_cost[x] for x in range(len(multiagent_main_planning_time))]
    multiagent_main_planning_time_1st = [multiagent_main_planning_time_1st[x] if multiagent_helper_planning_time_1st[x]!=-1 else singleagent_planning_time_1st[x] for x in range(len(multiagent_main_planning_time))]
    multiagent_main_cost_1st = [multiagent_main_cost_1st[x] if multiagent_helper_cost_1st[x]!=-1 else singleagent_cost_1st[x] for x in range(len(multiagent_main_planning_time))]
    multiagent_main_success = [multiagent_main_success[x] if multiagent_helper_success[x]!=0 else True for x in range(len(multiagent_main_planning_time))]
    multiagent_main_planning_time_opt = [multiagent_main_planning_time_opt[x] if multiagent_helper_planning_time_opt[x]!=-1 else singleagent_planning_time_opt[x] for x in range(len(multiagent_main_planning_time))]
    overall_plan_length = [overall_plan_length[x] if multiagent_helper_cost[x]!=-1 else singleagent_cost[x] for x in range(len(multiagent_main_planning_time))]


    # multiagent_main_planning_time = [multiagent_main_planning_time[x] if multiagent_main_success[x]!=0 else singleagent_planning_time[x] for x in range(len(multiagent_main_planning_time))]
    # multiagent_main_cost = [multiagent_main_cost[x] if multiagent_main_success[x]!=0 else singleagent_cost[x] for x in range(len(multiagent_main_planning_time))]
    # multiagent_main_planning_time_1st = [multiagent_main_planning_time_1st[x] if multiagent_main_success[x]!=0 else singleagent_planning_time_1st[x] for x in range(len(multiagent_main_planning_time))]
    # multiagent_main_cost_1st = [multiagent_main_cost_1st[x] if multiagent_main_success[x]!=0 else singleagent_cost_1st[x] for x in range(len(multiagent_main_planning_time))]
    # multiagent_main_success = [multiagent_main_success[x] if multiagent_main_success[x]!=0 else True for x in range(len(multiagent_main_planning_time))]
    # multiagent_main_planning_time_opt = [multiagent_main_planning_time_opt[x] if multiagent_main_success[x]!=0 else singleagent_planning_time_opt[x] for x in range(len(multiagent_main_planning_time))]
    # overall_plan_length = [overall_plan_length[x] if multiagent_main_success[x]!=0 else singleagent_cost[x] for x in range(len(multiagent_main_planning_time))]

    
    x = np.array([x for x in multiagent_main_planning_time if x!=-1]).mean(), \
    np.array([x for x in multiagent_main_planning_time_opt if x!=-1]).mean(), \
    np.array([x for x in multiagent_main_planning_time_1st if x!=-1]).mean(), \
    np.array([x for x in multiagent_main_cost if x!=-1]).mean(), \
    np.array([x for x in multiagent_main_cost_1st if x!=-1]).mean(), \
    sum(multiagent_main_success)/20, \
    np.array([x for x in overall_plan_length if x!=-1]).mean()
    main.append(x)

    x = np.array([x for x in multiagent_helper_planning_time if x!=-1]).mean(), \
    np.array([x for x in multiagent_helper_planning_time_opt if x!=-1]).mean(), \
    np.array([x for x in multiagent_helper_planning_time_1st if x!=-1]).mean(), \
    np.array([x for x in multiagent_helper_cost if x!=-1]).mean(), \
    np.array([x for x in multiagent_helper_cost_1st if x!=-1]).mean(), \
    sum(multiagent_helper_success)/20, \
    np.array([x for x in LLM_text_sg_time if x!=-1]).mean(), \
    np.array([x for x in LLM_pddl_sg_time if x!=-1]).mean()
    helper.append(x)


    singleagent_planning_time = np.array([x for x in singleagent_planning_time])
    singleagent_cost = np.array([x for x in singleagent_cost])
    multiagent_total_planning_time = np.array([x for x in multiagent_helper_planning_time]) + \
                                        np.array([x for x in LLM_text_sg_time]) + \
                                        np.array([x for x in LLM_pddl_sg_time]) + \
                                        np.array([x for x in multiagent_main_planning_time])
    overall_plan_length = np.array([x for x in overall_plan_length])

    return singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length

# if __name__ == "__main__":
np.set_printoptions(suppress=True)

print('LLM-PDDL planning results without single agent')
print('barman')
singleagent = []
helper = []
main = []
    # barman
singleagent_planning_time = [99.46, 47.96, 98.99, 47.66, 200.17, 200.22, 200.19, 200.2, 200.25, 200.25, 200.24, 200.27, 200.29, 200.27, 200.27, 200.29, 200.31, 200.3, 200.3, 200.31]
singleagent_planning_time_opt = [99.2061, 47.7393, 98.7744, 47.4436, 13.0095, 13.0857, 120.338, 123.974, 4.82308, 2.62755, 9.71439, 2.54757, 20.2563, 27.6744, 48.8685, 25.4958, 0.223953, 88.3498, 105.63, 81.3814]
singleagent_cost = [36.0, 36.0, 36.0, 36.0, 49.0, 49.0, 49.0, 49.0, 65.0, 65.0, 65.0, 65.0, 82.0, 84.0, 83.0, 82.0, 126.0, 92.0, 95.0, 94.0]
singleagent_planning_time_1st = [0.0199729, 0.00797656, 0.0199915, 0.00801197, 0.0319723, 0.0439534, 0.0239877, 0.0479725, 0.0520516, 0.071968, 0.0519469, 0.0759757, 0.155942, 0.0799466, 0.119965, 0.0839681, 0.223953, 0.147943, 0.159958, 0.127974]
singleagent_cost_1st = [48, 50, 48, 50, 67, 65, 56, 74, 74, 81, 82, 81, 107, 113, 106, 115, 126, 117, 134, 145]
LLM_text_sg_time = [3.3933610916137695, 4.246657848358154, 4.673633098602295, 3.4167892932891846, 3.7508175373077393, 4.709804058074951, 6.05735182762146, 6.32104229927063, 6.608708620071411, 3.213385581970215, 4.0370118618011475, 3.9542641639709473, 4.3914735317230225, 8.222650289535522, 4.709709644317627, 3.425982713699341, 3.1448607444763184, 5.152213096618652, 4.051896095275879, 3.219715118408203]
LLM_pddl_sg_time = [1.3906517028808594, 1.567596673965454, 2.2745981216430664, 1.5670413970947266, 2.667090892791748, 1.3216767311096191, 1.1732871532440186, 1.6111226081848145, 1.3905322551727295, 1.1704494953155518, 2.102853775024414, 1.939497947692871, 1.7430377006530762, 2.3010010719299316, 0.975214958190918, 1.1627986431121826, 3.035249710083008, 1.3436470031738281, 1.295471429824829, 1.1362133026123047]
multiagent_helper_planning_time = [0.22, 0.23, 0.23, 0.22, 0.25, 0.25, 0.24, 0.24, 0.28, 0.28, 0.27, 0.28, 0.36, 0.3, 0.34, 0.34, 0.41, 3.11, 0.41, 0.4]
multiagent_helper_planning_time_opt = [0.0159997, 0.015979, 0.0159483, 0.0119499, 0.0239675, 0.023966, 0.0239581, 0.0239817, 0.0356338, 0.0359719, 0.03598, 0.0359775, 0.0719642, 0.0719589, 0.0519745, 0.0719534, 0.0999579, 2.79537, 0.0719672, 0.0999611]
multiagent_helper_cost = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0, 4.0, 4.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0040039, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0039906, 0.0, 0.0]
multiagent_helper_cost_1st = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 4, 4]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [43.29, 45.6, 43.4, 44.29, 920.93, 948.6, 838.93, 1000.08, 1000.09, 999.11, 1000.2, 1000.31, 1000.29, 1000.2, 1000.0, 1000.04, 1000.19, 1000.22, 1000.14, 1000.23]
multiagent_main_planning_time_opt = [43.0983, 45.4309, 43.1676, 44.1264, 920.63, 948.303, 838.646, 455.669, 210.85, 202.925, 368.628, 192.386, 7.20695, 17.6445, 23.7046, 10.5407, 215.872, 46.1342, 152.621, 105.048]
multiagent_main_cost = [34.0, 34.0, 34.0, 34.0, 46.0, 46.0, 47.0, 46.0, 60.0, 61.0, 61.0, 60.0, 79.0, 77.0, 82.0, 78.0, 95.0, 84.0, 92.0, 90.0]
multiagent_main_planning_time_1st = [0.0119908, 0.00799384, 0.0119629, 0.00798929, 0.0199745, 0.0319706, 0.0199729, 0.0319628, 0.0439749, 0.0399682, 0.0519725, 0.0439791, 0.075969, 0.0519529, 0.355881, 0.155939, 4.87472, 0.151955, 0.0999593, 0.107969]
multiagent_main_cost_1st = [48, 40, 48, 40, 56, 56, 52, 67, 70, 74, 82, 74, 106, 125, 141, 124, 193, 111, 120, 119]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [38.0, 35.0, 38.0, 35.0, 47.0, 47.0, 48.0, 47.0, 64.0, 65.0, 62.0, 61.0, 80.0, 78.0, 83.0, 79.0, 96.0, 89.0, 93.0, 91.0]


singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [6.92679762840271, 3.9411065578460693, 5.40399169921875, 6.41474461555481, 7.93272066116333, 4.893869876861572, 5.240836143493652, 3.588689088821411, 5.282969951629639, 4.342296600341797, 5.99239444732666, 3.7671010494232178, 7.263261079788208, 3.899542808532715, 6.3865227699279785, 5.171237945556641, 3.293644428253174, 5.980103015899658, 4.367676496505737, 3.4466888904571533]
LLM_pddl_sg_time = [1.4858407974243164, 1.3775930404663086, 1.4078407287597656, 1.4163062572479248, 2.010366916656494, 1.6463449001312256, 1.0031704902648926, 1.3446645736694336, 1.0530662536621094, 1.3229830265045166, 1.3614799976348877, 1.135498046875, 1.6371512413024902, 1.516063928604126, 1.097221851348877, 1.1861097812652588, 1.1538655757904053, 1.5339815616607666, 1.1625051498413086, 1.4035401344299316]
multiagent_helper_planning_time = [0.21, 0.21, 0.21, 0.21, 0.24, 0.24, 0.24, 0.21, 0.27, 0.26, 0.27, 0.27, 0.35, 0.35, 0.36, 0.33, 0.47, 0.45, 0.4, 0.44]
multiagent_helper_planning_time_opt = [0.0119849, 0.0119938, 0.0159877, 0.0119702, 0.0239809, 0.0239766, 0.023955, 0.0199822, 0.0359665, 0.0239835, 0.0359672, 0.0359798, 0.0719614, 0.0719495, 0.0959238, 0.0719526, 0.0999499, 0.133317, 0.111944, 0.0759584]
multiagent_helper_cost = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400359, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
multiagent_helper_cost_1st = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [44.06, 46.78, 44.1, 46.09, 933.75, 951.63, 887.52, 1000.05, 999.89, 1000.15, 1000.22, 1000.14, 1000.26, 1000.26, 999.9, 1000.01, 1000.03, 1000.21, 1000.24, 1000.42]
multiagent_main_planning_time_opt = [43.8681, 46.5601, 43.8991, 45.9075, 933.432, 951.342, 887.236, 500.017, 208.912, 213.161, 360.287, 188.124, 7.4144, 17.5003, 32.4905, 12.3245, 210.682, 51.2608, 73.6314, 109.855]
multiagent_main_cost = [34.0, 34.0, 34.0, 34.0, 46.0, 46.0, 47.0, 46.0, 60.0, 61.0, 61.0, 60.0, 79.0, 77.0, 82.0, 78.0, 95.0, 88.0, 92.0, 90.0]
multiagent_main_planning_time_1st = [0.015987, 0.00400896, 0.0159952, 0.00796232, 0.0279691, 0.0319743, 0.0199628, 0.0319533, 0.0399737, 0.0279646, 0.0519721, 0.0479565, 0.091964, 0.0599792, 0.460847, 0.267858, 5.60631, 0.172199, 0.135935, 0.0759738]
multiagent_main_cost_1st = [48, 40, 48, 40, 56, 56, 52, 67, 70, 74, 82, 74, 106, 125, 141, 124, 193, 106, 152, 119]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [38.0, 35.0, 38.0, 35.0, 47.0, 47.0, 48.0, 47.0, 64.0, 65.0, 62.0, 61.0, 80.0, 78.0, 83.0, 79.0, 96.0, 92.0, 93.0, 91.0]


singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [5.726691246032715, 4.534425973892212, 3.4865663051605225, 3.5896668434143066, 4.712530612945557, 3.438365936279297, 3.6048176288604736, 4.930582284927368, 4.90873122215271, 4.957699298858643, 4.43347692489624, 2.652284860610962, 3.0306272506713867, 3.5101301670074463, 5.4466211795806885, 4.476284027099609, 7.2041637897491455, 2.8404300212860107, 3.0909111499786377, 4.938364744186401]
LLM_pddl_sg_time = [1.251746416091919, 1.6300065517425537, 1.426356315612793, 2.010960817337036, 2.6587982177734375, 1.2046735286712646, 1.3709635734558105, 1.4955966472625732, 1.1845433712005615, 1.020787239074707, 1.0116636753082275, 0.8408055305480957, 1.1946675777435303, 1.1113195419311523, 1.17716646194458, 1.3505034446716309, 1.833319902420044, 1.1482033729553223, 1.9506864547729492, 1.1037473678588867]
multiagent_helper_planning_time = [0.21, 0.22, 0.22, 0.18, 0.24, 0.22, 0.2, 0.25, 0.28, 0.27, 0.27, 0.27, 0.36, 0.3, 0.31, 0.32, 0.5, 0.49, 0.51, 0.43]
multiagent_helper_planning_time_opt = [0.0119627, 0.0160097, 0.0159907, 0.0119952, 0.0239771, 0.0239883, 0.0159867, 0.0239699, 0.0319719, 0.035951, 0.0359694, 0.0359504, 0.0799374, 0.0519714, 0.0909971, 0.0879436, 0.0799675, 0.143943, 0.0959524, 0.0999459]
multiagent_helper_cost = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.00400012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400436, 0.0, 0.0, 0.0, 0.0, 0.00400307, 0.0, 0.0, 0.0, 0.0]
multiagent_helper_cost_1st = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [45.78, 48.35, 45.38, 48.94, 977.65, 999.83, 999.84, 1000.1, 1000.29, 1000.14, 1000.1, 999.82, 1000.22, 1000.23, 1000.29, 1000.15, 1000.24, 1000.2, 1000.34, 1000.27]
multiagent_main_planning_time_opt = [45.5625, 48.1575, 45.1546, 48.7232, 977.329, 157.906, 87.4047, 449.787, 212.648, 187.978, 351.682, 188.693, 7.97407, 21.1671, 34.0435, 16.6092, 261.843, 79.5774, 154.302, 110.946]
multiagent_main_cost = [34.0, 34.0, 34.0, 34.0, 46.0, 46.0, 47.0, 46.0, 60.0, 61.0, 61.0, 60.0, 79.0, 77.0, 82.0, 80.0, 95.0, 91.0, 92.0, 90.0]
multiagent_main_planning_time_1st = [0.0119979, 0.00400611, 0.00797972, 0.00796176, 0.0279709, 0.031983, 0.0199772, 0.0359723, 0.0399889, 0.0399397, 0.0519589, 0.0479857, 0.0919422, 0.0679557, 0.379807, 0.0959631, 5.30127, 0.183926, 0.139956, 0.10396]
multiagent_main_cost_1st = [48, 40, 48, 40, 56, 56, 52, 67, 70, 74, 82, 74, 106, 125, 141, 111, 193, 106, 120, 119]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [38.0, 35.0, 38.0, 35.0, 47.0, 47.0, 48.0, 47.0, 64.0, 65.0, 62.0, 61.0, 80.0, 78.0, 83.0, 81.0, 96.0, 95.0, 93.0, 91.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))

fig_barman_time = plt.figure()
plt.plot(multiagent_total_planning_time, 'b-') #, figure=fig_barman_time)

# fig_barman_length = plt.figure()
# plt.plot(overall_plan_length, '-b') #, figure=fig_barman_length)


print('blocksworld')
singleagent = []
helper = []
main = []

# blocksworld
singleagent_planning_time = [0.18, 0.14, 0.14, 0.13, 0.13, 0.15, 0.15, 0.19, 1.09, 1.41, 11.22, 12.58, 149.44, 72.03, 200.15, 200.18, 200.19, 200.17, 200.18, 200.2]
singleagent_planning_time_opt = [0.00397713, 0.00400058, 0.00399994, 0.00800655, 0.00798315, 0.0160104, 0.0159515, 0.0439606, 0.955799, 1.26373, 11.0946, 12.4337, 149.263, 71.8674, 0.103955, 0.00797469, 0.00399981, 0.451908, 0.0759739, 0.0239934]
singleagent_cost = [6.0, 6.0, 6.0, 12.0, 8.0, 12.0, 8.0, 14.0, 14.0, 18.0, 22.0, 20.0, 26.0, 22.0, 26.0, 28.0, 22.0, 30.0, 24.0, 30.0]
singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400001, 0.00400003, 0.0039993, 0.0, 0.00399952, 0.00400352, 0.00400414, 0.00398052, 0.00397885, 0.00400369, 0.00400365]
singleagent_cost_1st = [6, 6, 6, 16, 8, 12, 8, 16, 18, 32, 22, 40, 32, 48, 36, 38, 40, 32, 28, 42]
LLM_text_sg_time = [6.29708456993103, 3.505523443222046, 5.1963136196136475, 3.628025770187378, 3.470987319946289, 4.765545129776001, 4.414251327514648, 3.646660566329956, 3.7780380249023438, 3.336383104324341, 6.046425104141235, 3.458359718322754, 5.015369415283203, 3.495983362197876, 3.207846164703369, 5.376424551010132, 8.777496814727783, 3.6610774993896484, 3.4645166397094727, 6.452033758163452]
LLM_pddl_sg_time = [0.7849593162536621, 1.640486240386963, 0.8830809593200684, 0.8880758285522461, 1.3304901123046875, 0.9051671028137207, 0.7795524597167969, 1.0650107860565186, 0.9535703659057617, 0.9679889678955078, 1.295577049255371, 1.0650572776794434, 1.1462836265563965, 1.146913766860962, 0.7050473690032959, 1.900099277496338, 2.9946188926696777, 0.7169339656829834, 1.5152232646942139, 2.707277774810791]
multiagent_helper_planning_time = [0.12, 0.12, 0.14, 0.13, 0.13, 0.13, 0.15, 0.14, 0.14, 0.15, 0.16, 0.14, 0.14, 0.16, 0.17, 0.18, 48.83, 0.52, 0.18, 17.97]
multiagent_helper_planning_time_opt = [0.0, 0.0, 0.0, 0.00397988, 0.00397391, 0.00397462, 0.00400115, 0.0, 0.00399966, 0.0, 0.00795263, 0.00399993, 0.0, 0.0120116, 0.00400222, 0.00400283, 48.6654, 0.359867, 0.00397566, 17.7918]
multiagent_helper_cost = [2.0, 2.0, 2.0, 6.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 10.0, 2.0, 2.0, 6.0, 2.0, 2.0, 16.0, 14.0, 4.0, 14.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00395457, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
multiagent_helper_cost_1st = [2, 2, 2, 6, 6, 2, 2, 2, 2, 2, 10, 2, 2, 6, 2, 2, 16, 14, 4, 14]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.13, 0.12, 0.13, 0.13, 0.13, 0.15, 0.13, 0.16, 2.35, 2.28, 8.24, 13.87, 198.32, 97.68, 504.9, 999.41, 999.81, 1000.35, 1000.49, 1000.33]
multiagent_main_planning_time_opt = [0.00800559, 0.00397914, 0.00398211, 0.0040011, 0.0, 0.0159862, 0.0119949, 0.0439566, 2.21946, 2.14749, 8.1014, 13.7171, 198.178, 97.5205, 504.729, 0.0119785, 0.0, 0.135945, 0.00799364, 0.8596]
multiagent_main_cost = [4.0, 4.0, 4.0, 6.0, 2.0, 10.0, 6.0, 12.0, 16.0, 16.0, 12.0, 18.0, 26.0, 20.0, 24.0, 28.0, 14.0, 20.0, 20.0, 16.0]
multiagent_main_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.00397475, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400083, 0.0, 0.00398663, 0.00401967, 0.00399934, 0.00400386, 0.0]
multiagent_main_cost_1st = [4, 4, 4, 10, 2, 10, 6, 14, 16, 34, 12, 38, 32, 58, 34, 34, 26, 24, 44, 40]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [5.0, 5.0, 5.0, 11.0, 6.0, 11.0, 7.0, 13.0, 16.0, 16.0, 19.0, 19.0, 28.0, 20.0, 25.0, 28.0, 27.0, 33.0, 21.0, 26.0]


singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [3.9147446155548096, 4.821291446685791, 6.021789312362671, 5.0475311279296875, 3.3416128158569336, 4.436768531799316, 4.174423456192017, 3.6815919876098633, 6.7870423793792725, 3.1343672275543213, 3.819150447845459, 4.392451524734497, 3.8596532344818115, 6.446202278137207, 5.550197601318359, 4.011528491973877, 6.719103813171387, 3.1277945041656494, 6.09855580329895, 5.730147838592529]
LLM_pddl_sg_time = [2.251662254333496, 1.04134202003479, 0.7885560989379883, 0.7364270687103271, 1.3526670932769775, 1.0921545028686523, 0.956151008605957, 1.0628352165222168, 1.7274110317230225, 0.8948991298675537, 1.0983736515045166, 1.736943244934082, 1.5860366821289062, 1.4213836193084717, 1.2971127033233643, 0.9969446659088135, 1.6131269931793213, 1.2877724170684814, 0.7241685390472412, 0.7662150859832764]
multiagent_helper_planning_time = [0.13, 0.13, 0.14, 0.13, 0.13, 0.13, 0.13, 0.13, 0.14, 0.14, 0.16, 0.19, 0.15, 0.16, 0.16, 0.16, 0.21, 0.16, 0.16, 0.14]
multiagent_helper_planning_time_opt = [0.0, 0.0, 0.00400044, 0.0, 0.0, 0.00400078, 0.0, 0.00399104, 0.0, 0.00400026, 0.0159774, 0.0479735, 0.00400061, 0.0119754, 0.00399942, 0.00800632, 0.051965, 0.0, 0.00400042, 0.0]
multiagent_helper_cost = [2.0, 2.0, 2.0, 2.0, 6.0, 2.0, 2.0, 2.0, 2.0, 2.0, 12.0, 8.0, 2.0, 6.0, 2.0, 6.0, 8.0, 2.0, 2.0, 2.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
multiagent_helper_cost_1st = [2, 2, 2, 2, 6, 2, 2, 2, 2, 2, 12, 8, 2, 6, 2, 6, 8, 2, 2, 2]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.13, 0.12, 0.13, 0.14, 0.12, 0.14, 0.14, 0.16, 2.34, 2.28, 2.86, 13.61, 183.48, 93.55, 461.69, 923.06, 1000.17, 999.93, 1000.02, 1000.35]
multiagent_main_planning_time_opt = [0.00397751, 0.00397767, 0.00397797, 0.0039684, 0.00400007, 0.0119748, 0.0120077, 0.0439643, 2.19952, 2.14356, 2.72344, 13.4693, 183.304, 93.3856, 461.498, 922.79, 0.0719557, 0.443891, 0.0718175, 0.0359427]
multiagent_main_cost = [4.0, 4.0, 4.0, 10.0, 2.0, 10.0, 6.0, 12.0, 16.0, 16.0, 10.0, 14.0, 26.0, 20.0, 24.0, 22.0, 18.0, 28.0, 22.0, 28.0]
multiagent_main_planning_time_1st = [0.0, 0.00397767, 0.00397797, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399952, 0.0, 0.0, 0.00399997, 0.00401297, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400028]
multiagent_main_cost_1st = [4, 4, 4, 14, 2, 10, 6, 14, 16, 34, 18, 34, 32, 58, 34, 28, 22, 30, 26, 40]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [5.0, 5.0, 5.0, 11.0, 6.0, 11.0, 7.0, 13.0, 16.0, 16.0, 19.0, 17.0, 28.0, 20.0, 25.0, 25.0, 25.0, 29.0, 22.0, 29.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [6.676055669784546, 4.115707635879517, 3.6536245346069336, 4.253403186798096, 4.639393091201782, 3.478442668914795, 4.3452160358428955, 4.502277135848999, 4.326763391494751, 6.223667621612549, 5.969688892364502, 7.679861307144165, 5.730375528335571, 4.758274078369141, 3.800812005996704, 4.579333305358887, 4.220141410827637, 3.710618734359741, 9.850685834884644, 5.298738718032837]
LLM_pddl_sg_time = [0.8682422637939453, 0.8496410846710205, 1.0173513889312744, 0.9325296878814697, 1.0712203979492188, 0.832155704498291, 0.9299802780151367, 0.8096020221710205, 0.84427809715271, 1.6165039539337158, 1.3704359531402588, 1.5492851734161377, 0.9085698127746582, 1.42095947265625, 0.8244838714599609, 0.9217672348022461, 1.4516043663024902, 1.5836772918701172, 2.036567211151123, 1.0427191257476807]
multiagent_helper_planning_time = [0.13, 0.13, 0.12, 0.12, 0.14, 0.14, 0.13, 0.14, 0.15, 0.13, 0.15, 0.19, 0.14, 0.16, 0.16, 0.15, 0.19, 0.13, 0.15, 2.54]
multiagent_helper_planning_time_opt = [0.0, 0.00400007, 0.0, 0.00398794, 0.00397696, 0.00400014, 0.0, 0.00800056, 0.0040006, 0.00400166, 0.00800242, 0.0479527, 0.00400003, 0.0119938, 0.00400118, 0.00400352, 0.0519581, 0.0040003, 0.00401354, 2.36761]
multiagent_helper_cost = [2.0, 2.0, 2.0, 2.0, 6.0, 2.0, 2.0, 10.0, 2.0, 2.0, 10.0, 8.0, 2.0, 6.0, 2.0, 6.0, 8.0, 2.0, 2.0, 12.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.00398794, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0040001, 0.0, 0.0, 0.0]
multiagent_helper_cost_1st = [2, 2, 2, 2, 6, 2, 2, 10, 2, 2, 10, 8, 2, 6, 2, 6, 8, 2, 2, 12]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.13, 0.14, 0.13, 0.13, 0.14, 0.15, 0.15, 0.17, 1.33, 2.28, 8.07, 14.64, 195.03, 92.36, 492.33, 997.96, 1000.17, 1000.25, 1000.3, 1000.41]
multiagent_main_planning_time_opt = [0.00400071, 0.00397808, 0.00399883, 0.00800259, 0.0, 0.011993, 0.0119825, 0.0399677, 1.19969, 2.13952, 7.9263, 14.4966, 194.876, 92.2133, 492.134, 997.716, 0.0759675, 0.470882, 0.067975, 0.0119731]
multiagent_main_cost = [4.0, 4.0, 4.0, 10.0, 2.0, 10.0, 6.0, 6.0, 12.0, 16.0, 12.0, 14.0, 26.0, 20.0, 24.0, 22.0, 18.0, 28.0, 22.0, 18.0]
multiagent_main_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399993, 0.0, 0.00399982, 0.0, 0.00400088, 0.0, 0.0039989]
multiagent_main_cost_1st = [4, 4, 4, 14, 2, 10, 6, 6, 16, 34, 12, 34, 32, 58, 34, 28, 22, 30, 26, 50]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [5.0, 5.0, 5.0, 11.0, 6.0, 11.0, 7.0, 16.0, 13.0, 16.0, 19.0, 17.0, 28.0, 20.0, 25.0, 25.0, 25.0, 29.0, 22.0, 27.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))

# fig_barman_time = plt.figure()
# plt.plot(multiagent_total_planning_time, 'b-') #, figure=fig_barman_time)

# fig_barman_length = plt.figure()
# plt.plot(overall_plan_length, '-b') #, figure=fig_barman_length)


print('termes')
singleagent = []
helper = []
main = []
# termes
singleagent_planning_time = [26.27, 184.68, 200.19, 200.18, 200.15, 200.2, 200.24, 200.24, 200.25, 200.25, 200.17, 74.14, 200.17, 200.18, 200.17, 200.2, 200.16, 200.18, 200.18, 200.18]
singleagent_planning_time_opt = [26.0271, 184.462, 1.32375, 31.8662, 103.652, 147.406, 20.6325, 5.56704, 74.2997, 23.2517, 1.42769, 73.9818, 63.4288, 46.424, 132.612, 41.7586, 48.803, 12.4497, 28.9865, 109.376]
singleagent_cost = [36.0, 54.0, 68.0, 80.0, 144.0, 172.0, 56.0, 66.0, 112.0, 100.0, 66.0, 46.0, 92.0, 108.0, 148.0, 164.0, 116.0, 76.0, 94.0, 112.0]
singleagent_planning_time_1st = [0.0119765, 0.175942, 0.175948, 0.187941, 12.3697, 3.76737, 0.143946, 0.0440838, 4.7872, 3.00752, 0.0239775, 0.0319928, 0.143954, 0.091952, 2.20758, 1.03197, 0.35593, 0.0239772, 0.403901, 0.515874]
singleagent_cost_1st = [66, 108, 138, 270, 258, 224, 152, 104, 298, 220, 142, 150, 202, 274, 310, 382, 292, 134, 170, 266]
LLM_text_sg_time = [6.097562789916992, 3.267554521560669, -1, 6.109844207763672, 5.607317686080933, 8.956210613250732, 4.085831165313721, -1, 2.4334428310394287, 3.8602025508880615, 6.133823394775391, 4.297271251678467, 4.538675546646118, 2.2476372718811035, 3.6223037242889404, 6.184155702590942, 3.4470112323760986, 4.280899286270142, 3.607539176940918, 4.911821126937866]
LLM_pddl_sg_time = [5.2873923778533936, 4.8588104248046875, -1, 7.118252992630005, 6.403248310089111, 4.9003520011901855, 8.507120132446289, -1, 7.073056936264038, 7.415719270706177, 5.3035595417022705, 5.540433406829834, 4.026700973510742, 1.0832784175872803, 6.777920722961426, 7.725134372711182, 1.1601479053497314, 1.2505109310150146, 6.036208868026733, 1.4494972229003906]
multiagent_helper_planning_time = [0.19, 6.49, -1, 1000.13, 0.44, 999.74, 0.46, -1, 19.49, 1.16, 19.01, 0.53, 5.5, 4.05, 7.49, 0.55, 3.18, 0.28, 0.53, 0.31]
multiagent_helper_planning_time_opt = [0.00800151, 6.29451, -1, 7.25857, 0.227893, 358.678, 0.203925, -1, 19.2089, 0.879802, 18.8235, 0.343875, 5.29479, 3.8231, 7.22951, 0.327194, 2.93367, 0.0559705, 0.315323, 0.0559835]
multiagent_helper_cost = [4.0, 32.0, -1, 80.0, 14.0, 130.0, 10.0, -1, 26.0, 14.0, 36.0, 14.0, 32.0, 25.0, 32.0, 14.0, 25.0, 10.0, 14.0, 10.0]
multiagent_helper_planning_time_1st = [0.0, 0.01201, -1, 0.311921, 0.0119984, 12.8597, 0.0, -1, 0.0119734, 0.015994, 0.0479604, 0.0, 0.00799258, 0.01199, 0.0199863, 0.00400394, 0.0119857, 0.0, 0.00399133, 0.0]
multiagent_helper_cost_1st = [4, 88, -1, 188, 74, 252, 10, -1, 82, 56, 112, 14, 88, 87, 88, 14, 97, 10, 14, 10]
multiagent_helper_success = [True, True, 0, True, True, True, True, 0, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [17.26, 15.3, -1, 4.65, 1000.0, 1000.08, 1000.23, -1, 1000.03, 1000.08, 734.44, 54.36, 851.05, 1000.1, 1000.17, 999.86, 1000.06, 589.41, 999.75, 999.96]
multiagent_main_planning_time_opt = [17.0724, 15.1089, -1, 4.44321, 388.124, 108.984, 12.8964, -1, 271.258, 506.805, 734.229, 54.1695, 850.745, 74.2203, 850.516, 416.479, 31.2476, 589.155, 116.835, 212.414]
multiagent_main_cost = [30.0, 32.0, -1, 32.0, 132.0, 128.0, 48.0, -1, 88.0, 72.0, 64.0, 40.0, 72.0, 78.0, 140.0, 148.0, 94.0, 66.0, 90.0, 96.0]
multiagent_main_planning_time_1st = [0.0159923, 0.123963, -1, 0.0120126, 13.7753, 3.37101, 0.135952, -1, 5.00623, 3.19543, 0.0239803, 0.0399778, 0.123938, 0.0839538, 2.58728, 1.23969, 0.483843, 0.0399338, 0.495858, 0.571856]
multiagent_main_cost_1st = [58, 44, -1, 86, 266, 164, 86, -1, 268, 208, 128, 126, 132, 270, 300, 332, 234, 132, 166, 306]
multiagent_main_success = [True, True, False, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [30.0, 58.0, 1000000.0, 102.0, 142.0, 239.0, 50.0, 1000000.0, 96.0, 72.0, 89.0, 46.0, 95.0, 90.0, 161.0, 148.0, 108.0, 66.0, 100.0, 96.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [5.010183095932007, 4.348458528518677, -1, 4.22338342666626, -1, 5.058634996414185, 4.636968374252319, 4.879310369491577, 3.5983104705810547, 6.040338516235352, 3.731416702270508, 3.842924118041992, 3.5705742835998535, 4.253904819488525, 4.70227313041687, 5.177730083465576, 5.120783567428589, 5.489572763442993, 3.768582582473755, 3.838406801223755]
LLM_pddl_sg_time = [7.232457876205444, 8.731390476226807, -1, 5.4088499546051025, -1, 6.242907285690308, 8.590171575546265, 8.185201406478882, 6.996794700622559, 8.688905715942383, 4.547187089920044, 4.042669773101807, 6.62408971786499, 4.100301504135132, 4.830919504165649, 5.661770820617676, 1.4540884494781494, 1.3962225914001465, 7.613854646682739, 3.848283529281616]
multiagent_helper_planning_time = [0.19, 5.7, -1, 999.95, -1, 1000.04, 0.25, 289.5, 19.8, 1.14, 27.94, 0.92, 15.95, 11.88, 5.77, 0.49, 6.09, 0.22, 0.37, 2.67]
multiagent_helper_planning_time_opt = [0.00393648, 5.5189, -1, 945.05, -1, 273.885, 0.015984, 289.254, 19.5121, 0.847828, 27.7601, 0.73168, 15.7349, 11.6818, 5.55876, 0.271913, 5.89831, 0.0359741, 0.139916, 2.46328]
multiagent_helper_cost = [4.0, 32.0, -1, 79.0, -1, 130.0, 6.0, 36.0, 26.0, 14.0, 36.0, 18.0, 32.0, 36.0, 32.0, 14.0, 28.0, 9.0, 12.0, 26.0]
multiagent_helper_planning_time_1st = [0.0, 0.0119562, -1, 0.0599642, -1, 10.2419, 0.0, 0.0319808, 0.0119839, 0.0159707, 0.00800468, 0.0280074, 0.00400831, 0.0080017, 0.0119871, 0.00797477, 0.0, 0.0, 0.0, 0.00800721]
multiagent_helper_cost_1st = [4, 88, -1, 83, -1, 252, 6, 110, 82, 56, 40, 76, 46, 64, 88, 42, 28, 9, 12, 54]
multiagent_helper_success = [True, True, 0, True, 0, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [15.63, 14.15, -1, 4.3, -1, 1000.08, 999.8, 1000.02, 1000.18, 1000.1, 226.26, 40.56, 787.59, 1000.06, 999.73, 999.98, 1000.14, 604.98, 999.9, 999.93]
multiagent_main_planning_time_opt = [15.4323, 13.9684, -1, 4.07898, -1, 108.408, 12.9572, 1.16374, 272.626, 515.877, 226.06, 40.3678, 787.322, 77.7948, 716.91, 363.379, 27.1463, 604.718, 64.7871, 264.482]
multiagent_main_cost = [30.0, 32.0, -1, 32.0, -1, 128.0, 48.0, 50.0, 88.0, 72.0, 50.0, 38.0, 72.0, 88.0, 140.0, 148.0, 94.0, 66.0, 86.0, 92.0]
multiagent_main_planning_time_1st = [0.0119782, 0.123964, -1, 0.0119712, -1, 3.35932, 0.111955, 0.0519623, 5.0026, 3.7434, 0.0159789, 0.03998, 0.123942, 0.0919565, 2.00757, 1.33571, 0.271929, 0.0439948, 0.427856, 0.403872]
multiagent_main_cost_1st = [58, 44, -1, 86, -1, 164, 86, 120, 268, 208, 92, 140, 132, 306, 300, 332, 234, 132, 202, 278]
multiagent_main_success = [True, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [30.0, 58.0, 1000000.0, 104.0, 1000000.0, 239.0, 50.0, 50.0, 96.0, 72.0, 81.0, 47.0, 87.0, 111.0, 161.0, 148.0, 108.0, 66.0, 92.0, 100.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [5.7995946407318115, 5.043030500411987, -1, 4.322331190109253, 6.111256837844849, 4.693840503692627, 5.691729545593262, 10.231188297271729, -1, 3.848888635635376, 4.2423272132873535, -1, 3.8979010581970215, 4.421138048171997, 3.707953691482544, 3.074275493621826, 3.379809617996216, 6.666421175003052, 3.56565260887146, 3.361194610595703]
LLM_pddl_sg_time = [7.380004167556763, 8.762645959854126, -1, 8.997777462005615, 4.668330430984497, 8.132147073745728, 8.478479623794556, 8.509749174118042, -1, 11.835912704467773, 4.137854099273682, -1, 4.768259286880493, 5.323733568191528, 4.352862596511841, 6.747029781341553, 1.2706983089447021, 1.49330472946167, 5.503289222717285, 7.765336990356445]
multiagent_helper_planning_time = [0.19, 6.19, -1, 1000.09, 0.54, 999.68, 0.26, 282.28, -1, 1.15, 11.26, -1, 6.52, 14.59, 6.83, 0.52, 2.86, 0.22, 1.1, 0.47]
multiagent_helper_planning_time_opt = [0.00400194, 5.99436, -1, 7.20235, 0.30389, 327.006, 0.023975, 282.009, -1, 0.875799, 11.0737, -1, 6.21799, 14.3881, 6.65288, 0.279912, 2.63921, 0.0653604, 0.847746, 0.211949]
multiagent_helper_cost = [2.0, 32.0, -1, 80.0, 14.0, 130.0, 6.0, 36.0, -1, 14.0, 36.0, -1, 32.0, 36.0, 32.0, 14.0, 25.0, 9.0, 20.0, 14.0]
multiagent_helper_planning_time_1st = [0.00400194, 0.0119946, -1, 0.415868, 0.0159996, 12.3058, 0.0, 0.0399782, -1, 0.0119767, 0.00800194, -1, 0.0159654, 0.0119951, 0.0119757, 0.00400368, 0.0198824, 0.0, 0.00797405, 0.0]
multiagent_helper_cost_1st = [2, 88, -1, 188, 74, 252, 6, 110, -1, 56, 64, -1, 88, 64, 88, 42, 97, 9, 60, 46]
multiagent_helper_success = [True, True, 0, True, True, True, True, True, 0, True, True, 0, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [25.95, 14.75, -1, 4.26, 1000.13, 1000.09, 1000.26, 1000.06, -1, 1000.1, 245.89, -1, 845.26, 999.78, 1000.04, 999.76, 1000.01, 777.71, 1000.0, 1000.1]
multiagent_main_planning_time_opt = [25.7685, 14.548, -1, 4.05079, 384.898, 112.163, 12.477, 0.863706, -1, 505.425, 245.697, -1, 844.976, 91.8988, 797.146, 417.495, 31.7269, 777.516, 14.6921, 206.594]
multiagent_main_cost = [34.0, 32.0, -1, 32.0, 132.0, 128.0, 48.0, 50.0, -1, 72.0, 50.0, -1, 72.0, 88.0, 140.0, 148.0, 94.0, 68.0, 82.0, 100.0]
multiagent_main_planning_time_1st = [0.0159833, 0.127901, -1, 0.0119707, 13.3173, 3.85532, 0.135942, 0.0399709, -1, 3.41131, 0.0160068, -1, 0.0919592, 0.0759668, 2.11542, 1.28087, 0.451776, 0.0199987, 0.315914, 0.379893]
multiagent_main_cost_1st = [72, 44, -1, 86, 266, 164, 86, 120, -1, 208, 92, -1, 132, 306, 300, 332, 234, 118, 178, 300]
multiagent_main_success = [True, True, False, True, True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True]
overall_plan_length = [34.0, 58.0, 1000000.0, 102.0, 142.0, 239.0, 50.0, 50.0, 1000000.0, 72.0, 81.0, 1000000.0, 95.0, 111.0, 161.0, 148.0, 108.0, 68.0, 94.0, 100.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))


# fig_barman_time = plt.figure()
# plt.plot(multiagent_total_planning_time, 'b-') #, figure=fig_barman_time)

# fig_barman_length = plt.figure()
# plt.plot(overall_plan_length, '-b') #, figure=fig_barman_length)

print('tyreworld')
singleagent = []
helper = []
main = []

#tyreworld
singleagent_planning_time = [0.25, 200.32, 200.16, 200.22, 200.26, 200.26, 200.27, 200.31, 200.29, 200.33, 200.31, 200.37, 200.42, 200.42, 200.44, 200.48, 200.47, 200.5, 200.54, 200.58]
singleagent_planning_time_opt = [0.0799144, 0.10796, 0.00399893, 0.00799589, 0.0199865, 0.0319657, 0.0479648, 0.0719439, 0.107956, 0.131949, 0.179941, 0.259916, 0.351869, 0.467857, 0.447893, 0.827747, 0.999708, 1.1901, 1.29568, 1.58758]
singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
singleagent_planning_time_1st = [0.0, 0.10796, 0.00399893, 0.00799589, 0.0199865, 0.0319657, 0.0479648, 0.0719439, 0.107956, 0.131949, 0.179941, 0.259916, 0.351869, 0.467857, 0.447893, 0.827747, 0.999708, 1.1901, 1.29568, 1.58758]
singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]
LLM_text_sg_time = [6.082302808761597, -1, -1, 3.128366231918335, 3.4409360885620117, 5.926218509674072, -1, 5.233426570892334, 4.4269373416900635, -1, 2.7469539642333984, -1, 2.5881669521331787, 4.160115718841553, 2.722769260406494, 3.8586084842681885, 8.346668004989624, 4.941295146942139, 2.4372239112854004, 3.0551671981811523]
LLM_pddl_sg_time = [0.9724795818328857, -1, -1, 1.4340028762817383, 2.336860179901123, 1.526453971862793, -1, 1.2633142471313477, 2.438910961151123, -1, 2.3731839656829834, -1, 3.3715906143188477, 3.0850775241851807, 5.225464344024658, 3.5900845527648926, 5.4373743534088135, 4.093838214874268, 6.402360439300537, 4.964510440826416]
multiagent_helper_planning_time = [0.13, -1, -1, 0.15, 0.14, 0.17, -1, 782.23, 0.27, -1, 0.32, -1, 0.67, 1.13, 1.86, 4.35, 7.87, 18.11, 1000.48, 68.9]
multiagent_helper_planning_time_opt = [0.0, -1, -1, 0.00397905, 0.0040013, 0.00400007, -1, 781.983, 0.0759612, -1, 0.0879456, -1, 0.391861, 0.827782, 1.53957, 3.97862, 7.37217, 17.675, 0.0, 68.3785]
multiagent_helper_cost = [1.0, -1, -1, 4.0, 5.0, 6.0, -1, 11.0, 9.0, -1, 11.0, -1, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
multiagent_helper_planning_time_1st = [0.0, -1, -1, 0.0, 0.0, 0.0, -1, 0.0, 0.0, -1, 0.0, -1, 0.00397913, 0.0, 0.00400567, 0.00397911, 0.00396982, 0.00400047, 0.0, 0.00398083]
multiagent_helper_cost_1st = [1, -1, -1, 4, 5, 6, -1, 11, 9, -1, 11, -1, 13, 14, 15, 16, 17, 18, 19, 20]
multiagent_helper_success = [True, 0, 0, True, True, True, 0, True, True, 0, True, 0, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.16, -1, -1, 1000.68, 1000.28, 1000.6, -1, 1000.94, 1000.81, -1, 1000.53, -1, 1000.47, 1000.75, 1000.69, 1000.63, 1000.94, 1000.85, 1000.74, 1000.72]
multiagent_main_planning_time_opt = [0.0399497, -1, -1, 0.00800152, 0.0119933, 0.0239811, -1, 0.051961, 0.0639406, -1, 0.147939, -1, 0.295903, 0.395876, 0.375877, 0.671794, 1.03431, 1.23954, 1.52743, 2.07923]
multiagent_main_cost = [12.0, -1, -1, 42.0, 52.0, 62.0, -1, 79.0, 92.0, -1, 112.0, -1, 132.0, 142.0, 152.0, 162.0, 172.0, 182.0, 192.0, 202.0]
multiagent_main_planning_time_1st = [0.0, -1, -1, 0.00800152, 0.0119933, 0.0239811, -1, 0.051961, 0.0639406, -1, 0.147939, -1, 0.295903, 0.395876, 0.375877, 0.671794, 1.03431, 1.23954, 1.52743, 2.07923]
multiagent_main_cost_1st = [12, -1, -1, 42, 52, 62, -1, 79, 92, -1, 112, -1, 132, 142, 152, 162, 172, 182, 192, 202]
multiagent_main_success = [True, False, False, True, True, True, False, True, True, False, True, False, True, True, True, True, True, True, True, True]
overall_plan_length = [12.0, 1000000.0, 1000000.0, 42.0, 52.0, 62.0, 1000000.0, 79.0, 92.0, 1000000.0, 112.0, 1000000.0, 132.0, 142.0, 152.0, 162.0, 172.0, 182.0, 192.0, 202.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [3.7485482692718506, 2.675309419631958, 4.241580247879028, -1, 2.7877755165100098, 5.114909410476685, 5.460669040679932, 4.5549845695495605, 3.512289524078369, 3.4981327056884766, 4.355679512023926, 3.9622042179107666, 3.3494372367858887, 4.6393349170684814, 5.6120710372924805, -1, 4.495105266571045, 3.1915602684020996, 3.459794521331787, 3.924915313720703]
LLM_pddl_sg_time = [0.829735279083252, 4.301196098327637, 0.920576810836792, -1, 1.359968900680542, 1.88289475440979, 2.112245559692383, 2.320904493331909, 3.4483628273010254, 5.2768285274505615, 2.40199613571167, 4.4645514488220215, 4.672170162200928, 6.848940134048462, 4.210396766662598, -1, 4.104921579360962, 7.522744655609131, 5.626300811767578, 5.20188307762146]
multiagent_helper_planning_time = [0.13, 1000.02, 0.14, -1, 0.12, 0.17, 0.19, 0.19, 0.22, 1000.01, 0.23, 0.49, 1000.29, 1000.5, 2.32, -1, 36.84, 1000.55, 39.01, 71.39]
multiagent_helper_planning_time_opt = [0.00398124, 274.82, 0.00397453, -1, 0.00399044, 0.00399971, 0.00799119, 0.0119784, 0.0207727, 0.0, 0.0639698, 0.19992, 0.00399099, 0.00401668, 1.88755, -1, 36.4276, 0.00400185, 38.4739, 70.831]
multiagent_helper_cost = [1.0, 9.0, 1.0, -1, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, -1, 17.0, 18.0, 19.0, 20.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.00397453, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400462, 0.00399099, 0.00401668, 0.0, -1, 0.00398346, 0.00400185, 0.0, 0.0]
multiagent_helper_cost_1st = [1, 9, 1, -1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, 17, 18, 19, 20]
multiagent_helper_success = [True, True, True, 0, True, True, True, True, True, True, True, True, True, True, True, 0, True, True, True, True]
multiagent_main_planning_time = [0.17, 1000.6, 1000.97, -1, 1000.92, 1000.76, 1000.85, 1000.66, 1000.94, 1000.88, 1000.68, 1000.77, 1000.73, 1000.46, 1000.69, -1, 1000.85, 1000.96, 1000.76, 1000.98]
multiagent_main_planning_time_opt = [0.0359826, 0.0879513, 0.00400001, -1, 0.0197242, 0.0239685, 0.0359759, 0.0439676, 0.0879636, 0.0799505, 0.143826, 0.151949, 0.247866, 0.567817, 0.54377, -1, 1.11926, 1.44725, 1.39936, 1.75369]
multiagent_main_cost = [12.0, 92.0, 34.0, -1, 52.0, 62.0, 72.0, 82.0, 92.0, 102.0, 112.0, 122.0, 132.0, 142.0, 152.0, -1, 172.0, 182.0, 192.0, 202.0]
multiagent_main_planning_time_1st = [0.0, 0.0879513, 0.00400001, -1, 0.0197242, 0.0239685, 0.0359759, 0.0439676, 0.0879636, 0.0799505, 0.143826, 0.151949, 0.247866, 0.567817, 0.54377, -1, 1.11926, 1.44725, 1.39936, 1.75369]
multiagent_main_cost_1st = [12, 92, 34, -1, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, -1, 172, 182, 192, 202]
multiagent_main_success = [True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True]
overall_plan_length = [12.0, 92.0, 34.0, 1000000.0, 52.0, 62.0, 72.0, 82.0, 92.0, 102.0, 112.0, 122.0, 132.0, 142.0, 152.0, 1000000.0, 172.0, 182.0, 192.0, 202.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [4.301205158233643, 4.235186815261841, 4.124997138977051, 6.739498853683472, 6.507340908050537, 5.352238416671753, 6.082157373428345, 5.685604810714722, 3.5893099308013916, -1, 2.631683111190796, -1, 4.205114126205444, 4.241487264633179, -1, 4.233266592025757, -1, 2.819767951965332, 2.8724842071533203, -1]
LLM_pddl_sg_time = [0.8684737682342529, 3.4373395442962646, 1.6787822246551514, 1.2600138187408447, 1.526252269744873, 3.2310636043548584, 2.112204074859619, 2.233955144882202, 2.8741886615753174, -1, 2.301515817642212, -1, 2.784573554992676, 2.784491539001465, -1, 4.318187952041626, -1, 6.969059228897095, 5.273959398269653, -1]
multiagent_helper_planning_time = [0.12, 0.22, 0.13, 0.14, 0.15, 0.17, 0.16, 0.17, 0.2, -1, 0.33, -1, 0.72, 1.32, -1, 3.84, -1, 17.78, 36.33, -1]
multiagent_helper_planning_time_opt = [0.0, 0.0239844, 0.0, 0.00397563, 0.00397732, 0.00400111, 0.00800298, 0.0119847, 0.0239662, -1, 0.0879612, -1, 0.463857, 1.0196, -1, 3.45262, -1, 17.3331, 35.8518, -1]
multiagent_helper_cost = [1.0, 9.0, 1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -1, 11.0, -1, 13.0, 14.0, -1, 16.0, -1, 18.0, 19.0, -1]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.00397732, 0.0, 0.0, 0.0, 0.0, -1, 0.0, -1, 0.0, 0.0, -1, 0.0, -1, 0.0, 0.0, -1]
multiagent_helper_cost_1st = [1, 9, 1, 4, 5, 6, 7, 8, 9, -1, 11, -1, 13, 14, -1, 16, -1, 18, 19, -1]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, 0, True, 0, True, True, 0, True, 0, True, True, 0]
multiagent_main_planning_time = [0.16, 1000.68, 1000.81, 999.58, 1001.16, 1001.1, 1000.8, 1000.65, 1000.34, -1, 1000.58, -1, 1000.72, 1000.66, -1, 1000.8, -1, 1000.78, 1000.82, -1]
multiagent_main_planning_time_opt = [0.0359754, 0.0919457, 0.00400554, 0.00399971, 0.0119837, 0.0239587, 0.0399842, 0.0599607, 0.0919692, -1, 0.151937, -1, 0.291905, 0.367774, -1, 0.575659, -1, 1.49538, 1.43537, -1]
multiagent_main_cost = [12.0, 92.0, 34.0, 42.0, 52.0, 62.0, 72.0, 82.0, 92.0, -1, 112.0, -1, 132.0, 142.0, -1, 162.0, -1, 182.0, 192.0, -1]
multiagent_main_planning_time_1st = [0.0, 0.0919457, 0.00400554, 0.00399971, 0.0119837, 0.0239587, 0.0399842, 0.0599607, 0.0919692, -1, 0.151937, -1, 0.291905, 0.367774, -1, 0.575659, -1, 1.49538, 1.43537, -1]
multiagent_main_cost_1st = [12, 92, 34, 42, 52, 62, 72, 82, 92, -1, 112, -1, 132, 142, -1, 162, -1, 182, 192, -1]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, False, True, False, True, True, False, True, False, True, True, False]
overall_plan_length = [12.0, 92.0, 34.0, 42.0, 52.0, 62.0, 72.0, 82.0, 92.0, 1000000.0, 112.0, 1000000.0, 132.0, 142.0, 1000000.0, 162.0, 1000000.0, 182.0, 192.0, 1000000.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))
# fig_barman_time = plt.figure()
# plt.plot(multiagent_total_planning_time, 'b-') #, figure=fig_barman_time)

# fig_barman_length = plt.figure()
# plt.plot(overall_plan_length, '-b') #, figure=fig_barman_length)

print('grippers')
singleagent = []
helper = []
main = []

#grippers

singleagent_planning_time = [0.14, 0.15, 0.15, 0.12, 0.13, 0.12, 0.15, 0.15, 6.91, 0.13, 0.12, 0.18, 0.09, 0.17, 0.18, 17.32, 0.12, 0.2, 4.12, 0.12]
singleagent_planning_time_opt = [0.00399954, 0.0199852, 0.00800343, 0.00399974, 0.00397479, 0.0, 0.0199503, 0.0239741, 6.78239, 0.00797872, 0.00393889, 0.0639596, 0.00397335, 0.0159758, 0.0439563, 17.1725, 0.0119918, 0.0879558, 3.98302, 0.0080041]
singleagent_cost = [5.0, 9.0, 6.0, 4.0, 4.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 6.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399972]
singleagent_cost_1st = [5, 10, 6, 4, 4, 4, 8, 13, 20, 10, 3, 9, 6, 6, 12, 22, 9, 10, 21, 7]
LLM_text_sg_time = [5.658768177032471, 2.6388728618621826, 2.8199551105499268, 4.178875207901001, 2.2543814182281494, 2.815185546875, 4.1606245040893555, 2.484114646911621, 2.606924533843994, 4.4416327476501465, 3.5869719982147217, 3.1442270278930664, 3.102780342102051, 2.8653290271759033, 4.442960977554321, 2.6099441051483154, 3.6116535663604736, 3.7325727939605713, 2.511477470397949, 2.5214052200317383]
LLM_pddl_sg_time = [1.2027511596679688, 1.347466230392456, 1.1218810081481934, 1.8704848289489746, 1.8118767738342285, 2.216928720474243, 1.068586826324463, 1.3565740585327148, 1.1936378479003906, 1.0590474605560303, 1.0315029621124268, 2.762011766433716, 1.014646291732788, 0.9529585838317871, 1.4283149242401123, 1.4787030220031738, 1.1969187259674072, 1.2997372150421143, 1.0235297679901123, 1.233590841293335]
multiagent_helper_planning_time = [0.14, 0.12, 0.13, 0.13, 0.14, 0.12, 0.13, 0.14, 0.14, 0.14, 0.11, -1, 0.12, 0.12, 0.13, 0.14, 0.09, 0.12, 0.13, 0.12]
multiagent_helper_planning_time_opt = [0.0, 0.00398455, 0.0, 0.00399938, 0.00397801, 0.00397282, 0.00399973, 0.00400012, 0.00400156, 0.0, 0.00400028, -1, 0.00399969, 0.00397354, 0.00399884, 0.00800007, 0.00397273, 0.00399955, 0.00399901, 0.0039996]
multiagent_helper_cost = [3.0, 4.0, 3.0, 4.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, -1, 4.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
multiagent_helper_cost_1st = [3, 4, 3, 4, 4, 4, 4, 3, 5, 4, 3, -1, 4, 4, 4, 4, 3, 5, 4, 5]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.12, 0.12, 0.13, -1, -1, -1, 0.14, 0.13, 3.93, 0.13, -1, 0.18, 0.12, 0.12, 0.14, 7.95, 0.13, 0.16, 6.23, 0.13]
multiagent_main_planning_time_opt = [0.0, 0.0119952, 0.0, -1, -1, -1, 0.00800328, 0.0119772, 3.80324, 0.00400021, -1, 0.0639497, 0.0, 0.00399731, 0.0159923, 7.80945, 0.00800181, 0.0319732, 6.10261, 0.00399969]
multiagent_main_cost = [3.0, 7.0, 3.0, -1, -1, -1, 6.0, 8.0, 14.0, 6.0, -1, 9.0, 4.0, 4.0, 8.0, 16.0, 6.0, 7.0, 15.0, 4.0]
multiagent_main_planning_time_1st = [0.0, 0.0, 0.0, -1, -1, -1, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00397694, 0.0]
multiagent_main_cost_1st = [3, 7, 3, -1, -1, -1, 6, 9, 16, 6, -1, 9, 4, 4, 10, 19, 6, 7, 16, 4]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [3.0, 7.0, 3.0, 4, 4, 4, 6.0, 8.0, 14.0, 6.0, 3, 9, 4.0, 4.0, 8.0, 16.0, 6.0, 7.0, 15.0, 4.0]


singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)


LLM_text_sg_time = [4.658865928649902, 3.0660109519958496, 4.183920621871948, 4.489145994186401, 3.7882862091064453, 2.1066057682037354, 5.183727502822876, 3.522084951400757, 5.456824779510498, 3.6045801639556885, 4.985196113586426, 5.1655261516571045, 2.7261569499969482, 3.159611940383911, 2.6470787525177, 2.753760814666748, 5.67988920211792, 5.060324668884277, 5.257177829742432, 4.388478755950928]
LLM_pddl_sg_time = [2.5462276935577393, 1.8216047286987305, 1.8183422088623047, 1.4655156135559082, 1.496476173400879, 1.2606022357940674, 2.6719977855682373, 1.127819538116455, 1.2096340656280518, 2.083310842514038, 1.255431890487671, 1.6597621440887451, 1.157665729522705, 1.9167404174804688, 1.4508388042449951, 1.7370758056640625, 1.7354755401611328, 1.4890758991241455, 1.657271146774292, 1.3173484802246094]
multiagent_helper_planning_time = [0.12, 0.14, 0.12, 0.13, 0.12, 0.14, 0.14, 0.14, 0.14, 0.13, 0.13, 0.13, 0.12, 0.13, 0.13, 0.13, 0.13, 0.14, 0.09, 0.14]
multiagent_helper_planning_time_opt = [0.0080208, 0.0, 0.00400075, 0.00400064, 0.0, 0.00397714, 0.00398125, 0.00397407, 0.00399999, 0.00397797, 0.00397989, 0.00397711, 0.00397353, 0.00400048, 0.0, 0.00800036, 0.0, 0.00400066, 0.00400491, 0.0040006]
multiagent_helper_cost = [3.0, 4.0, 3.0, 4.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00397407, 0.0, 0.0, 0.0, 0.0, 0.00397353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
multiagent_helper_cost_1st = [3, 4, 3, 4, 4, 4, 4, 3, 5, 4, 3, 3, 4, 4, 4, 4, 3, 5, 4, 5]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.13, 0.14, 0.12, -1, -1, -1, 0.14, 0.14, 3.81, 0.13, -1, 0.16, 0.11, 0.12, 0.15, 8.83, 0.13, 0.18, 6.28, 0.13]
multiagent_main_planning_time_opt = [0.00392997, 0.0159375, 0.00397177, -1, -1, -1, 0.00800322, 0.0239657, 3.67514, 0.00399827, -1, 0.0199999, 0.00400044, 0.00399996, 0.0199875, 8.69035, 0.00798065, 0.0519355, 6.17702, 0.00400005]
multiagent_main_cost = [3.0, 7.0, 3.0, -1, -1, -1, 6.0, 8.0, 14.0, 6.0, -1, 7.0, 4.0, 4.0, 8.0, 16.0, 6.0, 7.0, 15.0, 4.0]
multiagent_main_planning_time_1st = [0.0, 0.0, 0.0, -1, -1, -1, 0.0, 0.00397954, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00397775, 0.0]
multiagent_main_cost_1st = [3, 7, 3, -1, -1, -1, 6, 9, 16, 6, -1, 7, 4, 4, 10, 19, 6, 7, 16, 4]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [3.0, 7.0, 3.0, 4, 4, 4, 6.0, 8.0, 14.0, 6.0, 3, 7.0, 4.0, 4.0, 8.0, 16.0, 6.0, 7.0, 15.0, 4.0]


singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

LLM_text_sg_time = [6.003830909729004, 3.779432535171509, 3.262190341949463, 3.7271668910980225, 6.200535774230957, -1, 2.644796133041382, 4.013469219207764, 2.7639429569244385, 5.106813907623291, 3.6198928356170654, 4.664839744567871, -1, 2.0250065326690674, 2.9225637912750244, 2.8861310482025146, 2.9227824211120605, 2.873654842376709, 2.914386510848999, 4.542729377746582]
LLM_pddl_sg_time = [1.9714324474334717, 1.5588138103485107, 1.305748701095581, 2.1014065742492676, 1.1828184127807617, -1, 1.6853954792022705, 2.3497426509857178, 1.801018238067627, 1.5694565773010254, 2.5390865802764893, 1.0167412757873535, -1, 1.1171481609344482, 1.2353508472442627, 1.5438294410705566, 0.9596869945526123, 1.1955573558807373, 1.4019298553466797, 1.3718671798706055]
multiagent_helper_planning_time = [0.12, 0.12, 0.13, 0.13, 0.12, -1, 0.13, 0.12, 0.12, 0.11, 0.12, 0.11, -1, 0.14, 0.13, 0.13, 0.12, 0.12, 0.13, 0.11]
multiagent_helper_planning_time_opt = [0.0039817, 0.00400044, 0.00399994, 0.00399997, 0.00399882, -1, 0.00397505, 0.00400029, 0.00400023, 0.00400178, 0.00397708, 0.0, -1, 0.00798794, 0.00399994, 0.00797152, 0.00401121, 0.00397272, 0.00796963, 0.00398125]
multiagent_helper_cost = [3.0, 4.0, 3.0, 4.0, 4.0, -1, 3.0, 4.0, 4.0, 4.0, 3.0, 3.0, -1, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.00397505, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
multiagent_helper_cost_1st = [3, 4, 3, 4, 4, -1, 3, 4, 5, 4, 3, 3, -1, 4, 4, 4, 4, 5, 4, 5]
multiagent_helper_success = [True, True, True, True, True, 0, True, True, True, True, True, True, 0, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.12, 0.14, 0.13, -1, -1, -1, 0.13, 0.14, 5.46, 0.14, -1, 0.15, -1, 0.15, 0.14, 8.41, 0.13, 0.16, 6.76, 0.12]
multiagent_main_planning_time_opt = [0.00399916, 0.0119946, 0.00399375, -1, -1, -1, 0.00800355, 0.0199824, 5.32852, 0.012014, -1, 0.0200017, -1, 0.00400196, 0.0159832, 8.27419, 0.00800266, 0.0359627, 6.62668, 0.00400092]
multiagent_main_cost = [3.0, 7.0, 3.0, -1, -1, -1, 6.0, 9.0, 14.0, 6.0, -1, 7.0, -1, 4.0, 8.0, 16.0, 6.0, 7.0, 15.0, 4.0]
multiagent_main_planning_time_1st = [0.0, 0.0, 0.00399375, -1, -1, -1, 0.0, 0.0, 0.0, 0.0, -1, 0.0, -1, 0.0, 0.00397542, 0.0, 0.0, 0.0, 0.0, 0.0]
multiagent_main_cost_1st = [3, 7, 3, -1, -1, -1, 6, 9, 16, 6, -1, 7, -1, 4, 10, 19, 6, 7, 16, 4]
multiagent_main_success = [True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True]
overall_plan_length = [3.0, 7.0, 3.0, 4, 4, 1000000.0, 6.0, 9.0, 14.0, 6.0, 3, 7.0, 1000000.0, 4.0, 8.0, 16.0, 6.0, 7.0, 15.0, 4.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))

# fig_barman_time = plt.figure()
# plt.plot(multiagent_total_planning_time, 'b-') #, figure=fig_barman_time)

# fig_barman_length = plt.figure()
# plt.plot(overall_plan_length, '-b') #, figure=fig_barman_length)

################################################
print('LLM-PDDL planning results')
print('barman')
singleagent = []
helper = []
main = []
    # barman
# singleagent_planning_time = [99.46, 47.96, 98.99, 47.66, 200.17, 200.22, 200.19, 200.2, 200.25, 200.25, 200.24, 200.27, 200.29, 200.27, 200.27, 200.29, 200.31, 200.3, 200.3, 200.31]
# singleagent_planning_time_opt = [99.2061, 47.7393, 98.7744, 47.4436, 13.0095, 13.0857, 120.338, 123.974, 4.82308, 2.62755, 9.71439, 2.54757, 20.2563, 27.6744, 48.8685, 25.4958, 0.223953, 88.3498, 105.63, 81.3814]
# singleagent_cost = [36.0, 36.0, 36.0, 36.0, 49.0, 49.0, 49.0, 49.0, 65.0, 65.0, 65.0, 65.0, 82.0, 84.0, 83.0, 82.0, 126.0, 92.0, 95.0, 94.0]
# singleagent_planning_time_1st = [0.0199729, 0.00797656, 0.0199915, 0.00801197, 0.0319723, 0.0439534, 0.0239877, 0.0479725, 0.0520516, 0.071968, 0.0519469, 0.0759757, 0.155942, 0.0799466, 0.119965, 0.0839681, 0.223953, 0.147943, 0.159958, 0.127974]
# singleagent_cost_1st = [48, 50, 48, 50, 67, 65, 56, 74, 74, 81, 82, 81, 107, 113, 106, 115, 126, 117, 134, 145]
# LLM_text_sg_time = [9.234002113342285, 8.692440509796143, 8.47049880027771, 6.878967523574829, 9.231892347335815, 8.5101797580719, 10.027989864349365, 8.974435329437256, 8.860485076904297, 8.93682312965393, 7.909400701522827, 10.544155597686768, 6.924746751785278, 6.497351169586182, 8.402292251586914, 8.75000810623169, 8.801286458969116, 10.742023229598999, 9.161345958709717, 8.66089391708374]
# LLM_pddl_sg_time = [4.1909871101379395, 3.58054780960083, 4.3335888385772705, 3.241687059402466, 4.410735368728638, 4.280261516571045, 3.434445381164551, 3.8706507682800293, 2.7275946140289307, 3.8210058212280273, 4.474370956420898, 3.924147129058838, 2.431321382522583, 4.242633581161499, 2.5231640338897705, 3.7969510555267334, 5.665758848190308, 3.2457480430603027, 3.5148582458496094, 2.654412031173706]
# multiagent_helper_planning_time = [0.38, 0.36, 0.4, 0.24, 0.6, 0.6, 0.6, 0.44, 21.34, 0.96, 0.96, 8.24, 31.64, 1.87, 0.38, 90.02, 2.68, 165.84, 2.67, 49.0]
# multiagent_helper_planning_time_opt = [0.179913, 0.175307, 0.179937, 0.0119946, 0.367897, 0.359906, 0.37185, 0.263928, 21.0805, 0.683827, 0.703492, 7.98269, 31.3305, 1.57173, 0.0719607, 89.7103, 2.34359, 165.514, 2.32764, 48.6693]
# multiagent_helper_cost = [7.0, 7.0, 7.0, 4.0, 7.0, 7.0, 7.0, 7.0, 11.0, 7.0, 7.0, 10.0, 10.0, 7.0, 4.0, 11.0, 7.0, 11.0, 7.0, 10.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399243, 0.0, 0.00400567, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# multiagent_helper_cost_1st = [8, 8, 8, 4, 8, 8, 8, 8, 12, 8, 8, 11, 11, 8, 4, 12, 8, 12, 8, 11]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [77.24, 83.92, 76.7, 44.99, 200.22, 200.2, 200.21, 200.16, 200.19, 200.23, 200.22, 200.18, 200.3, 200.28, 200.3, 200.27, 200.3, 200.3, 200.31, 200.23]
# multiagent_main_planning_time_opt = [77.0139, 83.7149, 76.4813, 44.7673, 2.55551, 2.43557, 69.5548, 1.90756, 0.119965, 56.5384, 115.586, 0.227936, 3.13537, 7.17092, 37.5771, 4.76725, 121.46, 12.1542, 16.3854, 7.39095]
# multiagent_main_cost = [29.0, 30.0, 29.0, 34.0, 42.0, 42.0, 42.0, 43.0, 53.0, 56.0, 56.0, 55.0, 74.0, 74.0, 77.0, 76.0, 85.0, 83.0, 87.0, 86.0]
# multiagent_main_planning_time_1st = [0.0119798, 0.00400154, 0.00799457, 0.00799323, 0.0319669, 0.0359799, 0.0199934, 0.0319673, 0.0239765, 0.0559866, 0.035968, 0.0479602, 0.083979, 0.051974, 0.103965, 0.107941, 0.107972, 0.0839728, 0.115961, 0.0359655]
# multiagent_main_cost_1st = [39, 35, 35, 40, 49, 53, 60, 67, 65, 71, 71, 64, 97, 111, 113, 98, 132, 110, 119, 113]
# # multiagent_main_success = [True, True, True, True, True, True, True, True, False, True, True, False, False, True, True, False, True, False, True, False]
# # overall_plan_length = [33, 34, 33, 34, 46, 46, 46, 47, -1, 60, 60, -1, -1, 78, 77, -1, 91, -1, 91, -1]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [34.0, 35.0, 34.0, 35.0, 47.0, 47.0, 47.0, 48.0, 64.0, 61.0, 61.0, 65.0, 84.0, 79.0, 78.0, 87.0, 92.0, 94.0, 92.0, 96.0]

# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [100.36, 50.19, 100.69, 48.65, 200.23, 200.22, 200.21, 200.2, 200.18, 200.25, 200.24, 200.24, 200.28, 200.26, 200.28, 200.27, 200.32, 200.28, 200.29, 200.29]
# singleagent_planning_time_opt = [100.141, 50.0235, 100.499, 48.4361, 13.4937, 13.6923, 121.169, 122.828, 4.91918, 2.61156, 9.59442, 2.36361, 20.2767, 27.3547, 50.747, 25.7566, 0.223948, 87.4247, 106.367, 80.8882]
# singleagent_cost = [36.0, 36.0, 36.0, 36.0, 49.0, 49.0, 49.0, 49.0, 65.0, 65.0, 65.0, 65.0, 82.0, 84.0, 83.0, 82.0, 126.0, 92.0, 95.0, 94.0]
# singleagent_planning_time_1st = [0.0239808, 0.00798246, 0.0159724, 0.00798956, 0.0399531, 0.0439715, 0.0239851, 0.0479676, 0.051974, 0.0679825, 0.0519763, 0.0559745, 0.155965, 0.0799671, 0.119953, 0.111942, 0.223948, 0.139969, 0.159949, 0.123967]
# singleagent_cost_1st = [48, 50, 48, 50, 67, 65, 56, 74, 74, 81, 82, 81, 107, 113, 106, 115, 126, 117, 134, 145]
# LLM_text_sg_time = [7.596452713012695, 8.010849475860596, 10.99940037727356, 6.170204162597656, 7.3040266036987305, 9.174655675888062, 10.045835733413696, 9.330740213394165, 9.357281923294067, 7.761087417602539, 7.132886648178101, 9.322663068771362, 7.112830638885498, -1, 5.936628103256226, 9.299728393554688, 10.480162858963013, 7.431478500366211, 9.157324075698853, 8.664772272109985]
# LLM_pddl_sg_time = [2.794018268585205, 3.3178551197052, 5.144523859024048, 4.359992504119873, 4.162924766540527, 4.468400478363037, 3.0479252338409424, 3.878159284591675, 2.8777377605438232, 3.980527639389038, 2.918633222579956, 3.8236756324768066, 3.7502598762512207, -1, 3.79736328125, 5.782184362411499, 5.353451251983643, 2.45793080329895, 6.169855117797852, 3.151958703994751]
# multiagent_helper_planning_time = [0.39, 3.83, 0.38, 0.35, 0.6, 0.6, 0.59, 0.6, 21.4, 0.94, 0.95, 8.16, 2.21, -1, 0.37, 1.81, 2.69, 51.86, 164.11, 161.75]
# multiagent_helper_planning_time_opt = [0.17595, 3.61529, 0.175943, 0.175939, 0.363899, 0.367891, 0.355902, 0.363893, 21.1327, 0.679456, 0.691834, 7.91076, 1.90763, -1, 0.0679593, 1.50375, 2.35164, 51.5289, 163.768, 161.485]
# multiagent_helper_cost = [7.0, 11.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 11.0, 7.0, 7.0, 10.0, 7.0, -1, 4.0, 7.0, 7.0, 10.0, 11.0, 11.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400401, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400174]
# multiagent_helper_cost_1st = [8, 13, 8, 8, 8, 8, 8, 8, 12, 8, 8, 11, 8, -1, 4, 8, 8, 11, 12, 12]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, 0, True, True, True, True, True, True]
# multiagent_main_planning_time = [80.49, 10.84, 77.82, 84.74, 200.21, 200.2, 200.2, 200.2, 200.23, 200.22, 200.23, 200.22, 200.29, -1, 200.27, 200.28, 200.27, 200.3, 200.29, 200.3]
# multiagent_main_planning_time_opt = [80.2768, 10.6458, 77.6074, 84.5136, 2.29559, 1.99556, 68.836, 1.7197, 0.11997, 56.5223, 115.036, 27.7353, 7.46267, -1, 36.3736, 6.0543, 122.213, 13.4099, 11.9623, 21.9387]
# multiagent_main_cost = [29.0, 28.0, 29.0, 30.0, 42.0, 42.0, 42.0, 43.0, 53.0, 56.0, 56.0, 55.0, 75.0, -1, 77.0, 75.0, 85.0, 82.0, 83.0, 84.0]
# multiagent_main_planning_time_1st = [0.0119938, 0.00398401, 0.0120109, 0.00798025, 0.023978, 0.015962, 0.01997, 0.0319752, 0.0239718, 0.0559515, 0.0399717, 0.0359666, 0.0839742, -1, 0.107949, 0.0799697, 0.107949, 0.111957, 0.0759671, 0.0559863]
# multiagent_main_cost_1st = [39, 31, 39, 40, 49, 52, 60, 67, 65, 71, 71, 72, 102, -1, 113, 90, 132, 109, 117, 120]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True]
# overall_plan_length = [34.0, 39.0, 34.0, 35.0, 47.0, 47.0, 47.0, 48.0, 64.0, 61.0, 61.0, 65.0, 80.0, 0.0, 78.0, 80.0, 92.0, 92.0, 94.0, 95.0]


# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [101.57, 49.15, 103.26, 48.13, 200.2, 200.24, 200.2, 200.18, 200.24, 200.23, 200.24, 200.24, 200.28, 200.28, 200.27, 200.29, 200.31, 200.29, 200.28, 200.21]
# singleagent_planning_time_opt = [101.361, 48.9144, 103.032, 47.9033, 13.0618, 13.1937, 121.199, 123.274, 4.56325, 2.70352, 9.95028, 2.93543, 20.5326, 27.0634, 48.3245, 25.5399, 0.159959, 86.7608, 106.909, 81.291]
# singleagent_cost = [36.0, 36.0, 36.0, 36.0, 49.0, 49.0, 49.0, 49.0, 65.0, 65.0, 65.0, 65.0, 82.0, 84.0, 83.0, 82.0, 126.0, 92.0, 95.0, 94.0]
# singleagent_planning_time_1st = [0.0199601, 0.00799093, 0.0199862, 0.0119554, 0.028382, 0.0439626, 0.0239794, 0.0479607, 0.0359803, 0.0719581, 0.0519878, 0.075973, 0.155956, 0.0799762, 0.0839735, 0.111968, 0.159959, 0.135972, 0.159897, 0.123967]
# singleagent_cost_1st = [48, 50, 48, 50, 67, 65, 56, 74, 74, 81, 82, 81, 107, 113, 106, 115, 126, 117, 134, 145]
# LLM_text_sg_time = [8.239243507385254, 8.297706365585327, 9.15216326713562, 6.283897161483765, 7.960748672485352, 8.147128820419312, 8.140034675598145, 8.791679859161377, 8.878613710403442, 8.397556066513062, 8.883707046508789, 10.025081157684326, 8.513997793197632, 7.969979286193848, 5.9445719718933105, 11.19732117652893, 7.744269132614136, 8.703559637069702, 10.05246090888977, 8.608340740203857]
# LLM_pddl_sg_time = [3.019118309020996, 2.8229894638061523, 3.8769218921661377, 2.590059280395508, 4.953908920288086, 3.536733865737915, 3.572675943374634, 2.6025354862213135, 3.036684989929199, 4.191822052001953, 3.7151100635528564, 3.5825302600860596, 2.9103238582611084, 5.064577579498291, 4.23256254196167, 3.6599345207214355, 3.540863275527954, 2.179868221282959, 3.0264554023742676, 2.678596258163452]
# multiagent_helper_planning_time = [0.29, 0.4, 0.39, 0.24, 0.62, 0.6, 3.73, 0.61, 8.12, 0.94, 0.91, 0.96, 1.78, 1.83, 0.32, 1.91, 2.55, 51.71, 50.31, 160.43]
# multiagent_helper_planning_time_opt = [0.13195, 0.175928, 0.179923, 0.00797668, 0.363911, 0.363883, 3.50736, 0.367887, 7.85872, 0.679835, 0.679855, 0.695831, 1.53575, 1.53577, 0.0159757, 1.5997, 2.28353, 51.3788, 50.0509, 160.08]
# multiagent_helper_cost = [7.0, 7.0, 7.0, 4.0, 7.0, 7.0, 10.0, 7.0, 10.0, 7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 7.0, 7.0, 10.0, 10.0, 11.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400623, 0.0, 0.0, 0.0, 0.00400342, 0.0, 0.0, 0.0, 0.0, 0.00400004, 0.0, 0.00399946, 0.0, 0.0]
# multiagent_helper_cost_1st = [8, 8, 8, 4, 8, 8, 11, 8, 11, 8, 8, 8, 8, 8, 3, 8, 8, 11, 11, 12]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [77.53, 74.7, 78.3, 45.61, 200.21, 200.22, 200.23, 200.21, 200.23, 200.24, 200.22, 200.22, 200.21, 200.27, -1, 200.27, 200.25, 200.29, 200.29, 200.29]
# multiagent_main_planning_time_opt = [77.3302, 74.4988, 78.0818, 45.4037, 2.78743, 2.51157, 3.63137, 1.98754, 0.0959706, 57.4619, 116.54, 59.514, 6.61086, 7.49464, -1, 7.91402, 179.74, 13.6496, 15.2173, 21.928]
# multiagent_main_cost = [29.0, 29.0, 29.0, 34.0, 42.0, 42.0, 41.0, 43.0, 54.0, 56.0, 56.0, 56.0, 75.0, 74.0, -1, 77.0, 86.0, 82.0, 84.0, 84.0]
# multiagent_main_planning_time_1st = [0.0119944, 0.00398577, 0.0120127, 0.00799328, 0.0319645, 0.0319651, 0.0159828, 0.0319738, 0.0239883, 0.05598, 0.0439151, 0.0399696, 0.0559892, 0.0519708, -1, 0.0959673, 0.0959568, 0.11195, 0.0639733, 0.0559773]
# multiagent_main_cost_1st = [39, 33, 39, 40, 49, 53, 55, 67, 70, 71, 71, 67, 102, 111, -1, 103, 133, 109, 110, 120]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True]
# overall_plan_length = [34.0, 34.0, 34.0, 35.0, 47.0, 47.0, 51.0, 48.0, 64.0, 61.0, 61.0, 61.0, 80.0, 79.0, 83, 82.0, 91.0, 92.0, 94.0, 95.0] ## problem

singleagent_planning_time = [102.34, 48.66, 100.8, 48.19, 1000.09, 1000.09, 859.72, 920.07, 1000.22, 1000.25, 1000.18, 996.1, 997.68, 1000.38, 1000.34, 1000.31, 1000.3, 1000.36, 1000.33, 1000.33]
singleagent_planning_time_opt = [102.126, 48.447, 100.578, 47.975, 808.913, 810.832, 859.413, 919.758, 330.02, 361.987, 434.188, 354.294, 20.1528, 26.9197, 47.8523, 26.1478, 207.865, 86.7226, 105.961, 80.7508]
singleagent_cost = [36.0, 36.0, 36.0, 36.0, 49.0, 49.0, 49.0, 49.0, 62.0, 63.0, 63.0, 62.0, 82.0, 84.0, 83.0, 82.0, 95.0, 92.0, 95.0, 94.0]
singleagent_planning_time_1st = [0.0199581, 0.0119936, 0.0199772, 0.0120084, 0.039967, 0.0399734, 0.02399, 0.0479757, 0.0519559, 0.0719768, 0.051977, 0.0759645, 0.155959, 0.0799717, 0.115946, 0.11196, 0.223956, 0.139967, 0.159962, 0.12397]
singleagent_cost_1st = [48, 50, 48, 50, 67, 65, 56, 74, 74, 81, 82, 81, 107, 113, 106, 115, 126, 117, 134, 145]
LLM_text_sg_time = [6.506386995315552, 8.988324403762817, 7.849247455596924, 5.508822917938232, 8.783982753753662, 7.988748550415039, 9.96993899345398, 6.227627277374268, 6.445740461349487, 6.441887855529785, 6.164518117904663, 7.390116930007935, 3.9207262992858887, 8.518510580062866, 6.7864298820495605, 7.406783580780029, 9.622092247009277, 11.51863980293274, 5.760431528091431, 9.107093334197998]
LLM_pddl_sg_time = [3.473114490509033, 2.5766658782958984, 2.918614625930786, 2.9533445835113525, 2.9384636878967285, 2.501476287841797, 2.063462257385254, 2.9137580394744873, 3.4680538177490234, 3.2677972316741943, 3.855285882949829, 3.3579776287078857, 3.898425340652466, 3.3700449466705322, 3.0515925884246826, 2.1129133701324463, 3.0591862201690674, 2.225048303604126, 3.346231460571289, 2.82928729057312]
multiagent_helper_planning_time = [0.44, 0.4, 0.22, 0.23, 0.5, 9.99, 29.56, 0.61, 0.94, 0.93, 0.95, 0.95, 1.97, 2.16, 1.81, 0.38, 164.11, 165.77, 2.63, 161.19]
multiagent_helper_planning_time_opt = [0.22391, 0.179932, 0.00400352, 0.0159698, 0.26393, 9.76215, 29.3297, 0.367905, 0.687823, 0.67984, 0.687819, 0.687805, 1.66368, 1.86763, 1.51172, 0.071966, 163.768, 165.434, 2.29965, 160.857]
multiagent_helper_cost = [7.0, 7.0, 3.0, 4.0, 7.0, 11.0, 13.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 4.0, 11.0, 11.0, 7.0, 11.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.00400015, 0.00399996, 0.00799512, 0.00400721, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0359768, 0.0, 0.0, 0.00400063]
multiagent_helper_cost_1st = [8, 8, 3, 4, 8, 30, 14, 8, 8, 8, 8, 8, 8, 8, 8, 4, 30, 12, 8, 12]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [80.7, 77.27, -1, 44.06, 1000.06, 198.81, 101.36, 1000.05, 1000.29, 1000.2, 1000.21, 1000.21, 1000.27, 1000.25, 1000.23, 1000.28, 1000.26, 1000.28, 1000.31, 1000.19]
# multiagent_main_planning_time_opt = [80.5041, 77.0561, -1, 43.8515, 792.727, 198.637, 101.132, 810.351, 0.22394, 57.8252, 119.037, 57.188, 6.57485, 943.886, 13.9655, 11.074, 50.9808, 13.2818, 31.811, 21.8799]
# multiagent_main_cost = [29.0, 29.0, -1, 34.0, 42.0, 40.0, 36.0, 43.0, 55.0, 56.0, 56.0, 56.0, 75.0, 70.0, 77.0, 80.0, 86.0, 82.0, 89.0, 84.0]
# multiagent_main_planning_time_1st = [0.0119949, 0.00798698, -1, 0.00798019, 0.0199724, 0.0119858, 0.0115907, 0.0319682, 0.0319727, 0.0559788, 0.03998, 0.0519808, 0.0599685, 0.0559748, 0.0599763, 0.131964, 0.163961, 0.0999699, 0.119967, 0.0559894]
# multiagent_main_cost_1st = [39, 33, -1, 40, 52, 48, 51, 67, 63, 71, 71, 76, 102, 111, 98, 111, 113, 103, 144, 120]
multiagent_main_planning_time = [80.7, 77.27, 100.8, 44.06, 1000.06, 198.81, 101.36, 1000.05, 1000.29, 1000.2, 1000.21, 1000.21, 1000.27, 1000.25, 1000.23, 1000.28, 1000.26, 1000.28, 1000.31, 1000.19]
multiagent_main_planning_time_opt = [80.5041, 77.0561, 100.578, 43.8515, 792.727, 198.637, 101.132, 810.351, 0.22394, 57.8252, 119.037, 57.188, 6.57485, 943.886, 13.9655, 11.074, 50.9808, 13.2818, 31.811, 21.8799]
multiagent_main_cost = [29.0, 29.0, 36.0, 34.0, 42.0, 40.0, 36.0, 43.0, 55.0, 56.0, 56.0, 56.0, 75.0, 70.0, 77.0, 80.0, 86.0, 82.0, 89.0, 84.0]
multiagent_main_planning_time_1st = [0.0119949, 0.00798698, 0.0199772, 0.00798019, 0.0199724, 0.0119858, 0.0115907, 0.0319682, 0.0319727, 0.0559788, 0.03998, 0.0519808, 0.0599685, 0.0559748, 0.0599763, 0.131964, 0.163961, 0.0999699, 0.119967, 0.0559894]
multiagent_main_cost_1st = [39, 33, 48, 40, 52, 48, 51, 67, 63, 71, 71, 76, 102, 111, 98, 111, 113, 103, 144, 120]
multiagent_main_success = [True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [34.0, 34.0, 36.0, 35.0, 47.0, 51.0, 46.0, 48.0, 60.0, 61.0, 61.0, 61.0, 80.0, 75.0, 82.0, 81.0, 97.0, 93.0, 94.0, 95.0]


singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))

plt.plot(singleagent_planning_time, '-r') #, figure=fig_barman_time)
# plt.plot(singleagent_cost, '-r') #, figure=fig_barman_length)
plt.plot(multiagent_total_planning_time, '-g') #, figure=fig_barman_time)
# plt.plot(overall_plan_length, '-g') #, figure=fig_barman_length)

print('blocksworld')
singleagent = []
helper = []
main = []

# blocksworld
# singleagent_planning_time = [0.18, 0.14, 0.14, 0.13, 0.13, 0.15, 0.15, 0.19, 1.09, 1.41, 11.22, 12.58, 149.44, 72.03, 200.15, 200.18, 200.19, 200.17, 200.18, 200.2]
# singleagent_planning_time_opt = [0.00397713, 0.00400058, 0.00399994, 0.00800655, 0.00798315, 0.0160104, 0.0159515, 0.0439606, 0.955799, 1.26373, 11.0946, 12.4337, 149.263, 71.8674, 0.103955, 0.00797469, 0.00399981, 0.451908, 0.0759739, 0.0239934]
# singleagent_cost = [6.0, 6.0, 6.0, 12.0, 8.0, 12.0, 8.0, 14.0, 14.0, 18.0, 22.0, 20.0, 26.0, 22.0, 26.0, 28.0, 22.0, 30.0, 24.0, 30.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400001, 0.00400003, 0.0039993, 0.0, 0.00399952, 0.00400352, 0.00400414, 0.00398052, 0.00397885, 0.00400369, 0.00400365]
# singleagent_cost_1st = [6, 6, 6, 16, 8, 12, 8, 16, 18, 32, 22, 40, 32, 48, 36, 38, 40, 32, 28, 42]
# LLM_text_sg_time = [6.3164427280426025, 5.262510061264038, 4.401891469955444, 4.785141944885254, 6.492253065109253, 6.358168363571167, 6.504984378814697, 8.90913701057434, 4.86905312538147, 3.888023853302002, 5.81528377532959, 5.207462549209595, 6.302455186843872, 6.847730875015259, 5.9709532260894775, 5.399209976196289, 7.209357738494873, 6.368146181106567, 6.3678224086761475, 6.335895776748657]
# LLM_pddl_sg_time = [1.7463059425354004, 2.0477211475372314, 2.0085651874542236, 1.9283103942871094, 2.6064529418945312, 1.550365686416626, 1.6154544353485107, 2.586115837097168, 1.8714239597320557, 1.7371306419372559, 3.5265254974365234, 1.628971815109253, 1.7497236728668213, 3.830101251602173, 2.7503416538238525, 1.7319121360778809, 1.8133702278137207, 2.112332820892334, 1.9149878025054932, 3.443422317504883]
# multiagent_helper_planning_time = [0.13, 0.14, 0.14, 0.15, 0.13, 0.14, 0.16, 0.15, 0.19, 0.15, 0.15, 0.2, 0.16, 0.17, 0.24, 0.19, 6.75, 2.06, 0.17, 0.28]
# multiagent_helper_planning_time_opt = [0.0, 0.0, 0.00401136, 0.00399957, 0.00400003, 0.00399967, 0.00397024, 0.00400009, 0.0319599, 0.0, 0.00800289, 0.0479639, 0.00400094, 0.0119925, 0.0719529, 0.0359643, 6.5827, 1.89166, 0.0040002, 0.107937]
# multiagent_helper_cost = [2.0, 2.0, 2.0, 6.0, 6.0, 8.0, 4.0, 10.0, 8.0, 2.0, 10.0, 8.0, 2.0, 6.0, 14.0, 10.0, 14.0, 16.0, 2.0, 8.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0040013, 0.00401524, 0.00400048, 0.00397318, 0.0, 0.0]
# multiagent_helper_cost_1st = [2, 2, 2, 6, 6, 8, 4, 10, 8, 2, 10, 8, 2, 6, 14, 10, 14, 16, 2, 8]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [0.13, 0.13, 0.13, 0.14, 0.13, 0.16, 0.14, 0.14, 2.13, 1.4, 7.12, 14.06, 149.13, 95.2, 200.17, 200.16, 200.12, 200.18, 200.18, 200.23]
# multiagent_main_planning_time_opt = [0.00399996, 0.00400014, 0.00399926, 0.00800496, 0.0, 0.0159872, 0.00399873, 0.00399984, 1.9875, 1.25577, 6.96273, 13.9134, 148.966, 95.0398, 0.0, 0.0159887, 145.806, 0.183942, 0.411929, 0.675897]
# multiagent_main_cost = [4.0, 4.0, 4.0, 6.0, 2.0, 6.0, 4.0, 4.0, 14.0, 16.0, 12.0, 18.0, 28.0, 22.0, 14.0, 18.0, 12.0, 20.0, 24.0, 26.0]
# multiagent_main_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00397944, 0.0, 0.00399938, 0.0, 0.0, 0.00400315, 0.0]
# multiagent_main_cost_1st = [4, 4, 4, 10, 2, 6, 4, 4, 18, 30, 12, 34, 34, 40, 22, 24, 16, 24, 28, 36]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [5.0, 5.0, 5.0, 11.0, 6.0, 13.0, 7.0, 12.0, 18.0, 16.0, 19.0, 21.0, 30.0, 22.0, 24.0, 23.0, 18.0, 36.0, 24.0, 34.0]


# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.14, 0.13, 0.13, 0.13, 0.14, 0.15, 0.15, 0.19, 1.38, 1.65, 11.65, 13.48, 152.04, 75.2, 200.12, 200.17, 200.21, 200.17, 200.19, 200.23]
# singleagent_planning_time_opt = [0.00400006, 0.00399998, 0.0040002, 0.0080083, 0.00399998, 0.0159596, 0.0159895, 0.0439628, 1.24372, 1.49569, 11.4938, 13.3154, 151.863, 75.0469, 0.0719681, 0.00400203, 0.00399305, 0.455886, 0.0759798, 0.0239923]
# singleagent_cost = [6.0, 6.0, 6.0, 12.0, 8.0, 12.0, 8.0, 14.0, 14.0, 18.0, 22.0, 20.0, 26.0, 22.0, 26.0, 28.0, 22.0, 30.0, 24.0, 30.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.00399967, 0.0, 0.0, 0.00400001, 0.00399982, 0.00400113, 0.0, 0.00399986, 0.00400008, 0.0, 0.00401007, 0.00400031, 0.0, 0.0, 0.00400405]
# singleagent_cost_1st = [6, 6, 6, 16, 8, 12, 8, 16, 18, 32, 22, 40, 32, 48, 36, 38, 40, 32, 28, 42]
# LLM_text_sg_time = [5.531355142593384, 5.5118865966796875, 7.988949298858643, 6.165555715560913, 6.614049673080444, 7.586973428726196, 5.389688968658447, 4.5446882247924805, 4.412238359451294, 5.831475257873535, 6.567452669143677, 5.059210777282715, 6.036228895187378, 5.823169469833374, 7.283507823944092, 6.615394592285156, 6.389691114425659, 4.9752466678619385, 5.857750654220581, 5.876104116439819]
# LLM_pddl_sg_time = [1.626655101776123, 2.16660213470459, 1.5664198398590088, 2.3038923740386963, 2.285949945449829, 2.2242634296417236, 1.6741008758544922, 2.4772982597351074, 1.778665542602539, 4.5722739696502686, 2.399651527404785, 2.3750853538513184, 2.338705539703369, 2.4512555599212646, 1.8209965229034424, 1.92852783203125, 2.7487313747406006, 2.135610818862915, 1.7245628833770752, 2.0728046894073486]
# multiagent_helper_planning_time = [0.13, 0.14, 0.13, 0.13, 0.12, 0.15, 0.14, 0.15, 0.17, 0.19, 0.23, 0.19, 1.74, 0.17, 0.18, 0.2, 1.41, 2.02, 0.19, 0.27]
# multiagent_helper_planning_time_opt = [0.0, 0.00400014, 0.00397868, 0.00397817, 0.00397668, 0.0040002, 0.00399922, 0.0080022, 0.0439727, 0.0399682, 0.0679752, 0.0479666, 1.58766, 0.0119716, 0.0080058, 0.0319816, 1.24377, 1.84767, 0.00400065, 0.103964]
# multiagent_helper_cost = [2.0, 2.0, 2.0, 6.0, 6.0, 10.0, 4.0, 10.0, 8.0, 8.0, 14.0, 8.0, 16.0, 6.0, 10.0, 10.0, 12.0, 16.0, 1.0, 8.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.00397668, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00398324, 0.0, 0.00400013, 0.0, 0.0, 0.0, 0.0]
# multiagent_helper_cost_1st = [2, 2, 2, 6, 6, 10, 4, 10, 8, 8, 14, 8, 16, 6, 10, 10, 12, 16, 1, 8]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [0.13, 0.13, 0.13, 0.13, 0.16, 0.14, 0.15, 0.15, 2.41, 2.43, 2.45, 14.78, 78.46, 97.52, 70.04, 200.16, 200.18, 200.16, -1, 200.19]
# multiagent_main_planning_time_opt = [0.0, 0.00397668, 0.00402015, 0.00800314, 0.0040088, 0.00797725, 0.0040166, 0.00399855, 2.26354, 2.29155, 2.30746, 14.6252, 78.3136, 97.3484, 69.8752, 0.011987, 0.0239949, 0.123944, -1, 0.679884]
# multiagent_main_cost = [4.0, 4.0, 4.0, 6.0, 2.0, 4.0, 4.0, 4.0, 14.0, 14.0, 10.0, 18.0, 16.0, 22.0, 16.0, 18.0, 14.0, 20.0, -1, 26.0]
# multiagent_main_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0040088, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400003, 0.00400012, 0.0, 0.0, 0.0, 0.0, -1, 0.004009]
# multiagent_main_cost_1st = [4, 4, 4, 10, 2, 4, 4, 4, 18, 26, 22, 34, 24, 40, 24, 24, 18, 24, -1, 36]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True]
# overall_plan_length = [5.0, 5.0, 5.0, 11.0, 6.0, 10.0, 7.0, 12.0, 18.0, 22.0, 22.0, 21.0, 29.0, 22.0, 25.0, 23.0, 25.0, 36.0, 24, 34.0] ## problem

# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.13, 0.14, 0.13, 0.13, 0.14, 0.15, 0.16, 0.2, 1.37, 1.49, 11.21, 12.35, 150.26, 72.78, 200.18, 200.18, 200.2, 200.17, 200.17, 200.2]
# singleagent_planning_time_opt = [0.00399951, 0.00399988, 0.00400522, 0.00796767, 0.00398402, 0.0120073, 0.0159751, 0.0439677, 1.2277, 1.34369, 11.0702, 12.2058, 150.099, 72.5958, 0.103963, 0.00400213, 0.0, 0.451887, 0.0759756, 0.0239922]
# singleagent_cost = [6.0, 6.0, 6.0, 12.0, 8.0, 12.0, 8.0, 14.0, 14.0, 18.0, 22.0, 20.0, 26.0, 22.0, 26.0, 28.0, 22.0, 30.0, 24.0, 30.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.00398402, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399979, 0.0, 0.00400449, 0.0, 0.00400396]
# singleagent_cost_1st = [6, 6, 6, 16, 8, 12, 8, 16, 18, 32, 22, 40, 32, 48, 36, 38, 40, 32, 28, 42]
# LLM_text_sg_time = [5.526350975036621, 4.275106430053711, 5.920973777770996, 5.286208391189575, 5.990195035934448, 4.746309757232666, 4.603266954421997, 7.491765260696411, 5.671319484710693, 6.186772108078003, 5.5711259841918945, 4.718376398086548, 6.73906683921814, 5.861374616622925, 7.19300103187561, 5.7276647090911865, 5.340129137039185, 6.590932846069336, 7.427927017211914, 5.624540567398071]
# LLM_pddl_sg_time = [2.8352839946746826, 2.0374436378479004, 2.2863292694091797, 2.8328940868377686, 1.7685141563415527, 2.1727383136749268, 1.8322348594665527, 1.761584997177124, 1.6618216037750244, 1.8193330764770508, 1.9822163581848145, 4.44440484046936, 2.561349630355835, 1.7086846828460693, 2.3155109882354736, 1.8568859100341797, 2.243821382522583, 2.0284345149993896, 2.1453280448913574, 2.8407461643218994]
# multiagent_helper_planning_time = [0.13, 0.14, 0.14, 0.14, 0.13, 0.14, 0.16, 0.16, 0.15, 0.13, 0.16, 0.21, 0.17, 0.17, -1, 0.21, 6.75, 2.01, 0.18, 0.28]
# multiagent_helper_planning_time_opt = [0.0, 0.00400042, 0.00397328, 0.00397456, 0.0, 0.00399967, 0.011993, 0.00399872, 0.00400013, 0.0119917, 0.00800239, 0.0479633, 0.00401574, 0.0119857, -1, 0.0319651, 6.59477, 1.84767, 0.00800365, 0.107947]
# multiagent_helper_cost = [2.0, 2.0, 4.0, 6.0, 6.0, 8.0, 8.0, 10.0, 2.0, 6.0, 10.0, 8.0, 2.0, 6.0, -1, 10.0, 14.0, 16.0, 4.0, 8.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0039996, 0.0, 0.00400168]
# multiagent_helper_cost_1st = [2, 2, 4, 6, 6, 8, 8, 10, 2, 6, 10, 8, 2, 6, -1, 10, 14, 16, 4, 8]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [0.13, 0.13, 0.14, 0.14, 0.13, 0.14, 0.14, 0.15, 1.99, 2.07, 7.18, 13.78, 152.9, 95.26, 200.17, 200.16, 200.12, 200.16, 200.16, 200.22]
# multiagent_main_planning_time_opt = [0.00400017, 0.00397786, 0.0, 0.00399953, 0.00400431, 0.0119929, 0.00398365, 0.00399882, 1.83565, 1.92163, 7.02273, 13.6337, 152.727, 95.0937, 0.103963, 0.0119762, 144.886, 0.17595, 0.00399986, 0.679886]
# multiagent_main_cost = [4.0, 4.0, 2.0, 6.0, 2.0, 6.0, 4.0, 4.0, 16.0, 16.0, 12.0, 18.0, 28.0, 22.0, 26.0, 18.0, 12.0, 20.0, 24.0, 26.0]
# multiagent_main_planning_time_1st = [0.0, 0.00397786, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399992, 0.00399994, 0.00399911, 0.00399911, 0.0, 0.00401255, 0.0, 0.00399986, 0.00399991]
# multiagent_main_cost_1st = [4, 4, 2, 10, 2, 6, 4, 4, 16, 28, 12, 34, 34, 40, 36, 24, 16, 24, 24, 36]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [5.0, 5.0, 5.0, 11.0, 6.0, 13.0, 10.0, 12.0, 16.0, 20.0, 19.0, 21.0, 30.0, 22.0, 26, 23.0, 18.0, 36.0, 24.0, 34.0]
singleagent_planning_time = [0.14, 0.14, 0.14, 0.15, 0.13, 0.15, 0.16, 0.2, 1.48, 1.59, 11.71, 12.81, 151.3, 74.37, 224.63, 984.0, 1000.27, 1000.3, 1000.36, 1000.36]
singleagent_planning_time_opt = [0.00397906, 0.00399981, 0.00399978, 0.00797919, 0.00399987, 0.0120087, 0.0159788, 0.0399835, 1.34372, 1.44368, 11.5456, 12.6612, 151.145, 74.1918, 224.43, 983.726, 0.0, 0.447886, 0.0759779, 0.0239764]
singleagent_cost = [6.0, 6.0, 6.0, 12.0, 8.0, 12.0, 8.0, 14.0, 14.0, 18.0, 22.0, 20.0, 26.0, 22.0, 26.0, 28.0, 22.0, 30.0, 24.0, 30.0]
singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.00397864, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399081, 0.00399991, 0.0, 0.0, 0.00399989, 0.0039733, 0.0, 0.00397596]
singleagent_cost_1st = [6, 6, 6, 16, 8, 12, 8, 16, 18, 32, 22, 40, 32, 48, 36, 38, 40, 32, 28, 42]


LLM_text_sg_time = [4.779265403747559, 4.486076831817627, 3.1258010864257812, 3.6369874477386475, 7.049670457839966, 4.364583730697632, 5.893851041793823, 9.80495023727417, 5.832620143890381, -1, 5.894093751907349, 5.71201229095459, 2.498518228530884, 5.075425386428833, 5.174367904663086, 5.5404839515686035, 6.390017032623291, 6.561753511428833, 4.812869548797607, 6.599436283111572]
LLM_pddl_sg_time = [2.008441925048828, 2.0904102325439453, 2.5366055965423584, 1.8850414752960205, 2.255082368850708, 2.8295750617980957, 1.4210257530212402, 1.8184523582458496, 2.0170235633850098, -1, 2.2354979515075684, 1.9861860275268555, 2.0233280658721924, 1.629408836364746, 1.0452215671539307, 1.7455756664276123, 1.5297391414642334, 1.7767715454101562, 1.7896525859832764, 1.6766393184661865]
multiagent_helper_planning_time = [0.14, 0.13, 0.13, 0.14, 0.15, 0.14, 0.14, 0.15, 0.14, -1, 0.17, 0.19, 0.15, 0.13, -1, 0.18, 0.42, 0.58, -1, 0.25]
multiagent_helper_planning_time_opt = [0.00399078, 0.0, 0.00399942, 0.0, 0.00400027, 0.00397945, 0.00399943, 0.00800312, 0.00400044, -1, 0.0279663, 0.0439663, 0.00399997, 0.004, -1, 0.031969, 0.251916, 0.411901, -1, 0.0799657]
multiagent_helper_cost = [2.0, 2.0, 2.0, 6.0, 6.0, 8.0, 4.0, 10.0, 2.0, -1, 12.0, 8.0, 2.0, 4.0, -1, 10.0, 10.0, 14.0, -1, 8.0]
multiagent_helper_planning_time_1st = [0.00399078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.0, -1, 0.00399852, 0.0, 0.0, -1, 0.0]
multiagent_helper_cost_1st = [2, 2, 2, 6, 6, 8, 4, 10, 2, -1, 12, 8, 2, 4, -1, 10, 10, 14, -1, 8]
multiagent_helper_success = [True, True, True, True, True, True, True, True, True, 0, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.13, 0.12, 0.13, 0.14, 0.13, 0.16, 0.14, 0.14, 2.39, 2.46, 2.8, 14.77, 123.98, 91.39, 228.92, 1000.02, 1000.19, 1000.22, 1000.22, 1000.25]
multiagent_main_planning_time_opt = [0.0, 0.00399929, 0.00399954, 0.00798115, 0.00400014, 0.0239689, 0.0079807, 0.00397946, 2.2313, 2.31936, 2.66344, 14.6196, 123.825, 91.2296, 228.709, 775.833, 0.0679484, 0.00400377, 0.0759791, 0.671833]
multiagent_main_cost = [4.0, 4.0, 4.0, 6.0, 2.0, 6.0, 4.0, 4.0, 16.0, 16.0, 10.0, 18.0, 24.0, 22.0, 26.0, 18.0, 16.0, 22.0, 24.0, 26.0]
multiagent_main_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00398267, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399986, 0.00400473, 0.00399948]
multiagent_main_cost_1st = [4, 4, 4, 10, 2, 6, 4, 4, 16, 28, 18, 34, 30, 44, 36, 24, 20, 38, 28, 36]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [5.0, 5.0, 5.0, 11.0, 6.0, 13.0, 7.0, 12.0, 16.0, 1000000.0, 19.0, 21.0, 25.0, 26.0, 26, 23.0, 22.0, 35.0, 24, 34.0]

singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))

# # plt.plot(singleagent_planning_time, '-r') #, figure=fig_barman_time)
# plt.plot(singleagent_cost, '-r') #, figure=fig_barman_length)
# # plt.plot(multiagent_total_planning_time, '-g') #, figure=fig_barman_time)
# plt.plot(overall_plan_length, '-g') #, figure=fig_barman_length)

print('termes')
singleagent = []
helper = []
main = []
# termes
# singleagent_planning_time = [26.27, 184.68, 200.19, 200.18, 200.15, 200.2, 200.24, 200.24, 200.25, 200.25, 200.17, 74.14, 200.17, 200.18, 200.17, 200.2, 200.16, 200.18, 200.18, 200.18]
# singleagent_planning_time_opt = [26.0271, 184.462, 1.32375, 31.8662, 103.652, 147.406, 20.6325, 5.56704, 74.2997, 23.2517, 1.42769, 73.9818, 63.4288, 46.424, 132.612, 41.7586, 48.803, 12.4497, 28.9865, 109.376]
# singleagent_cost = [36.0, 54.0, 68.0, 80.0, 144.0, 172.0, 56.0, 66.0, 112.0, 100.0, 66.0, 46.0, 92.0, 108.0, 148.0, 164.0, 116.0, 76.0, 94.0, 112.0]
# singleagent_planning_time_1st = [0.0119765, 0.175942, 0.175948, 0.187941, 12.3697, 3.76737, 0.143946, 0.0440838, 4.7872, 3.00752, 0.0239775, 0.0319928, 0.143954, 0.091952, 2.20758, 1.03197, 0.35593, 0.0239772, 0.403901, 0.515874]
# singleagent_cost_1st = [66, 108, 138, 270, 258, 224, 152, 104, 298, 220, 142, 150, 202, 274, 310, 382, 292, 134, 170, 266]
# LLM_text_sg_time = [9.279735565185547, 8.382715225219727, 6.037511110305786, 6.528009414672852, 8.756736755371094, 7.2720947265625, 6.6878273487091064, 7.011935234069824, 7.781051397323608, 6.948996543884277, 6.139204263687134, 8.439755916595459, 8.165164232254028, 8.70543646812439, 9.065320491790771, 9.746226787567139, 6.601127624511719, 7.735633134841919, 6.31978178024292, 6.098556280136108]
# LLM_pddl_sg_time = [12.421375513076782, 10.331300497055054, 3.4618940353393555, 10.128491163253784, 10.828545093536377, 11.674389362335205, 3.4894962310791016, 2.8923606872558594, 3.1428701877593994, 2.540620803833008, 10.569710493087769, 12.368244886398315, 12.967143297195435, 2.960798740386963, 4.065953254699707, 10.36972689628601, 2.5414974689483643, 3.6154446601867676, 2.940749168395996, 2.4912521839141846]
# multiagent_helper_planning_time = [0.19, 5.45, 0.22, 200.19, 200.2, 200.19, 0.27, 0.27, 0.3, 200.27, 11.55, 17.3, 28.7, 5.17, 1.99, 200.21, 6.72, 0.46, 0.22, 0.39]
# multiagent_helper_planning_time_opt = [0.00800741, 5.25103, 0.00400007, 6.60681, 33.2661, 103.712, 0.00796517, 0.0, 0.00400273, 14.6973, 11.3498, 17.0969, 28.485, 4.95507, 1.75164, 62.6762, 6.52683, 0.263908, 0.00400008, 0.167923]
# multiagent_helper_cost = [4.0, 32.0, 2.0, 80.0, 134.0, 144.0, 4.0, 2.0, 2.0, 52.0, 36.0, 36.0, 36.0, 26.0, 22.0, 130.0, 33.0, 15.0, 2.0, 14.0]
# multiagent_helper_planning_time_1st = [0.0, 0.00799332, 0.0, 0.415906, 1.6397, 12.6412, 0.0, 0.0, 0.0, 1.22776, 0.011994, 0.0159947, 0.0079746, 0.103946, 0.0159854, 7.63864, 0.13596, 0.00400064, 0.0, 0.0]
# multiagent_helper_cost_1st = [4, 88, 2, 188, 218, 258, 4, 2, 2, 200, 64, 56, 66, 162, 62, 184, 117, 19, 2, 18]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [15.68, 13.53, 200.18, 3.53, -1, -1, 200.23, 200.27, 200.26, 200.26, -1, -1, 200.18, 200.17, 200.2, -1, 200.11, 200.16, 200.17, 200.17]
# multiagent_main_planning_time_opt = [15.4772, 13.3576, 1.45172, 3.31135, -1, -1, 19.9683, 4.90709, 75.0344, 0.375942, -1, -1, 3.7552, 146.611, 122.293, -1, 96.5273, 5.86282, 56.0936, 105.905]
# multiagent_main_cost = [30.0, 32.0, 66.0, 32.0, -1, -1, 52.0, 64.0, 110.0, 30.0, -1, -1, 72.0, 86.0, 142.0, -1, 84.0, 62.0, 92.0, 100.0]
# multiagent_main_planning_time_1st = [0.0119922, 0.0879708, 0.175954, 0.0159724, -1, -1, 0.175943, 0.039966, 4.91894, 0.0359749, -1, -1, 0.119956, 0.0999519, 1.92764, -1, 0.303924, 0.0239655, 0.411912, 0.447905]
# multiagent_main_cost_1st = [58, 44, 148, 86, -1, -1, 152, 104, 304, 94, -1, -1, 120, 258, 280, -1, 176, 114, 214, 304]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [30.0, 57.0, 66.0, 110.0, 311.0, 300.0, 55.0, 64.0, 110.0, 78.0, 82.0, 66.0, 93.0, 110.0, 159.0, 252.0, 100.0, 72.0, 92.0, 100.0]



# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)
# singleagent_planning_time = [26.56, 186.16, 200.18, 200.17, 200.21, 200.2, 200.21, 200.23, 200.25, 200.28, 200.16, 74.99, 200.13, 200.17, 200.21, 200.19, 200.15, 200.17, 200.19, 200.17]
# singleagent_planning_time_opt = [26.3946, 185.954, 1.28776, 32.526, 105.917, 147.438, 20.9961, 5.27908, 74.9922, 23.4399, 1.42771, 74.7976, 62.8604, 46.095, 132.096, 41.2043, 48.9466, 12.3618, 29.0731, 107.992]
# singleagent_cost = [36.0, 54.0, 68.0, 80.0, 144.0, 172.0, 56.0, 66.0, 112.0, 100.0, 66.0, 46.0, 92.0, 108.0, 148.0, 164.0, 116.0, 76.0, 94.0, 112.0]
# singleagent_planning_time_1st = [0.0119776, 0.171944, 0.171952, 0.135955, 12.2059, 4.16727, 0.14393, 0.0439679, 5.1672, 3.0755, 0.0239869, 0.0479666, 0.103963, 0.0919464, 1.7077, 1.17176, 0.499877, 0.0239821, 0.411907, 0.52386]
# singleagent_cost_1st = [66, 108, 138, 270, 258, 224, 152, 104, 298, 220, 142, 150, 202, 274, 310, 382, 292, 134, 170, 266]
# LLM_text_sg_time = [6.518463373184204, 6.864581346511841, 7.534044504165649, 9.009793043136597, 7.934242248535156, 7.784206390380859, 8.242501974105835, 7.010367155075073, 8.913845539093018, 7.532845735549927, 5.540596961975098, 7.716474294662476, 8.373908758163452, 7.687234401702881, 7.053053855895996, 6.587680816650391, 7.459374666213989, 5.925205230712891, 8.977534294128418, 7.4447996616363525]
# LLM_pddl_sg_time = [11.835987329483032, 14.545403718948364, 9.57753300666809, 11.291906595230103, 11.475646018981934, 2.7420217990875244, 2.5103421211242676, 14.380563020706177, 17.474737882614136, 19.15825891494751, 9.885596990585327, 10.080664873123169, 12.532721996307373, 2.3783135414123535, 11.327727794647217, 10.120661735534668, 3.262220621109009, 11.277963876724243, 2.94474196434021, 3.2682671546936035]
# multiagent_helper_planning_time = [26.6, 14.42, 200.19, 200.19, 200.2, 0.23, 0.27, 200.2, 200.26, 0.23, 11.39, 17.43, 28.58, 191.87, 5.69, 200.2, 5.55, 42.59, 66.95, 4.12]
# multiagent_helper_planning_time_opt = [26.3951, 14.2133, 1.31575, 6.61883, 33.4104, 0.00400449, 0.00393475, 97.7678, 75.6904, 0.0, 11.1856, 17.2288, 28.3672, 191.646, 5.45103, 62.6573, 5.34697, 42.396, 66.7407, 3.90728]
# multiagent_helper_cost = [36.0, 32.0, 68.0, 80.0, 134.0, 2.0, 3.0, 32.0, 112.0, 2.0, 36.0, 36.0, 36.0, 53.0, 32.0, 130.0, 36.0, 42.0, 46.0, 30.0]
# multiagent_helper_planning_time_1st = [0.0119939, 0.00400038, 0.171945, 0.415902, 1.73354, 0.0, 0.0, 0.0519252, 4.94318, 0.0, 0.00799468, 0.0159742, 0.00400356, 0.191906, 0.01198, 7.65875, 0.0, 0.027979, 0.191947, 0.00400378]
# multiagent_helper_cost_1st = [66, 46, 138, 188, 218, 2, 3, 88, 298, 2, 64, 56, 66, 119, 88, 184, 42, 96, 184, 36]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [-1, 13.85, -1, 3.53, -1, 200.21, 200.21, -1, -1, 200.26, -1, -1, 200.17, 200.18, 200.19, -1, 200.13, 200.17, 200.18, 200.19]
# multiagent_main_planning_time_opt = [-1, 13.6534, -1, 3.30735, -1, 158.12, 20.2201, -1, -1, 16.3317, -1, -1, 2.9435, 10.2583, 93.8068, -1, 108.325, 0.863834, 0.827852, 59.4644]
# multiagent_main_cost = [-1, 32.0, -1, 32.0, -1, 170.0, 52.0, -1, -1, 88.0, -1, -1, 72.0, 62.0, 148.0, -1, 84.0, 50.0, 54.0, 88.0]
# multiagent_main_planning_time_1st = [-1, 0.123957, -1, 0.0119946, -1, 3.63537, 0.203882, -1, -1, 3.67882, -1, -1, 0.107964, 0.0279836, 1.86366, -1, 0.411906, 0.023981, 0.379912, 0.439897]
# multiagent_main_cost_1st = [-1, 44, -1, 86, -1, 224, 152, -1, -1, 212, -1, -1, 120, 156, 300, -1, 302, 96, 162, 256]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [36.0, 57.0, 68.0, 110.0, 311.0, 170.0, 54.0, 88.0, 112.0, 89.0, 82.0, 66.0, 93.0, 99.0, 173.0, 252.0, 109.0, 85.0, 91.0, 100.0]


# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)
# singleagent_planning_time = [26.63, 186.03, 200.18, 200.17, 200.24, 200.2, 200.21, 200.23, 200.19, 200.25, 200.16, 73.37, 200.17, 200.14, 200.2, 200.21, 200.17, 200.17, 200.19, 200.13]
# singleagent_planning_time_opt = [26.435, 185.824, 1.43972, 33.1416, 104.936, 147.227, 20.8483, 5.48694, 75.6031, 23.3558, 1.15578, 73.1747, 62.6766, 45.8803, 131.353, 41.2531, 48.5428, 12.3376, 28.6148, 107.095]
# singleagent_cost = [36.0, 54.0, 68.0, 80.0, 144.0, 172.0, 56.0, 66.0, 112.0, 100.0, 66.0, 46.0, 92.0, 108.0, 148.0, 164.0, 116.0, 76.0, 94.0, 112.0]
# singleagent_planning_time_1st = [0.0119928, 0.171945, 0.175942, 0.18393, 12.7251, 3.9113, 0.143965, 0.0399786, 4.61504, 3.35142, 0.0199793, 0.0439655, 0.14394, 0.0679694, 2.33956, 1.17178, 0.503893, 0.0239806, 0.403897, 0.519881]
# singleagent_cost_1st = [66, 108, 138, 270, 258, 224, 152, 104, 298, 220, 142, 150, 202, 274, 310, 382, 292, 134, 170, 266]
# LLM_text_sg_time = [5.844706058502197, 6.521281003952026, 5.572528123855591, 10.266708374023438, 8.00331163406372, 6.598856687545776, 8.649260759353638, 6.016380786895752, 5.946560382843018, 7.040308952331543, 6.582411289215088, 9.1865873336792, 7.57603907585144, 9.160937547683716, 8.721288442611694, 7.750717401504517, 6.352092266082764, 5.514384746551514, 6.587815523147583, 9.815296411514282]
# LLM_pddl_sg_time = [2.87947678565979, 8.48282527923584, 10.00066065788269, 9.826209783554077, 10.574624300003052, 3.0284512042999268, 4.502398490905762, 2.989583730697632, 3.1735525131225586, 3.2705705165863037, 10.376953125, 9.658543348312378, 10.189276218414307, 4.141259431838989, 9.048610210418701, 10.793684244155884, 12.732467889785767, 2.369957208633423, 2.8408515453338623, 9.718902587890625]
# multiagent_helper_planning_time = [0.2, 5.31, 200.19, 200.17, 200.21, 0.24, 0.23, 0.26, 0.29, 200.27, 11.3, 0.93, 28.62, 45.18, 5.63, 200.21, 14.28, 0.47, 0.22, 2.45]
# multiagent_helper_planning_time_opt = [0.00400391, 5.09884, 1.18775, 6.62245, 33.3258, 0.0, 0.0079841, 0.0040005, 0.0, 14.0016, 11.0939, 0.735845, 28.3945, 44.9525, 5.40703, 6.8628, 14.069, 0.275865, 0.00400013, 2.22358]
# multiagent_helper_cost = [3.0, 32.0, 68.0, 80.0, 134.0, 2.0, 4.0, 2.0, 2.0, 52.0, 36.0, 20.0, 36.0, 42.0, 32.0, 80.0, 36.0, 15.0, 2.0, 26.0]
# multiagent_helper_planning_time_1st = [0.0, 0.00399918, 0.0599698, 0.415866, 1.65971, 0.0, 0.0, 0.0, 0.0, 1.07579, 0.00800495, 0.0159858, 0.00400913, 0.0159798, 0.00799338, 0.431893, 0.00799442, 0.0, 0.0, 0.00800803]
# multiagent_helper_cost_1st = [3, 88, 70, 188, 218, 2, 4, 2, 2, 200, 64, 86, 66, 83, 88, 188, 38, 19, 2, 54]
# multiagent_helper_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [8.6, 13.98, -1, 3.67, -1, 200.21, 200.22, 200.23, 200.25, 200.26, -1, -1, 200.17, 200.18, 200.19, -1, 200.17, 200.16, 200.16, 200.13]
# multiagent_main_planning_time_opt = [8.39843, 13.7815, -1, 3.4553, -1, 157.618, 20.1439, 4.79873, 74.79, 0.367932, -1, -1, 2.93549, 47.1073, 92.9823, -1, 17.401, 5.61399, 54.9227, 56.1997]
# multiagent_main_cost = [32.0, 32.0, -1, 32.0, -1, 170.0, 52.0, 64.0, 110.0, 30.0, -1, -1, 72.0, 68.0, 148.0, -1, 98.0, 62.0, 92.0, 94.0]
# multiagent_main_planning_time_1st = [0.0119778, 0.123945, -1, 0.0119765, -1, 3.65932, 0.175951, 0.0399613, 4.84322, 0.0479846, -1, -1, 0.111964, 0.0639793, 1.87165, -1, 0.395885, 0.0239778, 0.415568, 0.367918]
# multiagent_main_cost_1st = [64, 44, -1, 86, -1, 224, 152, 104, 304, 94, -1, -1, 120, 258, 300, -1, 292, 114, 214, 278]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [32.0, 57.0, 68.0, 110.0, 311.0, 170.0, 55.0, 64.0, 110.0, 78.0, 82.0, 52.0, 93.0, 98.0, 173.0, 201.0, 126.0, 72.0, 92.0, 97.0]
singleagent_planning_time = [27.37, 186.21, 1000.04, 966.96, 1000.1, 1000.14, 1000.16, 1000.11, 1000.17, 995.58, 657.44, 72.92, 1000.1, 1000.12, 1000.14, 1000.09, 1000.09, 490.61, 1000.21, 1000.09]
singleagent_planning_time_opt = [27.1429, 186.013, 370.143, 966.688, 361.868, 146.982, 20.8844, 233.261, 648.039, 198.713, 657.21, 72.7139, 62.6161, 165.048, 834.628, 515.865, 48.0233, 490.374, 28.7666, 202.901]
singleagent_cost = [36.0, 54.0, 68.0, 80.0, 136.0, 172.0, 56.0, 64.0, 100.0, 78.0, 66.0, 46.0, 92.0, 104.0, 146.0, 154.0, 116.0, 76.0, 94.0, 106.0]
singleagent_planning_time_1st = [0.0120059, 0.171934, 0.171815, 0.183939, 12.526, 3.66737, 0.143944, 0.0439676, 4.85917, 3.23139, 0.0199831, 0.0479713, 0.103982, 0.091939, 1.90766, 0.91984, 0.355918, 0.0239767, 0.403897, 0.519861]
singleagent_cost_1st = [66, 108, 138, 270, 258, 224, 152, 104, 298, 220, 142, 150, 202, 274, 310, 382, 292, 134, 170, 266]

LLM_text_sg_time = [5.73597264289856, 8.704964637756348, 8.40870189666748, 6.5770103931427, -1, 7.73762321472168, 7.041723012924194, -1, 8.521178245544434, 7.231583833694458, 6.79531455039978, 8.81351113319397, 10.339109182357788, 8.58828616142273, 8.372446298599243, 7.951304197311401, 8.53868579864502, 12.98108983039856, 6.996376037597656, 8.315569639205933]
LLM_pddl_sg_time = [10.607835054397583, 14.67959976196289, 10.962543725967407, 7.2732625007629395, -1, 9.029894351959229, 1.9943807125091553, -1, 3.506537437438965, 18.76013708114624, 12.901681184768677, 15.034024000167847, 11.869282007217407, 2.7353975772857666, 14.796633005142212, 4.397685289382935, 2.2767114639282227, 3.172424077987671, 15.065510511398315, 12.63187837600708]
multiagent_helper_planning_time = [0.2, 0.19, 0.22, 1000.05, -1, 1000.05, 0.26, -1, 0.29, 1.18, 0.2, 0.2, 0.22, 191.92, 5.81, 1000.14, 5.54, 0.82, 0.44, 2.73]
multiagent_helper_planning_time_opt = [0.00401333, 0.00400031, 0.00400025, 981.407, -1, 651.307, 0.00800847, -1, 0.00400818, 0.875804, 0.0, 0.00400052, 0.0079968, 191.723, 5.56275, 29.7984, 5.35094, 0.63583, 0.207941, 2.51546]
multiagent_helper_cost = [2.0, 2.0, 3.0, 80.0, -1, 126.0, 4.0, -1, 2.0, 14.0, 3.0, 3.0, 4.0, 53.0, 32.0, 86.0, 36.0, 18.0, 14.0, 26.0]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.411881, -1, 1.91559, 0.0, -1, 0.0, 0.0119792, 0.0, 0.0, 0.00411853, 0.215924, 0.0120054, 0.0159881, 0.0, 0.0, 0.0679747, 0.00799482]
multiagent_helper_cost_1st = [2, 2, 3, 188, -1, 218, 4, -1, 2, 56, 3, 3, 4, 119, 88, 108, 42, 24, 20, 54]
multiagent_helper_success = [True, True, True, True, 0, True, True, 0, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [26.23, 185.1, 1000.05, 3.99, -1, 1000.11, 1000.14, -1, 1000.23, 1000.11, 513.0, 67.68, 1000.09, 1000.12, 1000.12, 1000.11, 1000.01, 438.69, 1000.09, 999.98]
multiagent_main_planning_time_opt = [26.0222, 184.891, 367.39, 3.77478, -1, 278.008, 20.3518, -1, 75.6307, 502.476, 512.789, 67.4815, 57.493, 10.318, 625.741, 370.104, 108.264, 438.476, 89.9501, 706.051]
multiagent_main_cost = [34.0, 52.0, 64.0, 32.0, -1, 172.0, 52.0, -1, 110.0, 72.0, 62.0, 42.0, 86.0, 62.0, 140.0, 116.0, 84.0, 62.0, 90.0, 90.0]
multiagent_main_planning_time_1st = [0.0120079, 0.131953, 0.195944, 0.0119921, -1, 1.80361, 0.175927, -1, 5.35508, 3.6633, 0.0199806, 0.0439701, 0.143939, 0.0279721, 1.93164, 0.844088, 0.291928, 0.0239905, 0.367904, 0.367905]
multiagent_main_cost_1st = [72, 94, 134, 86, -1, 322, 152, -1, 304, 208, 136, 144, 208, 156, 300, 260, 302, 114, 166, 278]
multiagent_main_success = [True, True, True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [34.0, 52.0, 64.0, 102.0, 1000000.0, 287.0, 54.0, 1000000.0, 110.0, 72.0, 62.0, 44.0, 86.0, 86.0, 161.0, 198.0, 106.0, 77.0, 100.0, 98.0]


singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))

# # plt.plot(singleagent_planning_time, '-r') #, figure=fig_barman_time)
# plt.plot(singleagent_cost, '-r') #, figure=fig_barman_length)
# # plt.plot(multiagent_total_planning_time, '-g') #, figure=fig_barman_time)
# plt.plot(overall_plan_length, '-g') #, figure=fig_barman_length)

print('tyreworld')
singleagent = []
helper = []
main = []

#tyreworld
# singleagent_planning_time = [0.25, 200.32, 200.16, 200.22, 200.26, 200.26, 200.27, 200.31, 200.29, 200.33, 200.31, 200.37, 200.42, 200.42, 200.44, 200.48, 200.47, 200.5, 200.54, 200.58]
# singleagent_planning_time_opt = [0.0799144, 0.10796, 0.00399893, 0.00799589, 0.0199865, 0.0319657, 0.0479648, 0.0719439, 0.107956, 0.131949, 0.179941, 0.259916, 0.351869, 0.467857, 0.447893, 0.827747, 0.999708, 1.1901, 1.29568, 1.58758]
# singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
# singleagent_planning_time_1st = [0.0, 0.10796, 0.00399893, 0.00799589, 0.0199865, 0.0319657, 0.0479648, 0.0719439, 0.107956, 0.131949, 0.179941, 0.259916, 0.351869, 0.467857, 0.447893, 0.827747, 0.999708, 1.1901, 1.29568, 1.58758]
# singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]
# LLM_text_sg_time = [-1, 7.988981008529663, 6.144626140594482, 7.176294565200806, 10.584416627883911, 9.766372442245483, -1, 7.931746959686279, 8.15741229057312, 9.77629828453064, 8.86642599105835, 10.278109312057495, -1, -1, 7.887667655944824, -1, 5.8365325927734375, 5.794299125671387, 6.281625270843506, 7.647354602813721]
# LLM_pddl_sg_time = [-1, 11.861425161361694, 2.2765097618103027, 3.0531818866729736, 4.0119078159332275, 4.760014772415161, -1, 4.64972710609436, 13.286390542984009, 13.750185012817383, 5.522075653076172, 6.288310289382935, -1, -1, 9.135888814926147, -1, 9.712172985076904, 9.135564804077148, 9.39732813835144, 16.971848249435425]
# multiagent_helper_planning_time = [-1, 200.39, 0.14, 0.15, 0.16, 0.18, -1, 0.24, 200.32, 200.35, 0.35, 0.46, -1, -1, 1.79, -1, 6.06, 12.21, 24.82, 200.5]
# multiagent_helper_planning_time_opt = [-1, 0.00397313, 0.0, 0.00398247, 0.0079875, 0.0119815, -1, 0.0359588, 0.00798273, 0.00800281, 0.0879025, 0.18791, -1, -1, 1.43972, -1, 5.64298, 11.754, 24.3319, 0.00399322]
# multiagent_helper_cost = [-1, 56.0, 3.0, 4.0, 5.0, 6.0, -1, 8.0, 55.0, 61.0, 11.0, 12.0, -1, -1, 15.0, -1, 17.0, 18.0, 19.0, 20.0]
# multiagent_helper_planning_time_1st = [-1, 0.00397313, 0.0, 0.0, 0.0, 0.0, -1, 0.00397876, 0.00798273, 0.00800281, 0.0, 0.0, -1, -1, 0.0, -1, 0.0, 0.0, 0.00397732, 0.00399322]
# multiagent_helper_cost_1st = [-1, 56, 3, 4, 5, 6, -1, 8, 55, 61, 11, 12, -1, -1, 15, -1, 17, 18, 19, 20]
# multiagent_helper_success = [0, True, True, True, True, True, 0, True, True, True, True, True, 0, 0, True, 0, True, True, True, True]
# multiagent_main_planning_time = [-1, -1, 200.21, 200.22, 200.24, 200.25, -1, 200.27, 200.27, 200.3, 200.32, 200.36, -1, -1, 200.35, -1, 200.48, 200.5, 200.41, 200.56]
# multiagent_main_planning_time_opt = [-1, -1, 0.00399978, 0.00799311, 0.0159946, 0.0239821, -1, 0.0599539, 0.0159902, 0.0319798, 0.151937, 0.211944, -1, -1, 0.519817, -1, 0.815775, 0.943744, 1.21564, 1.51148]
# multiagent_main_cost = [-1, -1, 32.0, 42.0, 52.0, 62.0, -1, 82.0, 46.0, 51.0, 112.0, 122.0, -1, -1, 152.0, -1, 172.0, 182.0, 192.0, 202.0]
# multiagent_main_planning_time_1st = [-1, -1, 0.00399978, 0.00799311, 0.0159946, 0.0239821, -1, 0.0599539, 0.0159902, 0.0319798, 0.151937, 0.211944, -1, -1, 0.519817, -1, 0.815775, 0.943744, 1.21564, 1.51148]
# multiagent_main_cost_1st = [-1, -1, 32, 42, 52, 62, -1, 82, 46, 51, 112, 122, -1, -1, 152, -1, 172, 182, 192, 202]
# multiagent_main_success = [False, True, True, True, True, True, False, True, True, True, True, True, False, False, True, False, True, True, True, True]
# overall_plan_length = [0.0, 92.0, 32.0, 42.0, 52.0, 62.0, 0.0, 82.0, 92.0, 102.0, 112.0, 122.0, 0.0, 0.0, 152.0, 0.0, 172.0, 182.0, 192.0, 202.0]



# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.22, 200.29, 200.21, 200.22, 200.24, 200.31, 200.25, 200.27, 200.31, 200.32, 200.32, 200.41, 200.39, 200.44, 200.44, 200.43, 200.47, 200.47, 200.53, 200.6]
# singleagent_planning_time_opt = [0.0839397, 0.107938, 0.00399842, 0.0119951, 0.0159737, 0.027989, 0.0439539, 0.067966, 0.107944, 0.131959, 0.179938, 0.259907, 0.351885, 0.467856, 0.619466, 0.791734, 0.983725, 1.23549, 1.23567, 1.5196]
# singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
# singleagent_planning_time_1st = [0.00397868, 0.107938, 0.00399842, 0.0119951, 0.0159737, 0.027989, 0.0439539, 0.067966, 0.107944, 0.131959, 0.179938, 0.259907, 0.351885, 0.467856, 0.619466, 0.791734, 0.983725, 1.23549, 1.23567, 1.5196]
# singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]
# LLM_text_sg_time = [-1, 7.5391340255737305, 9.019395112991333, -1, 13.04223346710205, -1, 9.483526229858398, 6.020125150680542, 7.243467807769775, -1, 7.350370168685913, 9.141703605651855, 8.639878749847412, -1, 5.3214111328125, 6.56707239151001, 6.4704296588897705, 7.379274606704712, 7.814316749572754, 7.2394397258758545]
# LLM_pddl_sg_time = [-1, 6.071399211883545, 3.8115806579589844, -1, 4.371270656585693, -1, 7.233831405639648, 4.984260320663452, 6.631823778152466, -1, 5.901994943618774, 11.243041276931763, 7.805225610733032, -1, 7.040489912033081, 8.542266607284546, 8.900469541549683, 11.197028160095215, 8.23900318145752, 10.814849615097046]
# multiagent_helper_planning_time = [-1, 0.31, 0.14, -1, 0.14, -1, 200.33, 0.21, 0.46, -1, 0.34, 200.4, 0.69, -1, 1.82, 3.2, 6.15, 12.24, 24.77, 52.06]
# multiagent_helper_planning_time_opt = [-1, 0.0759418, 0.0, -1, 0.00796707, -1, 0.00400412, 0.0119735, 0.247906, -1, 0.0879478, 0.0119951, 0.387916, -1, 1.46369, 2.82348, 5.73843, 11.8021, 24.2827, 51.5401]
# multiagent_helper_cost = [-1, 9.0, 3.0, -1, 5.0, -1, 43.0, 8.0, 9.0, -1, 11.0, 73.0, 13.0, -1, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
# multiagent_helper_planning_time_1st = [-1, 0.0, 0.0, -1, 0.0, -1, 0.00400412, 0.00397304, 0.0, -1, 0.00397897, 0.0119951, 0.0, -1, 0.00397355, 0.00397473, 0.0, 0.00397637, 0.0, 0.0]
# multiagent_helper_cost_1st = [-1, 9, 3, -1, 5, -1, 43, 8, 9, -1, 11, 73, 13, -1, 15, 16, 17, 18, 19, 20]
# multiagent_helper_success = [0, True, True, 0, True, 0, True, True, True, 0, True, True, True, 0, True, True, True, True, True, True]
# multiagent_main_planning_time = [-1, 200.33, 200.28, -1, 200.3, -1, 200.23, 200.29, 200.29, -1, 200.31, 200.36, 200.39, -1, 200.4, 200.47, 200.47, 200.49, 200.53, 200.6]
# multiagent_main_planning_time_opt = [-1, 0.0879433, 0.00400001, -1, 0.0119806, -1, 0.0119847, 0.0599682, 0.0879732, -1, 0.15192, 0.0599525, 0.287905, -1, 0.519767, 0.659808, 0.595847, 1.02372, 1.17557, 1.35168]
# multiagent_main_cost = [-1, 92.0, 32.0, -1, 52.0, -1, 36.0, 82.0, 92.0, -1, 112.0, 61.0, 132.0, -1, 152.0, 162.0, 172.0, 182.0, 192.0, 202.0]
# multiagent_main_planning_time_1st = [-1, 0.0879433, 0.00400001, -1, 0.0119806, -1, 0.0119847, 0.0599682, 0.0879732, -1, 0.15192, 0.0599525, 0.287905, -1, 0.519767, 0.659808, 0.595847, 1.02372, 1.17557, 1.35168]
# multiagent_main_cost_1st = [-1, 92, 32, -1, 52, -1, 36, 82, 92, -1, 112, 61, 132, -1, 152, 162, 172, 182, 192, 202]
# multiagent_main_success = [False, True, True, False, True, False, True, True, True, False, True, True, True, False, True, True, True, True, True, True]
# overall_plan_length = [0.0, 92.0, 32.0, 0.0, 52.0, 0.0, 72.0, 82.0, 92.0, 0.0, 112.0, 122.0, 132.0, 0.0, 152.0, 162.0, 172.0, 182.0, 192.0, 202.0]

# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.21, 200.29, 200.26, 200.22, 200.23, 200.32, 200.26, 200.25, 200.26, 200.32, 200.31, 200.36, 200.38, 200.41, 200.42, 200.44, 200.46, 200.49, 200.54, 200.57]
# singleagent_planning_time_opt = [0.0879317, 0.107944, 0.00399917, 0.00800348, 0.0159965, 0.027977, 0.0439572, 0.0559649, 0.107952, 0.127952, 0.179939, 0.183943, 0.247938, 0.46785, 0.611833, 0.799787, 0.995711, 1.05573, 1.39507, 1.5636]
# singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
# singleagent_planning_time_1st = [0.0, 0.107944, 0.00399917, 0.00800348, 0.0159965, 0.027977, 0.0439572, 0.0559649, 0.107952, 0.127952, 0.179939, 0.183943, 0.247938, 0.46785, 0.611833, 0.799787, 0.995711, 1.05573, 1.39507, 1.5636]
# singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]
# LLM_text_sg_time = [7.381258964538574, 7.987065315246582, 8.131840467453003, 6.412057638168335, -1, 7.815313816070557, -1, 7.645282030105591, 6.8800506591796875, 5.4211204051971436, -1, 6.245121002197266, 8.007182359695435, 5.605431079864502, 8.911698579788208, 4.938750267028809, -1, 7.699563980102539, 5.842117071151733, 6.205753326416016]
# LLM_pddl_sg_time = [1.6612999439239502, 8.078295707702637, 3.880155324935913, 3.1719167232513428, -1, 4.929965972900391, -1, 5.124948501586914, 4.724615812301636, 5.536572456359863, -1, 6.46062445640564, 14.184890270233154, 7.657474756240845, 5.70011305809021, 7.034242153167725, -1, 8.231815338134766, 8.79175877571106, 9.595764636993408]
# multiagent_helper_planning_time = [0.13, 200.24, 0.16, 0.15, -1, 0.18, -1, 0.22, 0.24, 0.28, -1, 0.46, 200.4, 1.15, 2.1, 3.17, -1, 12.07, 24.9, 51.11]
# multiagent_helper_planning_time_opt = [0.00397654, 0.0, 0.00397188, 0.00397607, -1, 0.0119781, -1, 0.0120061, 0.0199909, 0.043976, -1, 0.187903, 0.0119941, 0.815823, 1.755, 2.78755, -1, 11.6375, 24.4115, 50.5758]
# multiagent_helper_cost = [1.0, 9.0, 3.0, 4.0, -1, 6.0, -1, 8.0, 9.0, 10.0, -1, 12.0, 80.0, 14.0, 15.0, 16.0, -1, 18.0, 19.0, 20.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, -1, 0.00397585, -1, 0.0, 0.0, 0.0, -1, 0.00397276, 0.0119941, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0]
# multiagent_helper_cost_1st = [1, 9, 3, 4, -1, 6, -1, 8, 9, 10, -1, 12, 80, 14, 15, 16, -1, 18, 19, 20]
# multiagent_helper_success = [True, True, True, True, 0, True, 0, True, True, True, 0, True, True, True, True, True, 0, True, True, True]
# multiagent_main_planning_time = [0.17, 200.3, 200.23, 200.26, -1, 200.28, -1, 200.27, 200.34, 200.33, -1, 200.36, -1, 200.33, 200.46, 200.36, -1, 200.49, 200.46, 200.56]
# multiagent_main_planning_time_opt = [0.0399532, 0.091939, 0.00399875, 0.00798658, -1, 0.0239755, -1, 0.0639476, 0.0879691, 0.111917, -1, 0.20794, -1, 0.399891, 0.51584, 0.643832, -1, 0.915763, 1.13572, 1.35967]
# multiagent_main_cost = [12.0, 92.0, 32.0, 42.0, -1, 62.0, -1, 82.0, 92.0, 102.0, -1, 122.0, -1, 142.0, 152.0, 162.0, -1, 182.0, 192.0, 202.0]
# multiagent_main_planning_time_1st = [0.0, 0.091939, 0.00399875, 0.00798658, -1, 0.0239755, -1, 0.0639476, 0.0879691, 0.111917, -1, 0.20794, -1, 0.399891, 0.51584, 0.643832, -1, 0.915763, 1.13572, 1.35967]
# multiagent_main_cost_1st = [12, 92, 32, 42, -1, 62, -1, 82, 92, 102, -1, 122, -1, 142, 152, 162, -1, 182, 192, 202]
# multiagent_main_success = [True, True, True, True, False, True, False, True, True, True, False, True, True, True, True, True, False, True, True, True]
# overall_plan_length = [12.0, 92.0, 32.0, 42.0, 0.0, 62.0, 0.0, 82.0, 92.0, 102.0, 0.0, 122.0, 132.0, 142.0, 152.0, 162.0, 0.0, 182.0, 192.0, 202.0]

singleagent_planning_time = [0.26, 1000.63, 1000.69, 1000.62, 1000.56, 1000.65, 1000.64, 1000.5, 1000.6, 1000.54, 1000.56, 1000.65, 1000.93, 1000.58, 1000.9, 1000.71, 1000.77, 1000.74, 1000.99, 1000.91]
singleagent_planning_time_opt = [0.0839496, 0.107966, 0.00400062, 0.0119944, 0.0200051, 0.0279846, 0.0479622, 0.0719413, 0.115946, 0.131931, 0.183931, 0.17995, 0.343889, 0.46784, 0.631819, 0.795769, 1.00371, 1.26758, 1.40361, 1.65955]
singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
singleagent_planning_time_1st = [0.0, 0.107966, 0.00400062, 0.0119944, 0.0200051, 0.0279846, 0.0479622, 0.0719413, 0.115946, 0.131931, 0.183931, 0.17995, 0.343889, 0.46784, 0.631819, 0.795769, 1.00371, 1.26758, 1.40361, 1.65955]
singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]
LLM_text_sg_time = [4.347130537033081, 9.544127941131592, 5.865442514419556, 8.571415424346924, 9.05169677734375, -1, -1, 7.968988656997681, 9.16797685623169, 10.197810173034668, 11.03992247581482, 7.9715495109558105, 8.70791482925415, -1, 10.93392014503479, 6.438980340957642, 10.72422194480896, -1, 5.993429183959961, -1]
LLM_pddl_sg_time = [1.2758638858795166, 5.4217848777771, 3.2902870178222656, 4.97832727432251, 3.6241862773895264, -1, -1, 6.1455397605896, 5.522342205047607, 8.333192586898804, 18.372519731521606, 9.02430772781372, 15.518158674240112, -1, 9.669440984725952, 4.860379219055176, 16.448933362960815, -1, 11.788738012313843, -1]
multiagent_helper_planning_time = [0.14, 1000.07, 0.15, 0.16, 0.16, -1, -1, 100.64, 0.29, 1000.33, 1000.85, 2.8, 1000.61, -1, 6.09, 3.17, 1000.59, -1, 24.52, -1]
multiagent_helper_planning_time_opt = [0.0, 278.056, 0.00397692, 0.00399968, 0.00397248, -1, -1, 100.419, 0.079936, 0.0, 0.011973, 2.52743, 0.0119937, -1, 5.74297, 2.79552, 0.0, -1, 24.0116, -1]
multiagent_helper_cost = [1.0, 9.0, 3.0, 4.0, 5.0, -1, -1, 8.0, 9.0, 10.0, 68.0, 12.0, 79.0, -1, 15.0, 16.0, 17.0, -1, 19.0, -1]
multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, -1, -1, 0.0, 0.00397597, 0.0, 0.011973, 0.0, 0.0119937, -1, 0.0039747, 0.0, 0.0, -1, 0.0, -1]
multiagent_helper_cost_1st = [1, 9, 3, 4, 5, -1, -1, 8, 9, 10, 68, 12, 79, -1, 15, 16, 17, -1, 19, -1]
multiagent_helper_success = [True, True, True, True, True, 0, 0, True, True, True, True, True, True, 0, True, True, True, 0, True, 0]
multiagent_main_planning_time = [0.17, 1000.6, 1000.26, 1000.41, 1000.5, -1, -1, 1000.5, 1000.51, 1000.45, 1000.35, 1000.51, 1000.43, -1, 1000.62, 1000.63, 1000.69, -1, 1000.62, -1]
multiagent_main_planning_time_opt = [0.0399359, 0.0879692, 0.00400065, 0.00800227, 0.0119744, -1, -1, 0.0599657, 0.0879256, 0.107964, 0.035981, 0.147954, 0.0839575, -1, 0.519862, 0.663806, 0.815641, -1, 0.939736, -1]
multiagent_main_cost = [12.0, 92.0, 32.0, 42.0, 52.0, -1, -1, 82.0, 92.0, 102.0, 57.0, 122.0, 66.0, -1, 152.0, 162.0, 172.0, -1, 192.0, -1]
multiagent_main_planning_time_1st = [0.00397828, 0.0879692, 0.00400065, 0.00800227, 0.0119744, -1, -1, 0.0599657, 0.0879256, 0.107964, 0.035981, 0.147954, 0.0839575, -1, 0.519862, 0.663806, 0.815641, -1, 0.939736, -1]
multiagent_main_cost_1st = [12, 92, 32, 42, 52, -1, -1, 82, 92, 102, 57, 122, 66, -1, 152, 162, 172, -1, 192, -1]
multiagent_main_success = [True, True, True, True, True, False, False, True, True, True, True, True, True, False, True, True, True, False, True, False]
overall_plan_length = [12.0, 92.0, 32.0, 42.0, 52.0, 1000000.0, 1000000.0, 82.0, 92.0, 102.0, 112.0, 122.0, 132.0, 1000000.0, 152.0, 162.0, 172.0, 1000000.0, 192.0, 1000000.0]


singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))

# # plt.plot(singleagent_planning_time, '-r') #, figure=fig_barman_time)
# plt.plot(singleagent_cost, '-r') #, figure=fig_barman_length)
# # plt.plot(multiagent_total_planning_time, '-g') #, figure=fig_barman_time)
# plt.plot(overall_plan_length, '-g') #, figure=fig_barman_length)

print('grippers')
singleagent = []
helper = []
main = []

#grippers

# singleagent_planning_time = [0.14, 0.15, 0.15, 0.12, 0.13, 0.12, 0.15, 0.15, 6.91, 0.13, 0.12, 0.18, 0.09, 0.17, 0.18, 17.32, 0.12, 0.2, 4.12, 0.12]
# singleagent_planning_time_opt = [0.00399954, 0.0199852, 0.00800343, 0.00399974, 0.00397479, 0.0, 0.0199503, 0.0239741, 6.78239, 0.00797872, 0.00393889, 0.0639596, 0.00397335, 0.0159758, 0.0439563, 17.1725, 0.0119918, 0.0879558, 3.98302, 0.0080041]
# singleagent_cost = [5.0, 9.0, 6.0, 4.0, 4.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 6.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00399972]
# singleagent_cost_1st = [5, 10, 6, 4, 4, 4, 8, 13, 20, 10, 3, 9, 6, 6, 12, 22, 9, 10, 21, 7]
# LLM_text_sg_time = [5.61240291595459, 3.527479648590088, -1, 3.0913472175598145, 5.433417558670044, -1, 5.050108194351196, 4.682713508605957, 4.71819281578064, 4.1320414543151855, 3.3714098930358887, 4.902026891708374, 4.463600158691406, 2.920133113861084, 2.541612386703491, 4.3077099323272705, 3.7687320709228516, 4.4170472621917725, 4.6888978481292725, 6.420567750930786]
# LLM_pddl_sg_time = [2.638540506362915, 2.9524166584014893, -1, 1.5925447940826416, 1.9457371234893799, -1, 2.152587413787842, 1.4955024719238281, 2.1681699752807617, 1.437345027923584, 2.9207117557525635, 1.4126975536346436, 1.5966949462890625, 1.9347529411315918, 2.9913809299468994, 2.5512969493865967, 2.0039429664611816, 3.0832748413085938, 2.046959400177002, 1.8330657482147217]
# multiagent_helper_planning_time = [0.14, 0.14, -1, 0.12, 0.13, -1, 0.13, 0.12, 0.12, -1, -1, -1, 0.15, -1, 0.13, 0.15, 0.14, 0.13, 0.15, 0.12]
# multiagent_helper_planning_time_opt = [0.0, 0.00400172, -1, 0.00394583, 0.00397457, -1, 0.00392865, 0.00400141, 0.00399997, -1, -1, -1, 0.0, -1, 0.00400417, 0.00798177, 0.0, 0.00400392, 0.00399966, 0.0039798]
# multiagent_helper_cost = [3.0, 4.0, -1, 4.0, 4.0, -1, 4.0, 4.0, 4.0, -1, -1, -1, 4.0, -1, 4.0, 4.0, 3.0, 4.0, 4.0, 4.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, -1, 0.0, 0.0, -1, 0.0, 0.0, 0.0, -1, -1, -1, 0.0, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0039798]
# multiagent_helper_cost_1st = [3, 4, -1, 4, 4, -1, 4, 4, 4, -1, -1, -1, 4, -1, 5, 4, 3, 5, 5, 5]
# multiagent_helper_success = [True, True, 0, True, True, 0, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [0.13, 0.19, -1, 0.13, 0.12, -1, 0.15, 0.16, 3.59, 0.12, 0.13, 0.22, 0.14, 0.12, 0.12, 26.38, 0.15, 0.15, 3.11, 0.12]
# multiagent_main_planning_time_opt = [0.00397583, 0.0479675, -1, 0.0159816, 0.00800291, -1, 0.00798988, 0.0359893, 3.4592, 0.01198, 0.003973, 0.063943, 0.00401586, 0.0119956, 0.01198, 26.2382, 0.0319715, 0.0359606, 2.97931, 0.00397515]
# multiagent_main_cost = [3.0, 12.0, -1, 7.0, 6.0, -1, 6.0, 13.0, 15.0, 10.0, 3.0, 9.0, 4.0, 6.0, 7.0, 21.0, 12.0, 7.0, 14.0, 4.0]
# multiagent_main_planning_time_1st = [0.0, 0.0, -1, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.00397564, 0.0, 0.00398221, 0.0, 0.0, 0.0, 0.00400002, 0.00397846, 0.0, 0.0, 0.0]
# multiagent_main_cost_1st = [3, 13, -1, 7, 6, -1, 6, 15, 18, 10, 3, 9, 4, 6, 8, 26, 12, 7, 17, 4]
# multiagent_main_success = [True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [3.0, 12.0, 0.0, 8.0, 10.0, 0.0, 6.0, 13.0, 15.0, 10, 3, 9, 4.0, 6, 7.0, 21.0, 12.0, 7.0, 14.0, 4.0]

# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)


# singleagent_planning_time = [0.14, 0.15, 0.14, 0.15, 0.13, 0.12, 0.16, 0.14, 6.88, 0.14, 0.11, 0.15, 0.12, 0.17, 0.17, 19.11, 0.14, 0.21, 4.1, 0.11]
# singleagent_planning_time_opt = [0.00399977, 0.0199888, 0.00800245, 0.00399954, 0.00797824, 0.0, 0.019985, 0.0239717, 6.75452, 0.011984, 0.00399983, 0.0479576, 0.00797943, 0.015995, 0.0439577, 18.956, 0.0119942, 0.0879494, 3.97099, 0.00400005]
# singleagent_cost = [5.0, 9.0, 6.0, 4.0, 4.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 6.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00397397, 0.0]
# singleagent_cost_1st = [5, 10, 6, 4, 4, 4, 8, 13, 20, 10, 3, 9, 6, 6, 12, 22, 9, 10, 21, 7]
# LLM_text_sg_time = [5.969178915023804, 3.7457242012023926, 4.36852240562439, 3.763730049133301, 2.946866989135742, -1, 5.247357368469238, 5.320229530334473, 6.752861738204956, 4.019553899765015, 3.537316083908081, 4.240198850631714, -1, 3.7881550788879395, 2.9566261768341064, 3.655266761779785, 3.597285747528076, 4.080933332443237, 5.925579071044922, 5.942858457565308]
# LLM_pddl_sg_time = [2.0277440547943115, 2.24588942527771, 2.3399345874786377, 2.1839561462402344, 1.8661201000213623, -1, 2.0835371017456055, 1.9771068096160889, 1.8557476997375488, 2.0881640911102295, 3.202016830444336, 1.620640754699707, -1, 1.4730515480041504, 2.5554652214050293, 2.154784917831421, 2.455260992050171, 2.917659044265747, 2.6089775562286377, 2.3661773204803467]
# multiagent_helper_planning_time = [0.14, 0.14, 0.14, -1, -1, -1, 0.14, 0.11, 0.14, -1, -1, -1, -1, -1, 0.14, 0.13, -1, 0.14, 0.12, 0.12]
# multiagent_helper_planning_time_opt = [0.0, 0.00397713, 0.0, -1, -1, -1, 0.00799139, 0.0039985, 0.00397115, -1, -1, -1, -1, -1, 0.00400021, 0.00799349, -1, 0.0040044, 0.0039756, 0.00399994]
# multiagent_helper_cost = [3.0, 4.0, 1.0, -1, -1, -1, 4.0, 4.0, 4.0, -1, -1, -1, -1, -1, 4.0, 4.0, -1, 4.0, 4.0, 4.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, 0.0, -1, -1, -1, 0.0, 0.0, 0.0, -1, -1, -1, -1, -1, 0.0, 0.0, -1, 0.0, 0.0, 0.0]
# multiagent_helper_cost_1st = [3, 4, 1, -1, -1, -1, 4, 4, 4, -1, -1, -1, -1, -1, 5, 4, -1, 5, 5, 5]
# multiagent_helper_success = [True, True, True, True, True, 0, True, True, True, True, True, True, 0, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [0.13, 0.19, 0.14, 0.13, 0.14, -1, 0.14, 0.15, 3.6, 0.11, 0.13, 0.2, -1, 0.15, 0.15, 8.28, 0.12, 0.14, 2.57, 0.14]
# multiagent_main_planning_time_opt = [0.00400057, 0.0479615, 0.00800243, 0.00397607, 0.00397592, -1, 0.0079914, 0.0319802, 3.46325, 0.0119936, 0.0, 0.0639397, -1, 0.0159831, 0.0119768, 8.13829, 0.0119941, 0.0359659, 2.43951, 0.0]
# multiagent_main_cost = [3.0, 12.0, 6.0, 4.0, 4.0, -1, 6.0, 13.0, 15.0, 10.0, 3.0, 9.0, -1, 6.0, 7.0, 16.0, 8.0, 7.0, 14.0, 4.0]
# multiagent_main_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.00397592, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0039701, 0.00400402, 0.0, 0.0, 0.00397731, 0.0]
# multiagent_main_cost_1st = [3, 13, 6, 4, 4, -1, 6, 15, 18, 10, 3, 9, -1, 6, 8, 19, 9, 7, 17, 4]
# multiagent_main_success = [True, True, True, True, True, False, True, True, True, True, True, True, False, True, True, True, True, True, True, True]
# overall_plan_length = [3.0, 12.0, 6.0, 4, 4, 0.0, 6.0, 13.0, 15.0, 10, 3, 9, 0.0, 6, 7.0, 16.0, 8, 7.0, 14.0, 4.0]


# singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)


# singleagent_planning_time = [0.13, 0.15, 0.14, 0.14, 0.12, 0.11, 0.14, 0.14, 6.72, 0.15, 0.11, 0.15, 0.14, 0.15, 0.18, 18.85, 0.14, 0.22, 3.28, 0.12]
# singleagent_planning_time_opt = [0.00399986, 0.0199751, 0.00800551, 0.00398207, 0.00396737, 0.0, 0.0199736, 0.0279524, 6.58271, 0.0119715, 0.00400013, 0.0639566, 0.00399978, 0.0159809, 0.0439673, 18.6876, 0.0119846, 0.0839463, 3.16337, 0.00400431]
# singleagent_cost = [5.0, 9.0, 6.0, 4.0, 4.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 6.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.00398207, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# singleagent_cost_1st = [5, 10, 6, 4, 4, 4, 8, 13, 20, 10, 3, 9, 6, 6, 12, 22, 9, 10, 21, 7]
# LLM_text_sg_time = [6.276122093200684, 5.304713487625122, -1, 3.097393035888672, 3.998051404953003, -1, 4.701598644256592, 3.5403647422790527, 3.740478515625, 4.158710718154907, 4.28812050819397, 3.052905797958374, 3.0350699424743652, 2.5315446853637695, 5.647526979446411, 5.73463249206543, 3.1245977878570557, 5.352196455001831, 5.6644606590271, 5.4615843296051025]
# LLM_pddl_sg_time = [2.4610300064086914, 2.1684138774871826, -1, 2.1447415351867676, 2.933377742767334, -1, 1.7393264770507812, 1.6894431114196777, 1.6626076698303223, 4.246434211730957, 2.27520751953125, 1.9843807220458984, 2.1946780681610107, 2.5717673301696777, 2.3454818725585938, 1.929079294204712, 1.8180060386657715, 1.6302800178527832, 1.8868935108184814, 2.1780354976654053]
# multiagent_helper_planning_time = [0.13, 0.13, -1, -1, -1, -1, 0.13, 0.12, 0.14, -1, -1, -1, 0.13, -1, 0.13, 0.14, -1, 0.13, 0.14, 0.12]
# multiagent_helper_planning_time_opt = [0.00397841, 0.0, -1, -1, -1, -1, 0.00399976, 0.0040014, 0.00399956, -1, -1, -1, 0.0, -1, 0.00397872, 0.00800598, -1, 0.00400429, 0.00400022, 0.00399913]
# multiagent_helper_cost = [3.0, 4.0, -1, -1, -1, -1, 4.0, 4.0, 4.0, -1, -1, -1, 4.0, -1, 4.0, 4.0, -1, 4.0, 4.0, 4.0]
# multiagent_helper_planning_time_1st = [0.0, 0.0, -1, -1, -1, -1, 0.0, 0.0, 0.0, -1, -1, -1, 0.0, -1, 0.0, 0.0, -1, 0.0, 0.0, 0.0]
# multiagent_helper_cost_1st = [3, 4, -1, -1, -1, -1, 4, 4, 4, -1, -1, -1, 4, -1, 5, 4, -1, 5, 5, 5]
# multiagent_helper_success = [True, True, 0, True, True, 0, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# multiagent_main_planning_time = [0.13, 0.17, -1, 0.13, 0.13, -1, 0.13, 0.15, 3.91, 0.14, 0.12, 0.21, 0.12, 0.14, 0.12, 26.13, 0.13, 0.16, 3.11, 0.12]
# multiagent_main_planning_time_opt = [0.0, 0.0479555, -1, 0.00400503, 0.00397416, -1, 0.011987, 0.0319715, 3.77913, 0.00799309, 0.00396974, 0.0639575, 0.0039716, 0.0159436, 0.0119681, 25.9986, 0.0119786, 0.0359625, 2.9783, 0.00398036]
# multiagent_main_cost = [3.0, 12.0, -1, 4.0, 4.0, -1, 6.0, 13.0, 15.0, 10.0, 3.0, 9.0, 4.0, 6.0, 7.0, 21.0, 8.0, 7.0, 14.0, 4.0]
# multiagent_main_planning_time_1st = [0.0, 0.0, -1, 0.0, 0.00397416, -1, 0.0, 0.0, 0.0, 0.0, 0.00396974, 0.0, 0.0, 0.00396388, 0.00400054, 0.0, 0.0, 0.00397661, 0.00400058, 0.0]
# multiagent_main_cost_1st = [3, 13, -1, 4, 4, -1, 6, 15, 18, 10, 3, 9, 4, 6, 9, 26, 9, 7, 17, 4]
# multiagent_main_success = [True, True, False, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [3.0, 12.0, 0.0, 4, 4, 0.0, 6.0, 13.0, 15.0, 10, 3, 9, 4.0, 6, 7.0, 21.0, 8, 7.0, 14.0, 4.0]
singleagent_planning_time = [0.13, 0.16, 0.13, 0.14, 0.13, 0.13, 0.15, 0.16, 7.22, 0.15, 0.13, 0.19, 0.13, 0.13, 0.18, 17.91, 0.14, 0.23, 3.31, 0.12]
singleagent_planning_time_opt = [0.00401923, 0.0279695, 0.00800241, 0.00797649, 0.00797608, 0.00399986, 0.0199725, 0.0239541, 7.08264, 0.011994, 0.00397829, 0.0639629, 0.00399986, 0.0159853, 0.0439537, 17.7568, 0.0119949, 0.0839565, 3.17516, 0.00397153]
singleagent_cost = [5.0, 9.0, 6.0, 4.0, 4.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 6.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.00397663, 0.0, 0.0, 0.00398008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00397153]
singleagent_cost_1st = [5, 10, 6, 4, 4, 4, 8, 13, 20, 10, 3, 9, 6, 6, 12, 22, 9, 10, 21, 7]

LLM_text_sg_time = [9.401407718658447, 8.105726718902588, 8.290205717086792, 2.9894001483917236, 6.213841676712036, -1, 5.203955411911011, 6.297537088394165, 7.780224084854126, 4.63329815864563, 3.975628137588501, 8.470896482467651, 7.3817503452301025, 2.949833393096924, 5.334117889404297, 7.607184648513794, 5.259685039520264, 6.959804534912109, 9.056033372879028, 8.641018390655518]
LLM_pddl_sg_time = [2.5507071018218994, 2.6342062950134277, 1.9002060890197754, 3.041137933731079, 3.363213300704956, -1, 1.8097717761993408, 2.839942455291748, 3.396552085876465, 2.6819400787353516, 2.391693592071533, 2.167673110961914, 2.98362398147583, 2.67110538482666, 2.238509178161621, 2.5968873500823975, 2.0480659008026123, 2.1356518268585205, 3.058311939239502, 2.5963265895843506]
multiagent_helper_planning_time = [0.14, -1, 0.14, -1, -1, -1, 0.14, 0.14, 0.16, -1, -1, 0.14, 0.13, -1, 0.14, 0.15, -1, -1, 0.15, 0.12]
multiagent_helper_planning_time_opt = [0.0, -1, 0.0, -1, -1, -1, 0.00400008, 0.00400452, 0.00400022, -1, -1, 0.00401537, 0.00397989, -1, 0.00399838, 0.00800269, -1, -1, 0.00798774, 0.0]
multiagent_helper_cost = [3.0, -1, 3.0, -1, -1, -1, 4.0, 4.0, 4.0, -1, -1, 4.0, 4.0, -1, 4.0, 4.0, -1, -1, 4.0, 4.0]
multiagent_helper_planning_time_1st = [0.0, -1, 0.0, -1, -1, -1, 0.0, 0.0, 0.0, -1, -1, 0.0, 0.00397989, -1, 0.0, 0.0, -1, -1, 0.0, 0.0]
multiagent_helper_cost_1st = [3, -1, 3, -1, -1, -1, 4, 4, 4, -1, -1, 4, 4, -1, 5, 5, -1, -1, 5, 5]
multiagent_helper_success = [True, True, True, True, True, 0, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
multiagent_main_planning_time = [0.12, 0.16, 0.15, 0.15, 0.13, -1, 0.14, 0.17, 3.47, 0.14, 0.13, 0.15, 0.14, 0.16, 0.15, 28.95, 0.13, 0.23, 2.78, 0.13]
multiagent_main_planning_time_opt = [0.00397574, 0.0199872, 0.00399876, 0.00401833, 0.00399991, -1, 0.00800255, 0.0319765, 3.32731, 0.0119778, 0.00401028, 0.0159846, 0.00400026, 0.0160046, 0.0120083, 28.8063, 0.011991, 0.0879369, 2.64349, 0.0040015]
multiagent_main_cost = [3.0, 9.0, 3.0, 4.0, 4.0, -1, 6.0, 13.0, 15.0, 10.0, 3.0, 6.0, 4.0, 6.0, 7.0, 22.0, 8.0, 9.0, 14.0, 4.0]
multiagent_main_planning_time_1st = [0.00397574, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, 0.0, 0.00397565, 0.00401028, 0.0, 0.0, 0.0, 0.0, 0.00399971, 0.0, 0.0, 0.0, 0.0]
multiagent_main_cost_1st = [3, 10, 3, 4, 4, -1, 6, 15, 18, 10, 3, 6, 4, 6, 8, 26, 9, 10, 17, 4]
multiagent_main_success = [True, True, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [3.0, 9, 3.0, 4, 4, 1000000.0, 6.0, 13.0, 15.0, 10, 3, 6.0, 4.0, 6, 7.0, 22.0, 8, 9, 14.0, 4.0]



singleagent, main, helper, singleagent_planning_time, singleagent_cost, multiagent_total_planning_time, overall_plan_length = get_multi_agent(singleagent, main, helper, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_planning_time, multiagent_main_planning_time_opt, multiagent_main_planning_time_1st, multiagent_main_cost, multiagent_main_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

print(np.array(helper).mean(axis=0).round(6))
print(np.array(helper).std(axis=0).round(3))

print(np.array(main).mean(axis=0).round(6))
print(np.array(main).std(axis=0).round(3))


# # plt.plot(singleagent_planning_time, '-r') #, figure=fig_barman_time)
# plt.plot(singleagent_cost, '-r') #, figure=fig_barman_length)
# # plt.plot(multiagent_total_planning_time, '-g') #, figure=fig_barman_time)
# plt.plot(overall_plan_length, '-g') #, figure=fig_barman_length)

################################################
print('multi-agent planning results')
print('barman-multi')
singleagent = []
helper = []
main = []

# singleagent_planning_time = [200.26, 200.25, 200.26, 200.24, 200.3, 200.27, 200.28, 200.31, 200.3, 200.34, 200.34, 200.32, 200.32, 200.4, 200.42, 200.42, 200.45, 200.4, 200.46, 200.45]
# singleagent_planning_time_opt = [7.67051, 11.102, 7.14282, 8.07458, 27.9873, 27.5297, 17.6855, 22.347, 190.227, 182.91, 0.111966, 181.649, 0.167951, 0.431891, 0.111633, 1.48774, 173.274, 0.387911, 0.327921, 0.19995]
# singleagent_cost = [27.0, 27.0, 27.0, 27.0, 37.0, 37.0, 38.0, 39.0, 47.0, 48.0, 59.0, 48.0, 60.0, 64.0, 60.0, 71.0, 76.0, 75.0, 70.0, 73.0]
# singleagent_planning_time_1st = [0.0119877, 0.00800222, 0.0119753, 0.00796605, 0.0199845, 0.0199868, 0.0160937, 0.0119771, 0.0279945, 0.03998, 0.0199959, 0.0439768, 0.0479764, 0.0439871, 0.0483231, 0.0639871, 0.103967, 0.0919629, 0.0599749, 0.0439925]
# singleagent_cost_1st = [41, 41, 41, 40, 57, 57, 40, 51, 65, 63, 62, 63, 84, 74, 86, 100, 108, 83, 91, 75]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [24.0, 24.0, 24.0, 24.0, 32.0, 32.0, 32.0, 32.0, 40.0, 40.0, 48.0, 40.0, 48.0, 52.0, 48.0, 56.0, 62.0, 61.0, 56.0, 58.0]



# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)


# singleagent_planning_time = [200.31, 200.25, 200.27, 200.24, 200.28, 200.24, 200.26, 200.28, 200.36, 200.33, 200.33, 200.34, 200.41, 200.41, 200.4, 200.41, 200.44, 200.45, 200.44, 200.46]
# singleagent_planning_time_opt = [7.14668, 10.8779, 7.11888, 7.98259, 27.823, 27.9355, 17.3528, 22.6279, 192.704, 182.022, 0.111959, 179.879, 0.16396, 0.4359, 0.12793, 1.59567, 171.717, 0.395917, 0.363867, 0.275925]
# singleagent_cost = [27.0, 27.0, 27.0, 27.0, 37.0, 37.0, 38.0, 39.0, 47.0, 48.0, 59.0, 48.0, 60.0, 64.0, 60.0, 71.0, 76.0, 75.0, 70.0, 73.0]
# singleagent_planning_time_1st = [0.00799073, 0.00800317, 0.0119703, 0.0119669, 0.0199598, 0.0199887, 0.0159869, 0.0119949, 0.0279923, 0.0279951, 0.0199975, 0.0439859, 0.0439804, 0.0439857, 0.0599843, 0.0639691, 0.10397, 0.0919703, 0.059984, 0.0599811]
# singleagent_cost_1st = [41, 41, 41, 40, 57, 57, 40, 51, 65, 63, 62, 63, 84, 74, 86, 100, 108, 83, 91, 75]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [24.0, 24.0, 24.0, 24.0, 32.0, 32.0, 32.0, 32.0, 40.0, 40.0, 48.0, 40.0, 48.0, 52.0, 48.0, 56.0, 62.0, 61.0, 56.0, 58.0]



# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [200.31, 200.25, 200.24, 200.26, 200.3, 200.29, 200.21, 200.23, 200.32, 200.35, 200.34, 200.32, 200.42, 200.42, 200.4, 200.41, 200.45, 200.46, 200.36, 200.48]
# singleagent_planning_time_opt = [7.71023, 10.1938, 7.15881, 8.10622, 28.0114, 27.8754, 17.5842, 22.6324, 188.548, 181.843, 0.111958, 184.323, 0.119977, 0.435909, 0.147965, 1.46376, 171.711, 0.319902, 0.235949, 0.275929]
# singleagent_cost = [27.0, 27.0, 27.0, 27.0, 37.0, 37.0, 38.0, 39.0, 47.0, 48.0, 59.0, 48.0, 60.0, 64.0, 60.0, 71.0, 76.0, 75.0, 70.0, 73.0]
# singleagent_planning_time_1st = [0.0119925, 0.0119945, 0.0119814, 0.00797467, 0.0199876, 0.0239711, 0.0159889, 0.0119962, 0.0239919, 0.039981, 0.0199962, 0.0439896, 0.0319899, 0.0439851, 0.0599806, 0.0639839, 0.107972, 0.0919596, 0.0399899, 0.0599743]
# singleagent_cost_1st = [41, 41, 41, 40, 57, 57, 40, 51, 65, 63, 62, 63, 84, 74, 86, 100, 108, 83, 91, 75]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [24.0, 24.0, 24.0, 24.0, 32.0, 32.0, 32.0, 32.0, 40.0, 40.0, 48.0, 40.0, 48.0, 52.0, 48.0, 56.0, 62.0, 61.0, 56.0, 58.0]

singleagent_planning_time = [1000.26, 1000.23, 1000.21, 1000.26, 1000.26, 1000.16, 1000.25, 1000.27, 1000.36, 1000.03, 999.91, 1000.37, 1000.46, 1000.43, 1000.46, 1000.43, 1000.34, 1000.45, 1000.49, 1000.46]
singleagent_planning_time_opt = [8.05831, 10.8068, 6.87493, 7.91072, 203.098, 203.412, 215.721, 257.348, 190.674, 182.315, 388.701, 181.053, 0.159964, 976.318, 0.147953, 1.45572, 171.605, 0.279949, 0.327914, 0.279913]
singleagent_cost = [27.0, 27.0, 27.0, 27.0, 36.0, 36.0, 36.0, 36.0, 47.0, 48.0, 45.0, 48.0, 60.0, 58.0, 60.0, 71.0, 76.0, 75.0, 70.0, 73.0]
singleagent_planning_time_1st = [0.00800833, 0.0119843, 0.0119852, 0.00800172, 0.019966, 0.0199883, 0.0159864, 0.0119752, 0.0279906, 0.0279731, 0.0159967, 0.043981, 0.047984, 0.043984, 0.0599823, 0.0679735, 0.103972, 0.0639816, 0.0559808, 0.0599827]
singleagent_cost_1st = [41, 41, 41, 40, 57, 57, 40, 51, 65, 63, 62, 63, 84, 74, 86, 100, 108, 83, 91, 75]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [24.0, 24.0, 24.0, 24.0, 32.0, 32.0, 32.0, 32.0, 40.0, 40.0, 40.0, 40.0, 48.0, 48.0, 48.0, 56.0, 62.0, 61.0, 56.0, 58.0]


singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

plt.plot(singleagent_planning_time, '--y')#, figure=fig_barman_time)
# plt.plot(singleagent_cost, '--y', figure=fig_barman_length)
# plt.plot(overall_plan_length, '--y') #, figure=fig_barman_length)
plt.savefig('time_barman.png') #, figure=fig_barman_time)
# plt.savefig('length_barman.png') #, figure=fig_barman_length)

print('blocksworld-multi')
singleagent = []
helper = []
main = []


# singleagent_planning_time = [0.18, 0.16, 0.14, 0.18, 0.15, 0.18, 0.19, 0.33, 5.03, 3.06, 11.02, 31.41, 200.16, 200.15, 200.2, 200.19, 200.19, 200.26, 200.25, 200.25]
# singleagent_planning_time_opt = [0.00400134, 0.00399973, 0.00800537, 0.015991, 0.0040317, 0.0319946, 0.0279637, 0.167915, 4.86302, 2.9074, 10.89, 31.2342, 0.0919725, 0.395907, 0.0119966, 0.0279431, 0.327931, 0.215939, 0.0359908, 0.0999407]
# singleagent_cost = [6.0, 6.0, 6.0, 10.0, 5.0, 10.0, 7.0, 12.0, 14.0, 14.0, 18.0, 16.0, 24.0, 20.0, 22.0, 24.0, 21.0, 26.0, 22.0, 28.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.00398871, 0.0, 0.0, 0.00399998, 0.0, 0.0, 0.00399911, 0.0, 0.0, 0.00432395, 0.00399992, 0.0, 0.0, 0.0, 0.00400596]
# singleagent_cost_1st = [6, 6, 6, 10, 5, 10, 9, 12, 22, 24, 28, 28, 30, 24, 34, 50, 24, 38, 34, 42]

# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [6, 6, 6, 8.0, 4.0, 9.0, 7.0, 10.0, 12.0, 11.0, 14.0, 16.0, 18.0, 16.0, 18.0, 20.0, 17.0, 26.0, 19.0, 25.0]

# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.15, 0.15, 0.15, 0.15, 0.18, 0.19, 0.18, 0.32, 5.3, 3.13, 11.46, 31.91, 200.28, 200.18, 200.17, 200.2, 200.23, 200.21, 200.27, 200.24]
# singleagent_planning_time_opt = [0.00800713, 0.00797768, 0.00799015, 0.0160106, 0.0040003, 0.0319657, 0.027949, 0.163911, 5.13018, 2.9793, 11.2938, 31.7737, 0.0919691, 0.291944, 0.0119957, 0.0239112, 0.323937, 0.20389, 0.0359901, 0.0959533]
# singleagent_cost = [6.0, 6.0, 6.0, 10.0, 5.0, 10.0, 7.0, 12.0, 14.0, 14.0, 18.0, 16.0, 24.0, 20.0, 22.0, 24.0, 21.0, 26.0, 22.0, 28.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.00400285, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0040003, 0.00399955, 0.00399049, 0.0, 0.00399978, 0.00398821, 0.00400413, 0.0, 0.00399119, 0.0, 0.00799861]
# singleagent_cost_1st = [6, 6, 6, 10, 5, 10, 9, 12, 22, 24, 28, 28, 30, 24, 34, 50, 24, 38, 34, 42]


# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [6, 6, 6, 8.0, 4.0, 9.0, 7.0, 10.0, 12.0, 11.0, 14.0, 16.0, 18.0, 16.0, 18.0, 20.0, 17.0, 26.0, 19.0, 25.0]


# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)


# singleagent_planning_time = [0.2, 0.16, 0.14, 0.17, 0.17, 0.21, 0.2, 0.31, 5.06, 2.82, 11.21, 31.67, 200.15, 200.15, 200.18, 200.21, 200.25, 200.24, 200.22, 200.24]
# singleagent_planning_time_opt = [0.00800265, 0.0040022, 0.00800304, 0.0119888, 0.00402523, 0.0319275, 0.0319783, 0.163923, 4.91098, 2.68737, 11.0698, 31.5021, 0.067977, 0.283654, 0.0119939, 0.0239514, 0.32794, 0.215932, 0.0319883, 0.135934]
# singleagent_cost = [6.0, 6.0, 6.0, 10.0, 5.0, 10.0, 7.0, 12.0, 14.0, 14.0, 18.0, 16.0, 24.0, 20.0, 22.0, 24.0, 21.0, 26.0, 22.0, 28.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400116, 0.0, 0.00398106, 0.0, 0.0, 0.00399994, 0.0, 0.00400343, 0.0, 0.0, 0.00400627, 0.0, 0.00396934]
# singleagent_cost_1st = [6, 6, 6, 10, 5, 10, 9, 12, 22, 24, 28, 28, 30, 24, 34, 50, 24, 38, 34, 42]

# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [6, 6, 6, 8.0, 4.0, 9.0, 7.0, 10.0, 12.0, 11.0, 14.0, 16.0, 18.0, 16.0, 18.0, 20.0, 17.0, 26.0, 19.0, 25.0]

singleagent_planning_time = [0.16, 0.13, 0.13, 0.13, 0.14, 0.17, 0.18, 0.28, 5.13, 3.11, 11.79, 33.7, 265.24, 529.85, 434.89, 1000.24, 1000.35, 1000.36, 1000.5, 1000.46]
singleagent_planning_time_opt = [0.00400488, 0.00400008, 0.00398624, 0.0159503, 0.00399959, 0.0279742, 0.0279795, 0.139942, 4.98276, 2.96331, 11.6295, 33.5481, 265.037, 529.656, 434.613, 0.0239738, 0.323926, 0.215938, 0.0359857, 0.139955]
singleagent_cost = [6.0, 6.0, 6.0, 10.0, 5.0, 10.0, 7.0, 12.0, 14.0, 14.0, 18.0, 16.0, 24.0, 20.0, 22.0, 24.0, 21.0, 26.0, 22.0, 28.0]
singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.00397665, 0.0, 0.0, 0.0, 0.0, 0.00398255, 0.00399918, 0.00400011, 0.0, 0.00399986, 0.0, 0.0040036, 0.00398189, 0.0, 0.0, 0.0, 0.00799074]
singleagent_cost_1st = [6, 6, 6, 10, 5, 10, 9, 12, 22, 24, 28, 28, 30, 24, 34, 50, 24, 38, 34, 42]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [6, 6, 6, 8.0, 4.0, 9.0, 7.0, 10.0, 12.0, 11.0, 14.0, 16.0, 18.0, 16.0, 18.0, 20.0, 17.0, 26.0, 19.0, 25.0]


singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

# # plt.plot(singleagent_planning_time, '--y')#, figure=fig_barman_time)
# # plt.plot(singleagent_cost, '--y', figure=fig_barman_length)
# plt.plot(overall_plan_length, '--y') #, figure=fig_barman_length)
# # plt.savefig('time_barman.png') #, figure=fig_barman_time)
# plt.savefig('length_blocks.png') #, figure=fig_barman_length)



print('termes-multi')
singleagent = []
helper = []
main = []

# singleagent_planning_time = [200.23, 200.21, 200.25, 200.24, 200.0, 200.0, 200.32, 200.33, 200.42, 200.4, 200.2, 200.22, 200.26, 200.27, 200.31, 200.33, 200.18, 200.22, 200.27, 200.27]
# singleagent_planning_time_opt = [1.36754, 47.4858, 95.3146, 151.115, 200.0, 200.0, 110.17, 173.554, 87.1852, 73.4853, 77.6457, 3.85924, 57.4104, 118.232, 34.7534, 13.7805, 61.5795, 50.7372, 30.566, 135.92]
# singleagent_cost = [32.0, 50.0, 68.0, 76.0, 0, 0, 52.0, 58.0, 260.0, 262.0, 58.0, 41.0, 104.0, 107.0, 320.0, 526.0, 116.0, 76.0, 90.0, 120.0]
# singleagent_planning_time_1st = [0.107899, 0.255901, 8.12214, 5.57827, 200.0, 200.0, 2.95516, 1.05964, 87.1852, 73.4853, 0.223885, 0.0599672, 4.85502, 2.25125, 34.7534, 13.7805, 0.755768, 0.167934, 0.787719, 1.82362]
# singleagent_cost_1st = [68, 132, 150, 280, 0, 0, 170, 154, 260, 262, 222, 85, 216, 326, 320, 526, 352, 237, 372, 417]
# multiagent_main_success = [False]*20
# overall_plan_length = [0]*20

# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [200.25, 200.18, 200.25, 200.26, 200.0, 200.0, 200.34, 200.35, 200.41, 200.4, 200.16, 200.22, 200.26, 200.24, 200.31, 200.33, 200.24, 200.22, 200.26, 200.27]
# singleagent_planning_time_opt = [1.37554, 49.4477, 94.236, 151.835, 200.0, 200.0, 108.583, 173.062, 87.8792, 73.5026, 77.624, 4.13916, 55.8124, 118.136, 34.7254, 13.2453, 62.1188, 52.9004, 30.6542, 136.748]
# singleagent_cost = [32.0, 50.0, 68.0, 76.0, 0, 0, 52.0, 58.0, 260.0, 262.0, 58.0, 41.0, 104.0, 107.0, 320.0, 526.0, 116.0, 76.0, 90.0, 120.0]
# singleagent_planning_time_1st = [0.107907, 0.255836, 8.14209, 5.62979, 200.0, 200.0, 2.95519, 1.0597, 87.8792, 73.5026, 0.183925, 0.0759535, 4.69907, 2.24722, 34.7254, 13.2453, 0.755779, 0.167931, 0.7838, 2.23129]
# singleagent_cost_1st = [68, 132, 150, 280, 0, 0, 170, 154, 260, 262, 222, 85, 216, 326, 320, 526, 352, 237, 372, 417]


# multiagent_main_success = [False]*20
# overall_plan_length = [0]*20

# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [200.23, 200.2, 200.25, 200.25, 200.0, 200.0, 200.32, 200.34, 200.39, 200.4, 200.2, 200.22, 200.25, 200.24, 200.25, 200.34, 200.22, 200.22, 200.25, 200.26]
# singleagent_planning_time_opt = [1.37154, 47.314, 94.1869, 153.286, 200.0, 200.0, 110.305, 175.535, 87.7634, 73.3268, 77.6156, 4.1592, 56.7788, 117.387, 34.6614, 13.2451, 61.4942, 51.0934, 29.9303, 136.209]
# singleagent_cost = [32.0, 50.0, 68.0, 76.0, 0, 0, 52.0, 58.0, 260.0, 262.0, 58.0, 41.0, 104.0, 107.0, 320.0, 526.0, 116.0, 76.0, 90.0, 120.0]
# singleagent_planning_time_1st = [0.107927, 0.251888, 8.55361, 5.60247, 200.0, 200.0, 2.96289, 1.07566, 87.7634, 73.3268, 0.223932, 0.0759461, 4.29513, 2.24328, 34.6614, 13.2451, 0.747802, 0.171899, 0.647842, 2.21149]
# singleagent_cost_1st = [68, 132, 150, 280, 0, 0, 170, 154, 260, 262, 222, 85, 216, 326, 320, 526, 352, 237, 372, 417]


# multiagent_main_success = [False]*20
# overall_plan_length = [0]*20
singleagent_planning_time = [513.45, 1000.22, 1000.17, 1000.15, 1000.27, 1000.27, 1000.25, 1000.29, 1000.38, 1000.44, 997.6, 822.6, 1000.2, 1000.26, 1000.32, 1000.33, 1000.18, 1000.15, 1000.19, 1000.24]
singleagent_planning_time_opt = [513.174, 42.4402, 225.839, 394.403, 178.164, 220.733, 562.832, 148.014, 73.9598, 496.297, 100.476, 822.303, 669.502, 102.406, 29.3514, 10.8943, 705.14, 257.917, 277.765, 115.928]
singleagent_cost = [32.0, 50.0, 64.0, 72.0, 284.0, 314.0, 50.0, 58.0, 260.0, 72.0, 58.0, 41.0, 78.0, 107.0, 320.0, 526.0, 103.0, 70.0, 86.0, 120.0]
singleagent_planning_time_1st = [0.107934, 0.187949, 6.60682, 4.45915, 178.164, 220.733, 2.38755, 1.05578, 73.9598, 62.1939, 0.314694, 0.0760668, 3.89846, 1.95561, 29.3514, 10.8943, 0.73183, 0.119968, 0.783831, 1.77969]
singleagent_cost_1st = [68, 132, 150, 280, 284, 314, 170, 154, 260, 262, 222, 85, 216, 326, 320, 526, 352, 237, 372, 417]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [25.0, 38.0, 52.0, 53.0, 210.0, 225.0, 33.0, 42.0, 188.0, 56.0, 44.0, 37.0, 66.0, 78.0, 242.0, 402.0, 76.0, 59.0, 62.0, 93.0]


singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

# # plt.plot(singleagent_planning_time, '--y')#, figure=fig_barman_time)
# # plt.plot(singleagent_cost, '--y', figure=fig_barman_length)
# plt.plot(overall_plan_length, '--y') #, figure=fig_barman_length)
# # plt.savefig('time_barman.png') #, figure=fig_barman_time)
# plt.savefig('length_temres.png') #, figure=fig_barman_length)


print('tyreworld-multi')
singleagent = []
helper = []
main = []

# singleagent_planning_time = [0.31, 200.41, 200.26, 200.31, 200.37, 200.38, 200.32, 200.36, 200.44, 200.49, 200.43, 200.51, 200.56, 200.56, 200.47, 200.64, 200.67, 200.67, 200.67, 200.86]
# singleagent_planning_time_opt = [0.115911, 0.191868, 0.0040051, 0.0119951, 0.023979, 0.0319788, 0.0639668, 0.103962, 0.151965, 0.187926, 0.275925, 0.383898, 0.519841, 0.691818, 0.947699, 0.927734, 1.2197, 1.54761, 1.85946, 2.0794]
# singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
# singleagent_planning_time_1st = [0.00399884, 0.191868, 0.0040051, 0.0119951, 0.023979, 0.0319788, 0.0639668, 0.103962, 0.151965, 0.187926, 0.275925, 0.383898, 0.519841, 0.691818, 0.947699, 0.927734, 1.2197, 1.54761, 1.85946, 2.0794]
# singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]

# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.28, 200.55, 200.27, 200.27, 200.4, 200.42, 200.34, 200.36, 200.41, 200.4, 200.43, 200.52, 200.49, 200.52, 200.5, 200.69, 200.74, 200.79, 200.71, 200.93]
# singleagent_planning_time_opt = [0.111904, 0.163872, 0.00399965, 0.00800209, 0.0159866, 0.0279914, 0.0519586, 0.103932, 0.155947, 0.139935, 0.195956, 0.283926, 0.519859, 0.50788, 0.863797, 1.09972, 1.09177, 1.3557, 1.69149, 2.21952]
# singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
# singleagent_planning_time_1st = [0.0, 0.163872, 0.00399965, 0.00800209, 0.0159866, 0.0279914, 0.0519586, 0.103932, 0.155947, 0.139935, 0.195956, 0.283926, 0.519859, 0.50788, 0.863797, 1.09972, 1.09177, 1.3557, 1.69149, 2.21952]
# singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]

# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]


# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.31, 200.53, 200.28, 200.32, 200.37, 200.36, 200.36, 200.35, 200.44, 200.42, 200.45, 200.54, 200.49, 200.46, 200.63, 200.61, 200.74, 200.78, 200.77, 200.93]
# singleagent_planning_time_opt = [0.119863, 0.155852, 0.00400178, 0.0080119, 0.0239748, 0.0319772, 0.0639797, 0.0839615, 0.111942, 0.13996, 0.267883, 0.383896, 0.515862, 0.49588, 0.959727, 1.05156, 1.08368, 1.39951, 1.91531, 2.31146]
# singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
# singleagent_planning_time_1st = [0.0, 0.155852, 0.00400178, 0.0080119, 0.0239748, 0.0319772, 0.0639797, 0.0839615, 0.111942, 0.13996, 0.267883, 0.383896, 0.515862, 0.49588, 0.959727, 1.05156, 1.08368, 1.39951, 1.91531, 2.31146]
# singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]


singleagent_planning_time = [0.27, 1000.79, 1000.69, 1001.09, 1001.12, 1001.02, 1000.9, 1000.92, 1000.96, 1001.0, 999.83, 996.55, 1000.9, 1001.01, 1001.11, 1000.95, 1001.22, 1001.01, 1001.21, 1001.31]
singleagent_planning_time_opt = [0.115923, 0.163906, 0.00800231, 0.00797744, 0.0239664, 0.0399552, 0.0639613, 0.107915, 0.151956, 0.187946, 0.279917, 0.379897, 0.51985, 0.707799, 0.843776, 1.02373, 1.31568, 1.51162, 1.96749, 2.11153]
singleagent_cost = [13.0, 101.0, 35.0, 46.0, 57.0, 68.0, 79.0, 90.0, 101.0, 112.0, 123.0, 134.0, 145.0, 156.0, 167.0, 178.0, 189.0, 200.0, 211.0, 222.0]
singleagent_planning_time_1st = [0.0, 0.163906, 0.00800231, 0.00797744, 0.0239664, 0.0399552, 0.0639613, 0.107915, 0.151956, 0.187946, 0.279917, 0.379897, 0.51985, 0.707799, 0.843776, 1.02373, 1.31568, 1.51162, 1.96749, 2.11153]
singleagent_cost_1st = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [13, 101, 35, 46, 57, 68, 79, 90, 101, 112, 123, 134, 145, 156, 167, 178, 189, 200, 211, 222]


singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

# # plt.plot(singleagent_planning_time, '--y')#, figure=fig_barman_time)
# # plt.plot(singleagent_cost, '--y', figure=fig_barman_length)
# plt.plot(overall_plan_length, '--y') #, figure=fig_barman_length)
# # plt.savefig('time_barman.png') #, figure=fig_barman_time)
# plt.savefig('length_tyreworld.png') #, figure=fig_barman_length)


print('grippers-multi')
singleagent = []
helper = []
main = []

# singleagent_planning_time = [0.13, 0.31, 0.2, 0.17, 0.15, 0.15, 0.27, 0.66, 200.11, 0.22, 0.11, 0.72, 0.14, 0.18, 0.51, 200.18, 0.3, 1.99, 200.13, 0.15]
# singleagent_planning_time_opt = [0.00797545, 0.167843, 0.0399654, 0.0159666, 0.00400117, 0.004, 0.103894, 0.531805, 166.675, 0.107943, 0.00399999, 0.59585, 0.0079829, 0.0519686, 0.38787, 0.423874, 0.175952, 1.85964, 136.301, 0.0120173]
# singleagent_cost = [5.0, 9.0, 6.0, 4.0, 3.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 5.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00397671, 0.00399991, 0.0, 0.0, 0.0, 0.0]
# singleagent_cost_1st = [5, 11, 6, 4, 3, 4, 11, 11, 18, 10, 3, 10, 6, 5, 11, 21, 8, 9, 19, 7]
# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [5, 9, 6, 4, 3, 4, 8, 8.0, 11.0, 6.0, 3, 6.0, 6, 5, 10, 11.0, 8, 9, 11.0, 7]

# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.16, 0.33, 0.19, 0.16, 0.16, 0.14, 0.24, 0.67, 200.09, 0.21, 0.12, 0.74, 0.13, 0.21, 0.52, 200.17, 0.28, 1.97, 200.09, 0.12]
# singleagent_planning_time_opt = [0.00800504, 0.183846, 0.0439513, 0.00797363, 0.00400058, 0.00400696, 0.107931, 0.527849, 173.136, 0.0839415, 0.00800898, 0.607846, 0.00797313, 0.0639258, 0.383881, 0.323527, 0.179925, 1.83958, 133.633, 0.0120084]
# singleagent_cost = [5.0, 9.0, 6.0, 4.0, 3.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 5.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.00400303, 0.0, 0.0, 0.00400696, 0.0, 0.00397707, 0.0, 0.0039766, 0.0, 0.00400608, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# singleagent_cost_1st = [5, 11, 6, 4, 3, 4, 11, 11, 18, 10, 3, 10, 6, 5, 11, 21, 8, 9, 19, 7]

# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [5, 9, 6, 4, 3, 4, 8, 8.0, 11.0, 6.0, 3, 6.0, 6, 5, 10, 11.0, 8, 9, 11.0, 7]

# singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

# singleagent_planning_time = [0.16, 0.32, 0.19, 0.15, 0.15, 0.14, 0.23, 0.66, 200.14, 0.2, 0.11, 0.76, 0.12, 0.19, 0.63, 200.18, 0.3, 1.91, 200.11, 0.12]
# singleagent_planning_time_opt = [0.00800604, 0.17187, 0.0439256, 0.00800284, 0.00400351, 0.00398022, 0.0999446, 0.527848, 174.19, 0.0879381, 0.00399982, 0.639614, 0.0119782, 0.0519681, 0.503864, 0.311904, 0.17988, 1.79559, 136.403, 0.0119918]
# singleagent_cost = [5.0, 9.0, 6.0, 4.0, 3.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 5.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
# singleagent_planning_time_1st = [0.0, 0.0, 0.00400297, 0.0, 0.0, 0.00398022, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0039969, 0.0, 0.0, 0.0, 0.0]
# singleagent_cost_1st = [5, 11, 6, 4, 3, 4, 11, 11, 18, 10, 3, 10, 6, 5, 11, 21, 8, 9, 19, 7]

# multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
# overall_plan_length = [5, 9, 6, 4, 3, 4, 8, 8.0, 11.0, 6.0, 3, 6.0, 6, 5, 10, 11.0, 8, 9, 11.0, 7]

singleagent_planning_time = [0.14, 0.31, 0.18, 0.16, 0.14, 0.12, 0.21, 0.66, 486.26, 0.23, 0.14, 0.94, 0.12, 0.21, 0.53, 1000.24, 0.36, 2.61, 249.67, 0.16]
singleagent_planning_time_opt = [0.00797785, 0.167916, 0.0479341, 0.0119899, 0.00399979, 0.00397745, 0.0959325, 0.523852, 486.08, 0.0999473, 0.00399979, 0.79983, 0.0080034, 0.0679634, 0.387819, 0.419876, 0.215925, 2.47941, 249.504, 0.0160103]
singleagent_cost = [5.0, 9.0, 6.0, 4.0, 3.0, 4.0, 8.0, 11.0, 17.0, 10.0, 3.0, 9.0, 6.0, 5.0, 10.0, 19.0, 8.0, 9.0, 17.0, 7.0]
singleagent_planning_time_1st = [0.00397799, 0.0, 0.0, 0.0, 0.0, 0.00397745, 0.0, 0.0, 0.00400027, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00400008, 0.0, 0.0, 0.00397561, 0.0]
singleagent_cost_1st = [5, 11, 6, 4, 3, 4, 11, 11, 18, 10, 3, 10, 6, 5, 11, 21, 8, 9, 19, 7]
multiagent_main_success = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
overall_plan_length = [5, 9, 6, 4, 3, 4, 8, 8.0, 11.0, 6.0, 3, 6.0, 6, 5, 10, 11.0, 8, 9, 11.0, 7]


singleagent, singleagent_planning_time, singleagent_cost, overall_plan_length = get_single_agent(singleagent, singleagent_planning_time, singleagent_planning_time_opt, singleagent_planning_time_1st, singleagent_cost, singleagent_cost_1st, multiagent_main_success, overall_plan_length)

print(np.array(singleagent).mean(axis=0).round(6))
print(np.array(singleagent).std(axis=0).round(3))

# # plt.plot(singleagent_planning_time, '--y')#, figure=fig_barman_time)
# # plt.plot(singleagent_cost, '--y', figure=fig_barman_length)
# plt.plot(overall_plan_length, '--y') #, figure=fig_barman_length)
# # plt.savefig('time_barman.png') #, figure=fig_barman_time)
# plt.savefig('length_grippers.png') #, figure=fig_barman_length)