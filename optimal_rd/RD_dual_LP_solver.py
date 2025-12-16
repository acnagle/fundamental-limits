import numpy as np
import scipy.spatial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json
import os
import gzip
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import wandb
from itertools import combinations


#############################################################
## code to solve (dual-LP) using Algorithm 1, to obtain the 
## distortion-rate function
#############################################################
## notation is different from that used in the paper: 
# we have (p,x,y) for prompt, query, output instead of (x,q,y) as in paper,
# the constants R_x and D_x are given by L_p and D_p.

############### functions to set up (dual-LP) ###############

## experiment name, wandb. 
def get_exp_name(args):
    if args.condition == 0:
        exp_name = f"uncond_{args.data_path}"
    else:
        exp_name = f"cond_{args.data_path}"
    return exp_name

## load the dataset from data_path
def load_dataset(data_path):
    full_path = os.path.join('datasets', data_path)
    dataset = []
    with open(full_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

## get unique query values from the dataset
def get_calX(dataset):
    calX = {}
    for entry in dataset:
        x_val = entry["query"]
        if x_val in calX:
            calX[x_val] = calX[x_val] + 1
        else:
            calX[x_val] = 1
    return calX

## get unique prompt values from the dataset
def get_calP(dataset, query=None):
    calP = {}
    for entry in dataset:
        if (entry["query"] == query) or (query == None):
            p_val = entry["context"]
            if p_val in calP:
                count, calM_p = calP[p_val]
                calP[p_val] = (count + 1, calM_p)
            else:
                # calP[p_val] = (1, generate_substrings(p_val))           # calM_p = pruned strings
                calP[p_val] = (1, list(entry["context_dict"].keys())) # calM_p = all shorter strings
    return calP

## get unique (priompt, query) values from the dataset
def get_calPX(dataset):
    calPX = {}
    for entry in dataset:
        px_val = (entry["context"], entry["query"])
        if px_val in calPX:
            count, calM_x = calPX[px_val]
            calPX[px_val] = (count + 1, calM_x)
        else:
            # calPX[px_val] = (1, generate_substrings(entry["context"]))  # calM_p = pruned strings
            calPX[px_val] = (1, list(entry["context_dict"].keys()))   # calM_p = all shorter strings
    return calPX

## compute D_x and R_x (needed as input to Algorithm 1)
def compute_L_D_mp(dataset, count_p, calM_p, p, l_p, N, x=None):
    L_mp_vals = []
    D_mp_vals = []

    for m in calM_p:        # length of calM_p = 2^len(p) - 2

        if True: #(len(m) & (len(m)-1) == 0):   # for approximations! set to True to go over all shorter strings without approx
            L_mp = (len(m) / l_p) * (count_p / N)
            D_mp = 0
            for entry in dataset:
                if entry["context"] == p and ((entry["query"] == x) or (x == None)):
                    # _, curr_D_mp = entry["context_dict"].get(m)
                    curr_D_mp = entry["context_dict"].get(m)
                    D_mp += curr_D_mp            

            D_mp = D_mp / N
            
            L_mp_vals.append(L_mp)
            D_mp_vals.append(D_mp)

    D_pp = 0
    flag = 1
    for entry in dataset:
        if "loss" not in entry:
            flag = 0
            break
        if entry["context"] == p and ((entry["query"] == x) or (x == None)):
            D_pp += entry["loss"]    
    if flag == 1:        
        D_pp = D_pp / N

        L_mp_vals.append(count_p / N)
        D_mp_vals.append(D_pp)

    L_mp_vals = np.array(L_mp_vals)
    D_mp_vals = np.array(D_mp_vals)
    
    return L_mp_vals, D_mp_vals      # length 2^len(p) - 1

## compute D_xq and R_xq 
def compute_L_D_mpx(dataset, count_px, calM_p, p, x, l_p, N):
    L_mpx_vals = []
    D_mpx_vals = []

    for m in calM_p:        # length of calM_p = 2^len(p) - 2

        if True: #(len(m) & (len(m)-1) == 0):   # for approximations! set to True to go over all shorter strings without approx
            L_mpx = (len(m) / l_p) * (count_px / N)
            D_mpx = 0
            for entry in dataset:
                if entry["context"] == p and entry["query"] == x:
                    # _, curr_D_mpx = entry["context_dict"].get(m)
                    curr_D_mpx = entry["context_dict"].get(m)
                    D_mpx += curr_D_mpx           

            D_mpx = D_mpx / N
            
            L_mpx_vals.append(L_mpx)
            D_mpx_vals.append(D_mpx)

    D_ppx = 0
    flag = 1
    for entry in dataset:
        if "loss" not in entry:
            flag = 0
            break
        if entry["context"] == p and entry["query"] == x:
            D_ppx += entry["loss"] 
    if flag == 1:               
        D_ppx = D_ppx / N

        L_mpx_vals.append(count_px / N)
        D_mpx_vals.append(D_ppx)

    L_mpx_vals = np.array(L_mpx_vals)
    D_mpx_vals = np.array(D_mpx_vals)
    
    return L_mpx_vals, D_mpx_vals      # length 2^len(p) - 1


## lower-left boundary, as required in step 2 of the algorithm
def get_lower(polygon):
    minx = np.argmin(polygon[:, 0])
    maxx = np.argmax(polygon[:, 0]) + 1
    if minx >= maxx:
        lower_curve = np.concatenate([polygon[minx:], polygon[:maxx]])
    else:
        lower_curve = polygon[minx:maxx]
    return lower_curve

## to generate all ''subtrings'', to restrict calM_p to be just pruned versions
def generate_substrings(string):
    substrings = []
    n = len(string)
    for i in range(1, n):
        substrings.extend([''.join(comb) for comb in combinations(string, i)])
    substrings = list(set(substrings))
    return substrings

######### solving Algorithm 1 ###############

## steps 1--11 
def nearly_max_sum_min(L_vals_for_p, D_vals_for_p):
    l_calP = len(L_vals_for_p)
    L_env_vals_for_p = [np.zeros(0) for _ in range(l_calP)]
    D_env_vals_for_p = [np.zeros(0) for _ in range(l_calP)]
    lambda_vals_for_p = [np.zeros(0) for _ in range(l_calP)]
    
    for p_idx in range(l_calP):
        L_D_vals = np.transpose(np.vstack((L_vals_for_p[p_idx], D_vals_for_p[p_idx])))

        try:
            hull = scipy.spatial.ConvexHull(L_D_vals)
            L_D_env = get_lower(L_D_vals[hull.vertices])
        except Exception as e:
            L_D_env = L_D_vals
            
        if L_D_env[0,0] >= L_D_env[1,0]:
            L_D_env[1,1] = min(L_D_env[0,1], L_D_env[1,1])
            L_D_env = np.delete(L_D_env, 0, axis=0)
        
        flag = 1
        while flag:
            flag = 0
            if len(L_D_env) == 1:
                break
            if (L_D_env[-2, 0] - L_D_env[-1, 0]) * (L_D_env[-1,1] - L_D_env[-2,1]) <= 0:
                L_D_env = np.delete(L_D_env, -1, axis=0)
                flag = 1
        L_env_vals_for_p[p_idx] = L_D_env[:,0]
        D_env_vals_for_p[p_idx] = L_D_env[:,1]
        
#         print(L_env_vals_for_p[p_idx], D_env_vals_for_p[p_idx])

        lambda_val = np.zeros(len(L_env_vals_for_p[p_idx])-1)

        for j in range(len(lambda_val)):
            lambda_val[j] =  (L_D_env[j,1] - L_D_env[j+1,1]) / (L_D_env[j+1,0] - L_D_env[j,0])

        lambda_vals = np.sort(np.concatenate((lambda_val, np.zeros(1))))
        lambda_vals = lambda_vals[::-1]
        lambda_vals_for_p[p_idx] = lambda_vals
#         print(lambda_vals)

    Lambda = np.sort(np.unique(np.concatenate(lambda_vals_for_p)))
    Lambda = Lambda[::-1]
    
#     print(Lambda)

    D_vals_for_lambda = []
    L_vals_for_lambda = []
    idx_for_p = [0 for _ in range(l_calP)]

    for i in range(len(Lambda)):
        lambda_val = Lambda[i]
        D_val = 0
        L_val = 0
        for p_idx in range(l_calP):
#             print(lambda_val)
#             print(idx_for_p)
            if lambda_val < (lambda_vals_for_p[p_idx])[(idx_for_p[p_idx])]:
                idx_for_p[p_idx] += 1
            D_val += (D_env_vals_for_p[p_idx])[(idx_for_p[p_idx])]
            L_val += (L_env_vals_for_p[p_idx])[(idx_for_p[p_idx])]
#             print(L_val, D_val)
#             print("===")
#         print("======")
        D_vals_for_lambda.append(D_val)
        L_vals_for_lambda.append(L_val)
        
    return D_vals_for_lambda, L_vals_for_lambda, Lambda

## steps 12--18 
def max_term(D_vals, L_vals, Lambda, rate_val):
    if L_vals[0] >= rate_val:
        return np.inf
    k = len(D_vals)
    max_val = 0
    for j in range(k):
        curr_term = D_vals[j] 
        if L_vals[j] >= rate_val:
            curr_term += (Lambda[j-1] * (L_vals[j] - rate_val))
        else:
            curr_term += (Lambda[j] * (L_vals[j] - rate_val))
        if curr_term > max_val:
            max_val = curr_term
    return max_val


######### Main ###############

## Main function
def main(data_path, rate_vals, condition):
    dataset = load_dataset(data_path)
    file_name = os.path.splitext(os.path.basename(data_path))[0]
 
    if condition == 0 or condition == 3:
        print("Query-agnostic.")
        print("---------------------------------------------")
        calP = get_calP(dataset)
        L_mp_vals_for_p = [np.zeros(5) for _ in range(len(calP))]
        D_mp_vals_for_p = [np.zeros(5) for _ in range(len(calP))]

        p_idx = 0
        for p in calP.keys():
            count_p, calM_p = calP.get(p)
            l_p = len(p)
            L_mp_vals, D_mp_vals = compute_L_D_mp(dataset, count_p, calM_p, p, l_p, len(dataset))
            L_mp_vals_for_p[p_idx] = L_mp_vals
            D_mp_vals_for_p[p_idx] = D_mp_vals
            p_idx += 1

        D_vals_for_lambda, L_vals_for_lambda, Lambda = nearly_max_sum_min(L_mp_vals_for_p, D_mp_vals_for_p)

        D_vals = []

        for r in rate_vals:
            D_val = max_term(D_vals_for_lambda, L_vals_for_lambda, Lambda, r)
            D_vals.append(D_val)
            # print("R: ", r, ", D: ", D_val)
            # wandb.log({"R": r, "D": D_val})

        print("Rate: ")
        print(rate_vals)
        print("Distortion: ")
        print(D_vals)
        print("---------------------------------------------")

    if condition == 1 or condition == 3:
        print("Query-aware, average.")
        print("---------------------------------------------")
        calPX = get_calPX(dataset)
        L_mpx_vals_for_px = [np.zeros(5) for _ in range(len(calPX))]
        D_mpx_vals_for_px = [np.zeros(5) for _ in range(len(calPX))]

        px_idx = 0
        for p, x in calPX.keys():
            count_px, calM_p = calPX.get((p, x))
            l_p = len(p)
            L_mpx_vals, D_mpx_vals = compute_L_D_mpx(dataset, count_px, calM_p, p, x, l_p, len(dataset))
            L_mpx_vals_for_px[px_idx] = L_mpx_vals
            D_mpx_vals_for_px[px_idx] = D_mpx_vals
            px_idx += 1

        D_vals_for_lambda, L_vals_for_lambda, Lambda = nearly_max_sum_min(L_mpx_vals_for_px, D_mpx_vals_for_px)

        cond_D_vals = []

        for r in rate_vals:
            D_val = max_term(D_vals_for_lambda, L_vals_for_lambda, Lambda, r)
            cond_D_vals.append(D_val)
            # print("R: ", r, ", D: ", D_val)
            # wandb.log({"R": r, "D": D_val})

        print("Rate: ")
        print(rate_vals)
        print("Distortion: ")
        print(cond_D_vals)
        print("---------------------------------------------")


    if condition == 2 or condition == 3:
        if condition == 3:
            output_file = {}
            output_file[data_path] = {
                "unconditional": {
                    "rate": rate_vals,
                    "distortion": D_vals
                },
                "conditional_avg": {
                    "rate": rate_vals,
                    "distortion": cond_D_vals
                },
                "conditional": {}
            }

        calX = get_calX(dataset)
        print("Query-aware, per query.")
        print("---------------------------------------------")
        print("\"queries\": [")
        for query in calX.keys():
            count_x = calX.get(query)
            calP = get_calP(dataset, query=query)
            L_mp_vals_for_p = [np.zeros(1) for _ in range(len(calP))]
            D_mp_vals_for_p = [np.zeros(1) for _ in range(len(calP))]

            p_idx = 0
            for p in calP.keys():
                count_p, calM_p = calP.get(p)
                l_p = len(p)
                L_mp_vals, D_mp_vals = compute_L_D_mp(dataset, count_p, calM_p, p, l_p, count_x, query)
                L_mp_vals_for_p[p_idx] = L_mp_vals
                D_mp_vals_for_p[p_idx] = D_mp_vals
                p_idx += 1

            D_vals_for_lambda, L_vals_for_lambda, Lambda = nearly_max_sum_min(L_mp_vals_for_p, D_mp_vals_for_p)

            D_vals = []

            for r in rate_vals:
                D_val = max_term(D_vals_for_lambda, L_vals_for_lambda, Lambda, r)
                D_vals.append(D_val)
                # print("R: ", r, ", D: ", D_val)
                # wandb.log({"R": r, "D": D_val})
            if condition == 3:
                output_file[data_path]["conditional"][query] = {
                    "rate": rate_vals,
                    "distortion": D_vals
                }
            print("{")
            print("\"query\": \"{}\", \"rate\": {}, \"distortion\": {}".format(query, rate_vals, D_vals))
            print("}")
        print("]")

        if condition == 3:
            outfile_path = f'output_RD/optimal_RD_{file_name}.json'
            pretty_json = json.dumps(output_file, indent=4)
             
            # Writing to sample.json
            with open(outfile_path, "w") as outfile:
                json.dump(output_file, outfile)

            print(data_path)
            print("done!!")




if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="Mistral-7B-Instruct-v0.2_optimal.jsonl", help="Dataset for computing RD curve")
    parser.add_argument("--condition", type=int, default = 0, help='(0) Query-agnostic; (1) Query-aware, average; (2) Query-aware, per query; (3) Do all and store results')
    parser.add_argument('--rate_vals', type=float, nargs='+', action='append', help='Rate values')
    parser.add_argument("--wandb_project", type=str, default="rd-lp-dual-eval", help="Name of wandb project")

    args = parser.parse_args()
    
    # # to set up weights and biases
    # exp_name = get_exp_name(args)
    # wandb.init(project=args.wandb_project, name=exp_name, config={
    #   "data_path": args.data_path,
    #   "rate_vals": args.rate_vals[0],
    #   "condition": args.condition
    #   })

    main(args.data_path, args.rate_vals[0], args.condition)

    # while True:       # wait to get stored files
    #     pass

    # wandb.finish()
