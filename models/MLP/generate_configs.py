import json
import os
import itertools

def generate_mlp_config_files(hsizes, lambds, wlrs, blrs, llrs, w_norm):
    allc = list(itertools.product(hsizes, lambds, wlrs, blrs, llrs, w_norm))
    rank = 0
    for comb in allc:
        config = {
            "hsize" : comb[0],
            "lambd" : comb[1],
            "w_lr" : comb[2],
            "b_lr" : comb[3],
            "l_lr" : comb[4], 
            "w_norm":comb[5]
        }
        with open("ConfigsMLP/config" + str(rank) + ".json", "w") as jfile:
            json.dump(config, jfile, indent=4)


def generate_mlp_config_files():
    p = [0.2, 0.5]
    num_test_for_each = 8
    for j in range(2):
        for i in range(0, num_test_for_each):
            config = {
            "hsize": 2048,
            "lambd": 125, 
            "w_lr": 0.1,
            "b_lr": 0.0033,
            "l_lr": 0.1,
            "w_norm": 0.01,
            "anti_hebb_factor": p[j]
            }
            with open("ConfigsMLP/config" + str(i + j*num_test_for_each) + ".json", "w") as jfile:
                json.dump(config, jfile, indent=4)


generate_mlp_config_files()