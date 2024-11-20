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
            "w_norm":comb[5],
            "num_layers": 1
        }
        with open("ConfigsMLP/config" + str(rank) + ".json", "w") as jfile:
            json.dump(config, jfile, indent=4)


def generate_mlp_config_files():

    num = [2,3,4,5]
    for j in range(len(num)):
        for i in range(0, 10):
            config = {
            "hsize": 2048,
            "lambd": 250, 
            "w_lr": 0.3,
            "b_lr": 0.0033,
            "l_lr": 0.1,
            "w_norm": 0.01,
            "num_layers": num[j]
            }
            with open("ConfigsMLP/config" + str(i + j*10) + ".json", "w") as jfile:
                json.dump(config, jfile, indent=4)


generate_mlp_config_files()
