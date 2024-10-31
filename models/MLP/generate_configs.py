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
    for i in range(0, 20):
        config = {
        "hsize": 2048,
        "lambd": 243,
        "w_lr": 0.3,
        "b_lr": 0.01,
        "l_lr": 0.01,
        "w_norm": 0.01
        }
        with open("ConfigsMLP/config" + str(i) + ".json", "w") as jfile:
            json.dump(config, jfile, indent=4)


generate_mlp_config_files()
