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
    p = [0.001, 0.0005]
    for j in range(len(p)):
        for i in range(0, 8):
            config = {
            "hsize": 2048,
            "lambd": 125, 
            "w_lr": 0.1,
            "b_lr": 0.0033,
            "l_lr": 0.1,
            "w_norm": 0.01
            }
            with open("ConfigsMLP/config" + str(i + j*8) + ".json", "w") as jfile:
                json.dump(config, jfile, indent=4)


generate_mlp_config_files()
