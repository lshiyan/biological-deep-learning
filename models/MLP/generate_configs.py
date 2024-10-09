import json
import os
import itertools

def generate_mlp_config_files(hsizes, lambds, wlrs, blrs, llrs):
    allc = list(itertools.product(hsizes, lambds, wlrs, blrs, llrs))
    rank = 0
    for comb in allc:
        config = {
            "hsize" : comb[0],
            "lambd" : comb[1],
            "w_lr" : comb[2],
            "b_lr" : comb[3],
            "l_lr" : comb[4]
        }
        with open("ConfigsMLP/config" + str(rank) + ".json", "w") as jfile:
            json.dump(config, jfile, indent=4)