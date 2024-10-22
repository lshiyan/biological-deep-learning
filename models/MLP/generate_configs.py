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


def generate_mlp_config_files():
    l_lr = [0, 0.1, 0.001, 0.033, 0.33]
    for j in range(len(l_lr)):
        for i in range(0, 10):
            config = {
                "hsize": 256,
                "lambd": 5,
                "w_lr": 0.033,
                "b_lr": 0.1,
                "l_lr": 0.01
            }
            with open("ConfigsMLP/config" + str(i+j*10) + ".json", "w") as jfile:
                json.dump(config, jfile, indent=4)


generate_mlp_config_files()
