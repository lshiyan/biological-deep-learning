import json
import os

def generate_cnn_config_files(base_config, output_dir="Configs", num_layers=[3, 2, 1], whiten_values=[True, False], greedytrain_values=[True, False]):
    os.makedirs(output_dir, exist_ok=True)
    config_number = 0

    for layers in num_layers:
        for whiten in whiten_values:
            for greedytrain in greedytrain_values:

                config = json.loads(json.dumps(base_config))
                config['greedytrain'] = greedytrain
                
                # Update whiten
                for i in range(1, layers + 1):
                    conv_key = f"Conv{i}"
                    config['Convolutions'][conv_key]['whiten'] = whiten

                # Remove convolution layers beyond the specified number
                for i in range(layers + 1, 4):
                    config['Convolutions'].pop(f"Conv{i}", None)
                
                filename = f"config{config_number}.json"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w') as json_file:
                    json.dump(config, json_file, indent=4)
                
                print(f"Generated: {filename}")
                config_number += 1

# Base configuration template
base_config = {
    "Lambda" : 1, 
    "Lr" : 1e-4,
    "beta" : 1,
    "greedytrain" : True,
    "Convolutions" : {
        "Conv1":{
            "out_channel" : 32,
            "kernel" : 5,
            "stride" : 1,
            "padding" : 2,
            "paddingmode" : "reflect",
            "triangle" : True, 
            "whiten" : False, 
            "batchnorm" : True,
            "inhibition" : "REPU"
        }, 
        "Conv2" : {
            "out_channel" : 128,
            "kernel" : 3,
            "stride" : 1,
            "padding" : 1,
            "paddingmode" : "reflect",
            "triangle" : True,
            "whiten" : False,
            "batchnorm" : True,
            "inhibition" : "REPU"
        }, 
        "Conv3" : {
            "out_channel" : 512,
            "kernel" : 3,
            "stride" : 1,
            "padding" : 1,
            "paddingmode" : "reflect",
            "triangle" : True,
            "whiten" : False,
            "batchnorm" : True,
            "inhibition" : "REPU"
        }
    }, 
    "PoolingBlock" : {
        "Pooling" : True, 
        "Conv1" : {
            "Type" : "Max",
            "kernel" : 4,
            "stride" : 2, 
            "padding" : 1 
        }, 
        "Conv2" : {
            "Type" : "Max",
            "kernel" : 4,
            "stride" : 2, 
            "padding" : 1 
        },
        "Conv3" : {
            "Type" : "Avg",
            "kernel" : 2,
            "stride" : 2, 
            "padding" : 0 
        }
    }, 
    "Topdown" : False,
    "Rho" : 1e-3, 
    "Eta" : 0.1
}

generate_cnn_config_files(base_config)
