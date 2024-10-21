import json
import os

def generate_cnn_config_files(base_config, output_dir="ConfigsCNN", num_layers=[1, 2, 3, 4], whiten_values=[False], 
    greedytrain_values=[True, False], inhibition_values=['REPU'], pooling_values = [True, False, 'NoPoolingStride1']
):
    
    # whiten = False for now
    # inhibition_values = REPU for now

    os.makedirs(output_dir, exist_ok=True)
    config_number = 0

    for greedytrain in greedytrain_values:
        for layers in num_layers:
            for whiten in whiten_values:
                for inhibition in inhibition_values:
                    for pool in pooling_values: 

                        config = json.loads(json.dumps(base_config))
                        config['greedytrain'] = greedytrain

                        if pool == True:
                            config['PoolingBlock']['Pooling'] = True
                        else:
                            config['PoolingBlock']['Pooling'] = False
                    
                        # Update whiten
                        for i in range(1, layers + 1):
                            conv_key = f"Conv{i}"
                            config['Convolutions'][conv_key]['whiten'] = whiten
                            config['Convolutions'][conv_key]['inhibition'] = inhibition

                            if pool == True:
                                config['Convolutions'][conv_key]['stride'] = 1
                            elif pool == False:
                                config['Convolutions'][conv_key]['stride'] = 2
                            elif pool == 'NoPoolingStride1':
                                config['Convolutions'][conv_key]['stride'] = 1
                                

                        # Remove convolution layers beyond the specified number
                        for i in range(layers + 1, 5):
                            config['Convolutions'].pop(f"Conv{i}", None)

                        # No need to remove extra pooling layers, just set the last one that will be used to avg pool
                        if pool == True:
                            config['PoolingBlock'][f"Conv{layers}"]['Type'] = "Avg"

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
            "padding" : 2,
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
            "padding" : 2,
            "paddingmode" : "reflect",
            "triangle" : True,
            "whiten" : False,
            "batchnorm" : True,
            "inhibition" : "REPU"
        },
        "Conv4" : {
            "out_channel" : 2048,
            "kernel" : 3,
            "stride" : 1,
            "padding" : 2,
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
            "stride" : 1, 
            "padding" : 1 
        }, 
        "Conv2" : {
            "Type" : "Max",
            "kernel" : 4,
            "stride" : 1, 
            "padding" : 1 
        },
        "Conv3" : {
            "Type" : "Max",
            "kernel" : 2,
            "stride" : 1, 
            "padding" : 0 
        },
        "Conv4" : {
            "Type" : "Max",
            "kernel" : 2,
            "stride" : 1, 
            "padding" : 0 
        }
    }, 
    "Topdown" : False,
    "Rho" : 1e-3, 
    "Eta" : 0.1
}

generate_cnn_config_files(base_config)
