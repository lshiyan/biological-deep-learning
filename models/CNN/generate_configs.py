import json
import os

def generate_cnn_config_files(base_config, output_dir="ConfigsCNN", num_layers=[1,2,3,4], whiten_values=[True], triangle_values=[True],
    greedytrain_values=[True], inhibition_values=['Softmax'], pooling_values = ['PoolingStride1']):
    
    # whiten = False for now
    # inhibition_values = REPU for now

    os.makedirs(output_dir, exist_ok=True)
    config_number = 0

    for greedytrain in greedytrain_values:
        for layers in num_layers:
            for whiten in whiten_values:
                for triangle in triangle_values:
                    for inhibition in inhibition_values:
                        for pool in pooling_values:
                            
                                config = json.loads(json.dumps(base_config))
                                config["Lambda"]= 125
                                config["classifierLr"]= 6e-4
                                config["w_norm"]= 0.0001
                                config["w_lr"]= 0.1
                                config["l_lr"]= 0.3
                                config["b_lr"]= 0.001
                                config['greedytrain'] = greedytrain
                                config['nConvLayers'] = layers

                                config['Convolutions']["GlobalParams"]['inhibition'] = inhibition
                                config['Convolutions']["GlobalParams"]['padding'] = 2
                                config['Convolutions']["GlobalParams"]['paddingmode'] = "reflect"
                                config['Convolutions']["GlobalParams"]['triangle'] = triangle
                                config['Convolutions']["GlobalParams"]['whiten'] = whiten
                                config['Convolutions']["GlobalParams"]['batchnorm'] = True

                                if pool == 'PoolingStride1':
                                    config['PoolingBlock']["GlobalParams"]['Pooling'] = True
                                else:
                                    config['PoolingBlock']["GlobalParams"]['Pooling'] = False

                                if pool == 'NoPoolingStride2':
                                    config['Convolutions']["GlobalParams"]['stride'] = 2
                                else:
                                    config['Convolutions']["GlobalParams"]['stride'] = 1

                                # Remove convolution layers beyond the specified number
                                for i in range(layers + 1, 5):
                                    config['Convolutions']['Layers'].pop(f"Conv{i}", None)

                                
                                if config['PoolingBlock']["GlobalParams"]['Pooling'] == True:
                                    config['PoolingBlock']['Layers'][f"Conv{layers}"]['Type'] = "Avg"
                                    for i in range(layers + 1, 5):
                                        config['PoolingBlock']['Layers'].pop(f"Conv{i}", None)
                                else:
                                    for i in range(0, 5):
                                        config['PoolingBlock']['Layers'].pop(f"Conv{i}", None)
                                    
                                
                                filename = f"config{config_number}.json"
                                filepath = os.path.join(output_dir, filename)
                                
                                with open(filepath, 'w') as json_file:
                                    json.dump(config, json_file, indent=4)
                                
                                print(f"Generated: {filename}")
                                config_number += 1

# Base configuration template
base_config = {

    "Lambda" : 125, 
    "classifierLr" : 0.01,
    "beta" : 1,
    "w_norm": 0.01,
    "w_lr": 0.01,
    "l_lr": 0.01,
    "b_lr": 0.01,
    "greedytrain" : True,
    "nConvLayers" : 1,

    "Convolutions" : {
        "GlobalParams": {
            "stride" : 1,
            "padding" : 2,
            "paddingmode" : "reflect",
            "triangle" : False,
            "whiten" : False,
            "batchnorm" : True,
            "inhibition" : "REPU"
        },
        "Layers": {
            "Conv1" : {
                "out_channel" : 512,
                "kernel" : 5
            }, 
            "Conv2" : {
                "out_channel" : 128,
                "kernel" : 3
            }, 
            "Conv3" : {
                "out_channel" : 512,
                "kernel" : 3
            },
            "Conv4" : {
                "out_channel" : 2048,
                "kernel" : 3
            }
        }
    }, 

    "PoolingBlock" : {
        "GlobalParams":{
            "Pooling" : True, 
            "stride" : 1
        },
        "Layers": {
            "Conv1" : {
                "Type" : "Max",
                "kernel" : 4,
                "padding" : 1 
            }, 
            "Conv2" : {
                "Type" : "Max",
                "kernel" : 4,
                "padding" : 1 
            },
            "Conv3" : {
                "Type" : "Max",
                "kernel" : 2,
                "padding" : 0 
            },
            "Conv4" : {
                "Type" : "Max",
                "kernel" : 2,
                "padding" : 0 
            }
        }
    }, 
    "Classifier" : {
        "type" : "Gradient",
        "out_channel" : 10
    },

    "Topdown" : False,
    "Rho" : 1e-3, 
    "Eta" : 0.1
}

generate_cnn_config_files(base_config)
