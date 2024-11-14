import json
import os

def generate_cnn_config_files(base_config, output_dir="ConfigsCNN", num_layers=[1], whiten_values=[True, False], triangle_values=[True],
    greedytrain_values=[True, False], inhibition_values=['Softmax'], pooling_values = ['PoolingStride1', 'NoPoolingStride2', 'NoPoolingStride1']
):
    
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
                            
                                ##### here add varying learning rate values and others as in mlp
                                config = json.loads(json.dumps(base_config))
                                config['greedytrain'] = greedytrain
                                config['nConvLayers'] = layers

                                if pool == 'PoolingStride1':
                                    config['PoolingBlock']['Pooling'] = True
                                else:
                                    config['PoolingBlock']['Pooling'] = False
                            
                                # Update whiten
                                for i in range(0, layers):
                                    conv_key = f"Conv{i+1}"
                                    config['Convolutions'][conv_key]['whiten'] = whiten
                                    config['Convolutions'][conv_key]['triangle'] = triangle
                                    config['Convolutions'][conv_key]['inhibition'] = inhibition

                                    if pool == 'NoPoolingStride2':
                                        config['Convolutions'][conv_key]['stride'] = 2
                                    else:
                                        config['Convolutions'][conv_key]['stride'] = 1
                                        

                                # Remove convolution layers beyond the specified number
                                for i in range(layers + 1, 5):
                                    config['Convolutions'].pop(f"Conv{i}", None)

                                # No need to remove extra pooling layers, just set the last one that will be used to avg pool
                                if pool == 'PoolingStride1':
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
    "Lr" : 0.01,
    "beta" : 1,
    "greedytrain" : True,
    "nConvLayers" : 1,

    "Convolutions" : {
        "stride" : 1,
        "padding" : 2,
        "paddingmode" : "reflect",
        "triangle" : False,
        "whiten" : False,
        "batchnorm" : True,
        "inhibition" : "REPU",

        "Conv1" : {
            "out_channel" : 32,
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
    }, 

    "PoolingBlock" : {
        "Pooling" : True, 
        "stride" : 1,

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
    }, 
    "GradientClassifier" : {
        
        
    },

    "Topdown" : False,
    "Rho" : 1e-3, 
    "Eta" : 0.1
}

generate_cnn_config_files(base_config)
