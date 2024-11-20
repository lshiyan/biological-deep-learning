import csv
import statistics

def extract_accuracies_from_csv(file_path):
    accuracies_dict = {}

    with open(file_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        for row in csvreader:
            accuracy = float(row["test_accuracy"])
            hsize = row["hsize"]
            lamb = row["lambda"]
            w_lr = row["w_lr"]
            b_lr = row["b_lr"]
            l_lr = row["l_lr"]
            triangle = row["triangle"]
            white = row["white"]
            func = row["func"]
            w_norm = row["w_norm"]
            num_layers = row["num_layers"]
            hypers_key = (hsize, lamb, w_lr, b_lr, l_lr, triangle, white, func, w_norm, num_layers)
            
            if hypers_key not in accuracies_dict:
                accuracies_dict[hypers_key] = []
            accuracies_dict[hypers_key].append(accuracy)

    for hypers_key in accuracies_dict:
        accuracies = accuracies_dict[hypers_key]
        avg = statistics.mean(accuracies)
        var = statistics.variance(accuracies) if len(accuracies) > 1 else 0
        accuracies_dict[hypers_key] = [avg, var]
    
    return accuracies_dict


file_path = "MLP_hyper_search/multi_layer_results.csv"
accuracies_dict = extract_accuracies_from_csv(file_path)

sorted_accuracies = sorted(accuracies_dict.items(), key=lambda item: item[1][0], reverse=True)

for hyperparams, accuracies in sorted_accuracies:
    print(f"Hyperparameters: {hyperparams} -> Accuracies (Avg, Var): {accuracies}")
