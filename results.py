import csv
from time import time
from data_loader import load_data
from train import train
from models import GCN, GraphSAGE, GAT

# Definicja eksperymentów
def run_experiments():
    models = {
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT
    }
    formats = ["COO", "CSR", "CSC"]
    libraries = ["PyG", "DGL"]

    # Otwieramy plik CSV do zapisu wyników
    with open('experiment_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Library", "Format", "Training Time (s)", "Memory Usage (MB)"])

        for model_name, model_class in models.items():
            for lib in libraries:
                for fmt in formats:
                    print(f"Running experiment for {model_name} with {lib} using {fmt} format.")
                    data, dataset = load_data(lib, fmt)

                    if lib == "PyG":
                        model = model_class(input_dim=data.num_node_features, hidden_dim=16, output_dim=dataset.num_classes)
                    else:
                        model = model_class(input_dim=data.ndata['feat'].shape[1], hidden_dim=16, output_dim=data.ndata['label'].max().item() + 1)

                    # Uruchamiamy trenowanie i zapisujemy wyniki
                    train_time, mem_usage = train(model, data)
                    writer.writerow([model_name, lib, fmt, train_time, mem_usage])
                    print(f"Finished {model_name} with {lib} using {fmt} format.")
