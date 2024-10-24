import csv
from time import time
import os
from data_loader import load_data
from train import train, evaluate_model
from models import GCN, GraphSAGE, GAT
import pandas as pd
#test


# Funkcja do zbierania wyników i zapisywania do CSV
def run_experiments():
    models = {
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT
    }
    formats = ["COO", "CSR", "CSC"]
    libraries = ["PyG", "DGL"]

    results = []

    file_exists = os.path.isfile('experiment_results.csv')

    with open('experiment_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Library", "Format", "Dataset", "Training Time (s)", "Memory Usage (MB)", "Validation Accuracy"])

        for model_name, model_class in models.items():
            for lib in libraries:
                for fmt in formats:
                    data, dataset_name, dataset = load_data(lib, fmt)
                    print(f"Running experiment for {model_name} with {lib} using {fmt} format on {dataset_name}.")

                    # # Obliczanie dodatkowych informacji o zbiorze
                    # num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.ndata['feat'].shape[0]
                    # num_edges = data.num_edges if hasattr(data, 'num_edges') else data.num_edges()  # DGL
                    # avg_degree = num_edges / num_nodes  # Średni stopień wierzchołka
                    # sparsity = (1 - (num_edges / (num_nodes * (num_nodes - 1)))) * 100  # Rzadkość grafu

                    # # Obliczanie rozmiaru w pamięci
                    # if lib == "PyG":
                    #     memory_size_gb = round(data.x.element_size() * data.x.nelement() / (1024 ** 3), 4)
                    # else:
                    #     memory_size_gb = round(
                    #         data.ndata['feat'].element_size() * data.ndata['feat'].nelement() / (1024 ** 3), 4
                    #     )
                    #
                    # save_dataset_info(dataset_name, num_nodes, num_edges, memory_size_gb, dataset.num_classes, avg_degree, sparsity)

                    if lib == "PyG":
                        model = model_class(input_dim=data.num_node_features, hidden_dim=16, output_dim=dataset.num_classes)
                    else:
                        feat = data.ndata['feat']
                        label = data.ndata['label']

                        # Sprawdzenie, czy 'feat' jest słownikiem (np. w grafie heterogenicznym)
                        if isinstance(feat, dict):
                            feat = feat['_N']  # Zakładamy, że '_N' to typ węzła w grafie heterogenicznym
                        if isinstance(label, dict):
                            label = label['_N']  # Zakładamy, że '_N' to typ węzła w grafie heterogenicznym
                        model = model_class(input_dim=feat.shape[1], hidden_dim=16, output_dim=label.max().item() + 1)

                    train_time, mem_usage = train(model, data)
                    accuracy = evaluate_model(model, data)

                    writer.writerow([model_name, lib, fmt, dataset_name, train_time, mem_usage, accuracy])
                    results.append([model_name, lib, fmt, dataset_name, train_time, mem_usage, accuracy])

                    print(f"Finished {model_name} with {lib} using {fmt} format on {dataset_name}.")

    return results



def save_dataset_info(dataset_name, num_nodes, num_edges, memory_size_gb, num_classes, avg_degree, sparsity):
    # Sprawdzenie, czy rekord już istnieje
    file_exists = os.path.isfile('dataset_info.csv')
    df = pd.read_csv('dataset_info.csv') if file_exists else pd.DataFrame()

    # Sprawdzenie, czy rekord już istnieje dla tego zbioru danych
    if not df.empty and dataset_name in df['Dataset'].values:
        print(f"Rekord dla zbioru {dataset_name} już istnieje. Pomijam zapis.")
        return

    # Zapisanie informacji o zbiorze danych
    with open('dataset_info.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Jeśli plik nie istnieje, zapisz nagłówki
        if not file_exists:
            writer.writerow(
                ['Dataset', 'Num Nodes', 'Num Edges', 'Memory Size (GB)', 'Num Classes', 'Avg Degree', 'Sparsity'])

        # Zapisz informacje
        writer.writerow([dataset_name, num_nodes, num_edges, memory_size_gb, num_classes, avg_degree, sparsity])
        print(f"Zapisano informacje o zbiorze {dataset_name}.")
