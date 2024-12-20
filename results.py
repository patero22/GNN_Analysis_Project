import csv
import time
import os
from data_loader import load_data
from train import train, evaluate_model
from models import GCN, GraphSAGE, GAT
import pandas as pd
#OGB_git


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
            writer.writerow(["Model", "Library", "Format", "Dataset", "E2E Time (s)", "Training Time (s)", "Memory Usage (MB)", "Validation Accuracy"])

        for model_name, model_class in models.items():
            for lib in libraries:
                for fmt in formats:
                    start_e2e_time = time.time()  # Rozpoczęcie pomiaru E2E

                    data, dataset_name, dataset = load_data(lib, fmt)
                    print(f"Running experiment for {model_name} with {lib} using {fmt} format on {dataset_name}.")

                    if lib == "PyG":
                        model = model_class(input_dim=data.num_node_features, hidden_dim=16, output_dim=dataset.num_classes)
                    else:
                        feat = data.ndata['feat']
                        label = data.ndata['label']

                        # Sprawdzenie, czy feat to dict
                        if isinstance(feat, dict):
                            feat = feat['_N']
                        if isinstance(label, dict):
                            label = label['_N']
                        model = model_class(input_dim=feat.shape[1], hidden_dim=16, output_dim=label.max().item() + 1)

                    train_time, mem_usage = train(model, data)  # Pomiar czasu trenowania i pamięci
                    accuracy = evaluate_model(model, data)  # Ewaluacja modelu
                    end_e2e_time = time.time()  # Koniec pomiaru E2E

                    e2e_time = end_e2e_time - start_e2e_time
                    print(f"E2E time: {e2e_time:.2f}s")

                    writer.writerow([model_name, lib, fmt, dataset_name, e2e_time, train_time, mem_usage, accuracy])
                    results.append([model_name, lib, fmt, dataset_name, e2e_time, train_time, mem_usage, accuracy])

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
