import csv
from time import time
import os
from data_loader import load_data
from train import train, evaluate_model
from models import GCN, GraphSAGE, GAT


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
        # Zapisz nagłówki tylko, jeśli plik nie istnieje
        if not file_exists:
            writer.writerow(["Model", "Library", "Format", "Dataset", "Training Time (s)", "Memory Usage (MB)", "Validation Accuracy"])

        for model_name, model_class in models.items():
            for lib in libraries:
                for fmt in formats:
                    # Ładujemy dane z odpowiednim zbiorem danych i formatem
                    data, dataset_name, dataset = load_data(lib, fmt)
                    print(f"Running experiment for {model_name} with {lib} using {fmt} format on {dataset_name}.")

                    if lib == "PyG":
                        model = model_class(input_dim=data.num_node_features, hidden_dim=16, output_dim=dataset.num_classes)
                    else:
                        # Sprawdzamy, czy cechy węzłów są w słowniku (dla heterogenicznych grafów)
                        if isinstance(data.ndata['feat'], dict):
                            # Pobieramy cechy dla konkretnego typu węzła (zakładamy, że używamy 'node_type' np. '_N')
                            feat = data.ndata['feat']['_N']
                            label = data.ndata['label']['_N']
                        else:
                            # Dla grafów jednorodnych
                            feat = data.ndata['feat']
                            label = data.ndata['label']

                        model = model_class(input_dim=feat.shape[1], hidden_dim=16, output_dim=label.max().item() + 1)

                    train_time, mem_usage = train(model, data, lib)
                    accuracy = evaluate_model(model, data, lib)  # Opcjonalna funkcja oceny modelu

                    # Zapisujemy wyniki z nazwą zbioru danych
                    writer.writerow([model_name, lib, fmt, dataset_name, train_time, mem_usage, accuracy])
                    results.append([model_name, lib, fmt, dataset_name, train_time, mem_usage, accuracy])

                    print(f"Finished {model_name} with {lib} using {fmt} format on {dataset_name}.")

    return results