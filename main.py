from data_loader import load_data
from train import train
from models import GCN
from results import run_experiments
from plot_generator import generate_plots

# library = "PyG"  # Możliwe wartości: "PyG", "DGL"
# matrix_format = "COO"  # Możliwe wartości: "COO", "CSR", "CSC"
# data, dataset = load_data(library, matrix_format)

# if library == "PyG":
#     model = GCN(input_dim=data.num_node_features, hidden_dim=16, output_dim=dataset.num_classes)
# else:
#     model = GCN(input_dim=data.ndata['feat'].shape[1], hidden_dim=16, output_dim=data.ndata['label'].max().item() + 1)

#result = run_experiments()

#train_time, mem_usage = train(model, data)
#print(f"Czas trenowania: {train_time:.2f} s, Zużycie pamięci: {mem_usage:.2f} MB")

#generate_plots(result)

# Uruchamiamy eksperymenty
results = run_experiments()

# Możesz dodać generowanie wykresów po zakończeniu eksperymentów
#generate_plots(results)