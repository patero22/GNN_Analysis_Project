from torch_geometric.datasets import Planetoid
import dgl
from format_manager import convert_pyg_format, convert_dgl_format
import torch


# Funkcja do tworzenia masek dla DGL
def create_masks(data, train_ratio=0.8, val_ratio=0.1):
    num_nodes = data.num_nodes()
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    perm = torch.randperm(num_nodes)
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size + val_size]] = True
    test_mask[perm[train_size + val_size:]] = True

    data.ndata['train_mask'] = train_mask
    data.ndata['val_mask'] = val_mask
    data.ndata['test_mask'] = test_mask


# Funkcja do ładowania danych w zależności od wybranej biblioteki i formatu
def load_data(library, matrix_format):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    if library == "PyG":
        # Konwertujemy format dla PyG
        data = convert_pyg_format(data, matrix_format)
        return data, dataset

    elif library == "DGL":
        # Konwertujemy format dla DGL
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        graph = convert_dgl_format(graph, matrix_format)

        # Tworzymy maski dla DGL
        create_masks(graph)

        return graph, dataset
