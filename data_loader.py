from torch_geometric.datasets import Planetoid
import torch
import dgl
from format_manager import convert_pyg_format, convert_dgl_format
from torch_geometric.transforms import ToSparseTensor


def load_data(library, matrix_format):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    if library == "PyG":
        # Konwersja dla PyG
        data = convert_pyg_format(data, matrix_format)
        return data, dataset

    elif library == "DGL":
        # Tworzenie grafu DGL
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y

        # Tworzenie masek trenowania w DGL
        create_dgl_masks(graph)

        return graph, dataset

# Funkcja do tworzenia masek dla DGL
def create_dgl_masks(graph, train_ratio=0.8, val_ratio=0.1):
    num_nodes = graph.num_nodes()
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    indices = torch.randperm(num_nodes)
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask