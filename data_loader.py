from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Planetoid
import torch
import dgl
from format_manager import convert_pyg_format, convert_dgl_format
from torch_geometric.transforms import ToSparseTensor
#OGB

def load_data(library, matrix_format, dataset_name="ogbn-arxiv"):
    if dataset_name.startswith("ogbn-"):  # Jeśli jest to zbiór danych z OGB
        dataset = NodePropPredDataset(name=dataset_name)
        split_idx = dataset.get_idx_split()  # Podziały na train/valid/test
        graph, labels = dataset[0]  # Pobieramy graf i etykiety

        if library == "PyG":
            import torch_geometric.data as pyg_data
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
            x = torch.tensor(graph["node_feat"], dtype=torch.float)
            y = torch.tensor(labels, dtype=torch.long)

            data = pyg_data.Data(x=x, edge_index=to_undirected(edge_index), y=y)
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[split_idx["train"]] = True
            data.val_mask[split_idx["valid"]] = True
            data.test_mask[split_idx["test"]] = True

            # Konwersja do wybranego formatu
            data = convert_pyg_format(data, matrix_format)
            return data, dataset_name, dataset

        elif library == "DGL":
            graph_dgl = dgl.graph((graph["edge_index"][0], graph["edge_index"][1]))
            graph_dgl.ndata['feat'] = torch.tensor(graph["node_feat"], dtype=torch.float)
            graph_dgl.ndata['label'] = torch.tensor(labels, dtype=torch.long)
            graph_dgl.ndata['train_mask'] = torch.zeros(graph_dgl.num_nodes(), dtype=torch.bool)
            graph_dgl.ndata['val_mask'] = torch.zeros(graph_dgl.num_nodes(), dtype=torch.bool)
            graph_dgl.ndata['test_mask'] = torch.zeros(graph_dgl.num_nodes(), dtype=torch.bool)
            graph_dgl.ndata['train_mask'][split_idx["train"]] = True
            graph_dgl.ndata['val_mask'][split_idx["valid"]] = True
            graph_dgl.ndata['test_mask'][split_idx["test"]] = True

            # Konwersja do wybranego formatu
            graph_dgl = convert_dgl_format(graph_dgl, matrix_format)
            return graph_dgl, dataset_name, dataset

    else:
        # Obsługa innych zbiorów danych
        return 1,2,3

# Funkcja do ładowania danych w zależności od wybranej biblioteki, formatu i zbioru danych
# def load_data(library, matrix_format):
#
#     dataset_name = "CiteSeer"  #np. "PubMed", "Cora", "CiteSeer"
#
#     if library == "PyG":
#         dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
#         data = dataset[0]
#
#         # Konwertujemy format dla PyG
#         data = convert_pyg_format(data, matrix_format)
#         return data, dataset_name, dataset
#
#     elif library == "DGL":
#         dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
#         data = dataset[0]
#
#         # Tworzenie grafu DGL
#         graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
#         graph.ndata['feat'] = data.x
#         graph.ndata['label'] = data.y
#
#         # Konwertujemy format dla DGL
#         graph = convert_dgl_format(graph, matrix_format)
#
#         # Ustawiamy maski (np. train_mask, val_mask, test_mask) dla DGL
#         create_dgl_masks(graph)
#
#         return graph, dataset_name, dataset
#
#
#
#
# def create_dgl_masks(graph):
#     # Sprawdzamy, czy graf jest heterogeniczny
#     if len(graph.ntypes) > 1:  # Mamy więcej niż jeden typ węzłów
#         for node_type in graph.ntypes:
#             num_nodes = graph.num_nodes(node_type)
#             train_mask = torch.zeros(num_nodes, dtype=torch.bool)
#             val_mask = torch.zeros(num_nodes, dtype=torch.bool)
#             test_mask = torch.zeros(num_nodes, dtype=torch.bool)
#
#             train_mask[:int(0.6 * num_nodes)] = True
#             val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
#             test_mask[int(0.8 * num_nodes):] = True
#
#             # Przypisujemy maski do konkretnego typu węzła
#             graph.nodes[node_type].data['train_mask'] = train_mask
#             graph.nodes[node_type].data['val_mask'] = val_mask
#             graph.nodes[node_type].data['test_mask'] = test_mask
#     else:  # Dla grafu jednorodnego
#         num_nodes = graph.num_nodes()
#         train_mask = torch.zeros(num_nodes, dtype=torch.bool)
#         val_mask = torch.zeros(num_nodes, dtype=torch.bool)
#         test_mask = torch.zeros(num_nodes, dtype=torch.bool)
#
#         train_mask[:int(0.6 * num_nodes)] = True
#         val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
#         test_mask[int(0.8 * num_nodes):] = True
#
#         graph.ndata['train_mask'] = train_mask
#         graph.ndata['val_mask'] = val_mask
#         graph.ndata['test_mask'] = test_mask