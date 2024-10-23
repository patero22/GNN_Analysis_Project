from torch_geometric.datasets import Planetoid, Reddit
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
import torch
import dgl
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import PPI
from dgl.data import FlickrDataset
from format_manager import convert_pyg_format, convert_dgl_format
from torch_geometric.transforms import ToSparseTensor

# def load_data(library, matrix_format):
#     dataset_name = "ogbn-arxiv"  # Teraz używamy zbioru ogbn-arxiv
#
#     if dataset_name == "ogbn-arxiv":
#         if library == "PyG":
#             dataset = PygNodePropPredDataset(name=dataset_name)
#             data = dataset[0]
#
#             # Konwertujemy format dla PyG
#             data = convert_pyg_format(data, matrix_format)
#             return data, dataset_name, dataset
#
#         elif library == "DGL":
#             dataset = DglNodePropPredDataset(name=dataset_name)
#             graph, labels = dataset[0]
#
#             # Dodajemy cechy do grafu DGL
#             graph.ndata['feat'] = graph.ndata.pop('feat')
#             graph.ndata['label'] = labels.squeeze()  # OGB zwraca etykiety jako 2D, więc wyciskamy je
#
#             # Konwertujemy format dla DGL
#             graph = convert_dgl_format(graph, matrix_format)
#
#             # Tworzymy maski treningowe dla DGL
#             split_idx = dataset.get_idx_split()
#             graph.ndata['train_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
#             graph.ndata['train_mask'][split_idx['train']] = True
#             graph.ndata['val_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
#             graph.ndata['val_mask'][split_idx['valid']] = True
#             graph.ndata['test_mask'] = torch.zeros(graph.num_nodes(), dtype=torch.bool)
#             graph.ndata['test_mask'][split_idx['test']] = True
#
#             return graph, dataset_name, dataset
#
#     # Obsługa innych zbiorów danych (np. Cora, CiteSeer) pozostaje bez zmian
#     elif library == "PyG":
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

# Funkcja do ładowania danych w zależności od wybranej biblioteki, formatu i zbioru danych
# def load_data(library, matrix_format):
#     # Zmiana tutaj: ustawiamy zbiór danych na "CiteSeer" lub "PubMed"
#     #dataset_name = "CiteSeer"  # Możesz tu zmienić na np. "PubMed", "Cora", "CiteSeer"
#     dataset_name = "Reddit"
#
#     if library == "PyG":
#         #dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
#         dataset = Reddit(root=f'/tmp/{dataset_name}')
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

# Funkcja do ładowania danych w zależności od wybranej biblioteki, formatu i zbioru danych
# def load_data(library, matrix_format):
#     dataset_name = "ogbn-arxiv"  # Możesz tu zmienić na inne wspierane przez PyG i DGL zbiory
#
#     if dataset_name.startswith("ogbn"):  # Obsługa zbiorów OGB
#         if library == "PyG":
#             dataset = PygNodePropPredDataset(name=dataset_name)
#             data = dataset[0]
#
#             # Przekształcamy do odpowiedniego formatu
#             data = convert_pyg_format(data, matrix_format)
#             data.y = data.y.squeeze()  # OGB dane mają inną strukturę, trzeba je dostosować
#             return data, dataset_name, dataset
#
#         elif library == "DGL":
#             dataset = DglNodePropPredDataset(name=dataset_name)
#             graph, labels = dataset[0]
#
#             # Dostosowujemy do DGL
#             graph.ndata['feat'] = graph.ndata['feat']
#             graph.ndata['label'] = labels.squeeze()
#
#             # Konwertujemy format dla DGL
#             graph = convert_dgl_format(graph, matrix_format)
#             create_dgl_masks(graph, dataset)
#             return graph, dataset_name, dataset
#     else:
#         raise ValueError(f"Nieznany zbiór danych: {dataset_name}")

##
# def load_data(library, matrix_format):
#     dataset_name = "ogbn-products"  # Ustawiamy nazwę zbioru danych na ogbn-products
#
#     if library == "PyG":
#         dataset = PygNodePropPredDataset(name=dataset_name)
#         data = dataset[0]
#         data.y = data.y.squeeze()  # Dopasowanie do formatu
#
#         # Konwersja formatu
#         data = convert_pyg_format(data, matrix_format)
#         return data, dataset_name, dataset
#
#     elif library == "DGL":
#         dataset = DglNodePropPredDataset(name=dataset_name)
#         graph, labels = dataset[0]
#         graph.ndata['feat'] = graph.ndata['feat']
#         graph.ndata['label'] = labels.squeeze()
#
#         # Konwersja formatu
#         graph = convert_dgl_format(graph, matrix_format)
#         create_dgl_masks(graph)
#         return graph, dataset_name, dataset

# Funkcja do ładowania danych w zależności od wybranej biblioteki, formatu i zbioru danych
def load_data(library, matrix_format):
    dataset_name = "PPI"

    if library == "PyG":
        dataset = PPI(root=f'/tmp/{dataset_name}')
        data = dataset[0]

        # Konwertujemy format dla PyG
        data = convert_pyg_format(data, matrix_format)
        return data, dataset_name, dataset

    elif library == "DGL":
        dataset = PPI(root=f'/tmp/{dataset_name}')
        data = dataset[0]

        # Tworzenie grafu DGL
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y

        # Konwertujemy format dla DGL
        graph = convert_dgl_format(graph, matrix_format)

        # Ustawiamy maski (np. train_mask, val_mask, test_mask) dla DGL
        create_dgl_masks(graph)

        return graph, dataset_name, dataset


def create_dgl_masks(graph):
    # Sprawdzamy, czy graf jest heterogeniczny
    if len(graph.ntypes) > 1:  # Mamy więcej niż jeden typ węzłów
        for node_type in graph.ntypes:
            num_nodes = graph.num_nodes(node_type)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[:int(0.6 * num_nodes)] = True
            val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
            test_mask[int(0.8 * num_nodes):] = True

            # Przypisujemy maski do konkretnego typu węzła
            graph.nodes[node_type].data['train_mask'] = train_mask
            graph.nodes[node_type].data['val_mask'] = val_mask
            graph.nodes[node_type].data['test_mask'] = test_mask
    else:  # Dla grafu jednorodnego
        num_nodes = graph.num_nodes()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[:int(0.6 * num_nodes)] = True
        val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
        test_mask[int(0.8 * num_nodes):] = True

        graph.ndata['train_mask'] = train_mask
        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask