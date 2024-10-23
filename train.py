import torch.optim as optim
import torch.nn.functional as F
import time
import memory_profiler
import torch

import torch.optim as optim
import torch.nn.functional as F
import time
import memory_profiler
import torch


def create_masks_if_missing(data, train_size=0.6, val_size=0.2):
    num_nodes = data.num_nodes
    if not hasattr(data, 'train_mask') or not hasattr(data, 'val_mask') or not hasattr(data, 'test_mask'):
        # Tworzymy maski treningowe, walidacyjne i testowe
        perm = torch.randperm(num_nodes)
        train_end = int(train_size * num_nodes)
        val_end = int((train_size + val_size) * num_nodes)

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[perm[:train_end]] = True
        data.val_mask[perm[train_end:val_end]] = True
        data.test_mask[perm[val_end:]] = True


# def train(model, data):
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#     model.train()
#
#     start_time = time.time()
#     mem_usage_before = memory_profiler.memory_usage()[0]
#
#     for epoch in range(20):
#         optimizer.zero_grad()
#
#         out = model(data)
#
#         # PyG: używamy atrybutów specyficznych dla PyG
#         if hasattr(data, 'train_mask') and hasattr(data, 'y'):
#             # Sprawdzamy, czy maski i etykiety są dostępne
#             train_mask = data.train_mask
#             labels = data.y
#             loss = F.nll_loss(out[train_mask], labels[train_mask])
#         else:
#             # DGL: Używamy ndata dla DGL
#             train_mask = data.ndata['train_mask']
#             label = data.ndata['label']
#
#             # Sprawdzenie, czy są to słowniki (dla heterogenicznego grafu)
#             if isinstance(train_mask, dict):
#                 train_mask = train_mask['_N']  # Dostosowanie dla heterogenicznych grafów
#                 label = label['_N']
#
#             loss = F.nll_loss(out[train_mask], label[train_mask])
#
#         loss.backward()
#         optimizer.step()
#
#         if epoch % 25 == 0:
#             print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
#
#     end_time = time.time()
#     mem_usage_after = memory_profiler.memory_usage()[0]
#
#     train_time = end_time - start_time
#     mem_usage = mem_usage_after - mem_usage_before
#     print(f"Czas trenowania: {train_time:.2f} s, Zużycie pamięci: {mem_usage:.2f} MB")
#     return train_time, mem_usage
from torch_geometric.loader import DataLoader


def train(model, data, lib):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    # Jeśli dane mają więcej niż jeden graf, iterujemy po każdym grafie oddzielnie
    print(f"Liczba grafów w zbiorze danych: {len(data)}")

    start_time = time.time()
    mem_usage_before = memory_profiler.memory_usage()[0]

    criterion = torch.nn.BCEWithLogitsLoss()

    # Zakładam, że 'data' to lista grafów (np. dla PPI)
    for graph in data:
        print(f"Przetwarzanie grafu: {graph}")

        # Rozpakowujemy tuple, jeśli graph jest krotką (tuple)
        if isinstance(graph, tuple):
            x, edge_index = graph  # Rozpakowanie danych z tuple
        else:
            x = graph.x  # cechy węzłów
            edge_index = graph.edge_index  # macierz sąsiedztwa (krawędzie)

        # Upewniamy się, że są tensorami
        if isinstance(x, torch.Tensor) and isinstance(edge_index, torch.Tensor):
            out = model((x, edge_index))  # Przekazujemy dane do modelu
        else:
            raise ValueError("Wartość 'x' lub 'edge_index' nie jest tensorem!")

        # Używamy odpowiednich masek (train_mask) i etykiet (y)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask].float())
        loss.backward()
        optimizer.step()

    end_time = time.time()
    mem_usage_after = memory_profiler.memory_usage()[0]

    train_time = end_time - start_time
    mem_usage = mem_usage_after - mem_usage_before
    print(f"Czas trenowania: {train_time:.2f} s, Zużycie pamięci: {mem_usage:.2f} MB")

    return train_time, mem_usage


def evaluate_model(model, data, library):
    model.eval()  # Tryb ewaluacji
    with torch.no_grad():
        out = model(data)
        if library == "PyG":
            # PyG: out i data.y są wieloetykietowe dla PPI
            pred = (out > 0).float()  # Ponieważ BCEWithLogitsLoss używa wartości logitów, musimy przekonwertować wyniki na etykiety
            correct = (pred[data.test_mask] == data.y[data.test_mask].float()).sum().item()
            total = torch.numel(data.y[data.test_mask])
            accuracy = correct / total
        elif library == "DGL":
            # DGL: dostęp przez ndata
            test_mask = data.ndata['test_mask']
            label = data.ndata['label']

            if isinstance(test_mask, dict):
                test_mask = test_mask['_N']
                label = label['_N']

            pred = (out > 0).float()
            correct = (pred[test_mask] == label[test_mask].float()).sum().item()
            total = torch.numel(label[test_mask])
            accuracy = correct / total

    return accuracy




# def evaluate_model(model, data):
#     model.eval()  # Tryb ewaluacji
#     with torch.no_grad():
#         out = model(data)
#         if hasattr(data, 'test_mask'):  # PyG
#             pred = out.argmax(dim=1)
#             correct = pred[data.test_mask] == data.y[data.test_mask]
#             accuracy = int(correct.sum()) / int(data.test_mask.sum())
#         else:  # DGL
#             test_mask = data.ndata['test_mask']
#             label = data.ndata['label']
#
#             # Sprawdzenie, czy są to słowniki (np. dla heterogenicznego grafu)
#             if isinstance(test_mask, dict):
#                 test_mask = test_mask['_N']
#                 label = label['_N']
#
#             pred = out.argmax(dim=1)
#             correct = pred[test_mask] == label[test_mask]
#             accuracy = int(correct.sum()) / int(test_mask.sum())
#     return accuracy
