import torch_geometric.transforms as T
import dgl
from torch_sparse import SparseTensor
##OGB_git
# Funkcja obsługująca formaty macierzy dla PyTorch Geometric (PyG)
def convert_pyg_format(data, matrix_format):
    if matrix_format == "COO":
        # Dane już w formacie COO (edge_index)
        return data
    elif matrix_format == "CSR":
        # Konwersja do CSR (SparseTensor)
        data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=None).to('cpu')
        return data
    elif matrix_format == "CSC":
        # Konwersja do CSR, a potem transpozycja do CSC
        data.adj_t = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], value=None).to('cpu')
        data.adj_t = data.adj_t.t()  # Użycie t() zamiast transpose()
        return data
    else:
        raise ValueError("Nieznany format macierzy dla PyG: {}".format(matrix_format))

# Funkcja obsługująca formaty macierzy dla DGL
def convert_dgl_format(graph, matrix_format):
    if matrix_format == "COO":
        # DGL domyślnie obsługuje COO, nie trzeba konwertować
        return graph
    elif matrix_format == "CSR":
        # Konwersja do formatu CSR
        return dgl.to_block(dgl.to_simple(graph))  # Konwersja na CSR
    elif matrix_format == "CSC":
        # DGL nie wspiera natywnie formatu CSC, ale można próbować inne alternatywy
        graph = dgl.add_self_loop(graph)  # DGL nie wspiera natywnie CSC, dodajemy self-loop
        return graph
    else:
        raise ValueError("Nieznany format macierzy dla DGL: {}".format(matrix_format))
