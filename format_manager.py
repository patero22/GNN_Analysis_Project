import torch_geometric.transforms as T
import dgl

# Funkcja obsługująca formaty macierzy dla PyTorch Geometric (PyG)
def convert_pyg_format(data, matrix_format):
    if matrix_format == "COO":
        # PyG domyślnie używa formatu COO, nie trzeba konwertować
        return data
    elif matrix_format == "CSR":
        # Konwersja do formatu CSR
        data = T.ToSparseTensor()(data)  # Konwersja na CSR
        return data
    elif matrix_format == "CSC":
        # Konwersja do CSR i transpozycja do formatu CSC
        data = T.ToSparseTensor()(data)
        data.adj_t = data.adj_t.transpose()  # CSR -> CSC przez transpozycję
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
