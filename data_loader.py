from torch_geometric.datasets import Planetoid
import dgl
from format_manager import convert_pyg_format, convert_dgl_format
from torch_geometric.transforms import ToSparseTensor

def load_data(library, matrix_format):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    if library == "PyG":
        if matrix_format == "CSR":
            # Konwersja do formatu CSR
            data = ToSparseTensor()(data)  # Zamienia edge_index na SparseTensor
        else:
            # Dla innych formatów
            data = convert_pyg_format(data, matrix_format)
        return data, dataset

    elif library == "DGL":
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        graph = convert_dgl_format(graph, matrix_format)
        return graph, dataset