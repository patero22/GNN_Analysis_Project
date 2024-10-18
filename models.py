import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_sparse import SparseTensor


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        # Obsługa CSR (SparseTensor)
        if hasattr(data, 'adj_t') and isinstance(data.adj_t, SparseTensor):
            x = data.x
            edge_index = data.adj_t  # Bezpośrednie użycie SparseTensor (CSR)

            # Używamy SparseTensor w warstwach GCN
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)

        # Obsługa COO (standardowe edge_index)
        elif hasattr(data, 'x') and hasattr(data, 'edge_index'):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)

        # Obsługa DGL
        else:
            x = data.ndata['feat']
            edge_index = torch.stack(data.edges(), dim=0)
            x = F.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            x, edge_index = data.x, data.edge_index  # PyG
        else:
            x = data.ndata['feat']
            edge_index = torch.stack(data.edges(), dim=0)  # DGL

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, data):
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            x, edge_index = data.x, data.edge_index  # PyG
        else:
            x = data.ndata['feat']
            edge_index = torch.stack(data.edges(), dim=0)  # DGL

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
