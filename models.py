import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_sparse import SparseTensor


# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(input_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, output_dim)
#
#     def forward(self, data):
#         # Sprawdzenie, czy dane są z PyG (PyTorch Geometric)
#         if hasattr(data, 'x') and (hasattr(data, 'edge_index') or hasattr(data, 'adj_t')):
#             x = data.x
#
#             # Sprawdzenie, czy dane są w formacie CSR (SparseTensor)
#             if hasattr(data, 'adj_t') and isinstance(data.adj_t, SparseTensor):
#                 edge_index = data.adj_t
#             else:
#                 edge_index = data.edge_index
#
#             x = F.relu(self.conv1(x, edge_index))
#             x = self.conv2(x, edge_index)
#
#         # Sprawdzenie, czy dane są z DGL
#         elif hasattr(data, 'ndata'):
#             if isinstance(data.ndata['feat'], dict):
#                 x = data.ndata['feat']['_N']  # Pobieramy tensor dla 'feat'
#             else:
#                 x = data.ndata['feat']
#
#             edge_index = torch.stack(data.edges(), dim=0)
#             x = F.relu(self.conv1(x, edge_index))
#             x = self.conv2(x, edge_index)
#
#             # Sprawdzenie, czy dane są w formie batcha (PPI w PyG)
#         elif isinstance(data, tuple) and len(data) == 2:
#             x, edge_index = data  # Dostosowanie dla batcha w formie tuple
#
#             x = F.relu(self.conv1(x, edge_index))
#             x = self.conv2(x, edge_index)
#
#         return F.log_softmax(x, dim=1)
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        if isinstance(data, tuple):
            x, edge_index = data  # Zakładamy, że tuple zawiera `x` i `edge_index`
        else:
            # Sprawdzenie czy dane zawierają właściwe klucze
            if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                x = data.x
                edge_index = data.edge_index
            else:
                raise ValueError("Dane wejściowe nie zawierają odpowiednich kluczy ('x', 'edge_index').")

        # Sprawdzamy, czy `x` i `edge_index` są tensorami, a nie stringami
        if isinstance(x, torch.Tensor) and isinstance(edge_index, torch.Tensor):
            print(f"Przetwarzanie batcha: x = {x.shape}, edge_index = {edge_index.shape}")
        else:
            raise ValueError("Wartość 'x' lub 'edge_index' jest ciągiem znaków, a nie tensorem!")

        # Przetwarzamy dane przez GCN
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)



class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):  # PyG
            x, edge_index = data.x, data.edge_index
        else:  # DGL
            if isinstance(data.ndata['feat'], dict):
                # Heterogeniczny graf: Pobieramy dane dla typu węzła '_N'
                x = data.ndata['feat']['_N']
                edge_index = torch.stack(data.edges(), dim=0)
            else:
                # Graf jednorodny
                x = data.ndata['feat']
                edge_index = torch.stack(data.edges(), dim=0)

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, data):
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):  # PyG
            x, edge_index = data.x, data.edge_index
        else:  # DGL
            if isinstance(data.ndata['feat'], dict):
                x = data.ndata['feat']['_N']
            else:
                x = data.ndata['feat']

            edge_index = torch.stack(data.edges(), dim=0)

        # Dla PyG, nadal używamy edge_index z PyG
        x = F.relu(self.conv1(x, edge_index))  # Pierwsza warstwa GATConv
        x = self.conv2(x, edge_index)          # Druga warstwa GATConv

        return F.log_softmax(x, dim=1)

