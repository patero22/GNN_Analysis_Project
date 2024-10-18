import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        # Sprawdź, czy to PyG czy DGL
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            # PyTorch Geometric
            x, edge_index = data.x, data.edge_index
        else:
            # DGL - musimy skonwertować krawędzie
            x = data.ndata['feat']
            # DGL przechowuje krawędzie w inny sposób, użyjemy metody edges() do pozyskania edge_index
            edge_index = torch.stack(data.edges(), dim=0)  # Zwróć krawędzie w formacie [2, num_edges]

        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
