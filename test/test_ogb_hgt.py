
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HGTConv, Linear

# 本文件仅用来测试HGT的运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ogb是一个学术的图网络
dataset = OGB_MAG(root='./data', preprocess='metapath2vec', transform=T.ToUndirected())
data = dataset[0]

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author'])


model = HGT(hidden_channels=64, out_channels=dataset.num_classes,
            num_heads=2, num_layers=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_loader = NeighborLoader(
    data,
    # Sample 15 neighbors for each node and each edge type for 2 iterations:
    num_neighbors=[15] * 2,
    # Use a batch size of 128 for sampling training nodes of type "paper":
    batch_size=128,
    input_nodes=('paper', data['paper'].train_mask),
)

def train():
    model.train()
    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = F.cross_entropy(out[:batch_size],
                               batch['paper'].y[:batch_size])
        print(loss.data)
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    print(total_loss / total_examples)
    return total_loss / total_examples

train()

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)