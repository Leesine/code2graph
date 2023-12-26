import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear
# from torch_geometric.transforms import GCNNorm
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
from model.common_layer import STEncoder


class Readout(nn.Module):
    """
    This module learns a single graph level representation for a molecule given GNN generated node embeddings
    """

    def __init__(self, attr_dim, embedding_dim, hidden_dim, output_dim, num_cats):
        super(Readout, self).__init__()
        self.attr_dim = attr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cats = num_cats

        self.layer1 = nn.Linear(attr_dim + embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Linear(output_dim, num_cats)
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.layer1.weight, gain=0.2)
        nn.init.xavier_normal_(self.layer2.weight, gain=0.2)
        nn.init.xavier_normal_(self.output.weight, gain=0.2)

    def forward(self, x_dict, node_embeddings_dict):

        for i, fea in enumerate(list(x_dict.values())):             # 拼接不同node_type 的 node_feature
            if i == 0:
                node_features = fea
            else:
                node_features = torch.cat((node_features, fea), dim=0)

        for i, fea in enumerate(list(node_embeddings_dict.values())):
            if i == 0:
                node_embeddings = fea
            else:
                node_embeddings = torch.cat((node_embeddings, fea), dim=0)

        combined_rep = torch.cat((node_features, node_embeddings), dim=1)        # Concat initial node attributed with embeddings from sage
        hidden_rep = self.act(self.layer1(combined_rep))
        graph_rep = self.act(self.layer2(hidden_rep))                       # Generate final graph level embedding

        logits = torch.mean(self.output(graph_rep), dim=0)                  # Generated logits for multilabel classification

        return logits


class HGT(torch.nn.Module):
    def __init__(self, config, vocab, vocabulary_size, pad_idx, hidden_channels, num_heads, num_layers, node_types, metadata):     # num_layers HGTConv的层数     # data 需要data.node_types 和  data.metadata()
        super().__init__()

        # self.gcn_norm = GCNNorm()

        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)    # doc2vec  得到初始化的node_embedding

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):

        for node_type, x in x_dict.items():         # doc2vec 生成初始的node embedding
            x_dict[node_type] = self.__st_embedding(x).relu_()

        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict                   # node embedding        x_dict[node_type] = (num_node, node_fea)


class HGTClassification(nn.Module):
    """
    Network that consolidates GCN + Readout into a single nn.Module
    """

    def __init__(self, config, vocab, vocabulary_size, pad_idx,in_channels, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata, num_categories):        # 此处channel 就相当于 dim
        super(HGTClassification, self).__init__()
        self.hgt = HGT(config, vocab, vocabulary_size, pad_idx, hidden_channels, num_heads=num_heads, num_layers=num_layers, node_types=node_types, metadata=metadata)
        self.readout = Readout(in_channels, hidden_channels, hidden_channels, out_channels, num_categories)

    def forward(self, x_dict, edge_index_dict):
        node_embeddings_dict = self.hgt(x_dict, edge_index_dict)
        logits = self.readout(x_dict, node_embeddings_dict)
        return logits