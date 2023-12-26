import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from vocabulary import Vocabulary
from os.path import exists
import networkx as nx
from torch_geometric.data import Data

class CPGDataset(Dataset):
    def  __init__(self, CPG_paths_json: str, config: DictConfig, vocab: Vocabulary) -> None:
        """
        Args:
            CPG_root_path: json file of list of CPG paths
        return:
            返回
        """
        super().__init__()
        self.__config = config
        assert exists(CPG_paths_json), f"{CPG_paths_json} not exists!"
        with open(CPG_paths_json, "r") as f:
            # self.__CPG_paths_all_origin = list(json.load(f))
            # 这边由于是在虚拟机环境的地址，对于虚拟机，无法使用gpu加速，所以改变了路径
            self.__CPG_paths_all = [item.replace('/home/zhang/share/', 'C:/data/share/') for item in list(json.load(f))]
        self.__vocab = vocab
        self.__n_samples = len(self.__CPG_paths_all)

    def __len__(self) -> int:
        return self.__n_samples

    def init_graph(self, cpg_nx: nx.DiGraph, max_len=16):
        nodes, edges, tokens_list = [], [], []
        node_to_idx = {}
        node_type_idx = {}           # 对每个node type 下的 节点重新编号
        k_to_nodes = {}
        for idx, n in enumerate(cpg_nx):
            tokens = cpg_nx.nodes[n]['code_sym_token']
            cpg_node = cpg_nx.nodes[n]
            tokens_list.append(tokens)
            nodes.append(cpg_node)
            k_to_nodes[n] = cpg_node
            node_to_idx[n] = idx

        edge_dict = {}
        for edge in cpg_nx.edges:
            start_node_id, end_node_id = edge
            start_node_type = cpg_nx.nodes[start_node_id]["node_type"]
            end_node_type = cpg_nx.nodes[end_node_id]["node_type"]
            edge_type = tuple([start_node_type, cpg_nx.get_edge_data(start_node_id, end_node_id)["edge_type"], end_node_type])
            edge_dict.setdefault(edge_type, []).append(edge)

            # 反向边
            reverse_edge_type = tuple([end_node_type, "reverse_"+cpg_nx.get_edge_data(start_node_id, end_node_id)["edge_type"], start_node_type])
            reverse_edge = tuple([end_node_id, start_node_id])
            edge_dict.setdefault(reverse_edge_type, []).append(reverse_edge)

            # 自环
            edge_dict.setdefault(tuple([start_node_type, "self_loop", start_node_type]), []).append(tuple([start_node_id, start_node_id]))
            edge_dict.setdefault(tuple([end_node_type, "self_loop", end_node_type]), []).append(tuple([end_node_id, end_node_id]))

        for edge_type, edge_index in edge_dict.items():
            edge_dict[edge_type] = list(set(edge_index))        # 去重

        label = cpg_nx.graph["label"]  # 标签

        node_features = {}                  # {node_type: [num_nodes: node_feature]}
        for idx, n in enumerate(nodes):
            node_type = n["node_type"]
            node_token = tokens_list[idx]
            node_ids = np.full((max_len,), self.__vocab.get_pad_id(), dtype=np.long)
            ids = self.__vocab.convert_tokens_to_ids(node_token)
            less_len = min(max_len, len(ids))
            node_ids[:less_len] = ids[:less_len]          # 取前less_len
            node_type_idx.setdefault(node_type, []).append(idx)
            node_features.setdefault(node_type, []).append(node_ids)

            row_index = n["row_index"]

        node_to_idx_tmp = {}
        for node_type in node_type_idx.keys():
            idx_list = node_type_idx.get(node_type)
            for kk, idx in node_to_idx.items():
                if idx in idx_list:
                    node_to_idx_tmp[kk] = idx_list.index(idx)

        node_to_idx = node_to_idx_tmp

        edge_dict_tmp = {}
        for edge_type in edge_dict.keys():
            edge_index = torch.tensor(list(
                zip(*[[node_to_idx[e[0]],
                       node_to_idx[e[1]]] for e in edge_dict[edge_type]])),  # edge_index 重新编号后的idx
                dtype=torch.long)
            edge_dict_tmp[edge_type] = edge_index

        edge_dict = edge_dict_tmp

        for node_type in node_features.keys():
            node_features[node_type] = torch.tensor(node_features[node_type])

        return node_features, edge_dict, torch.tensor(label)

    def __getitem__(self, index):
        cpg_path = self.__CPG_paths_all[index]
        try:
            cpg = nx.read_gpickle(cpg_path)
            node_features, edge_dict, label = self.init_graph(cpg, max_len=16)
        except Exception:
            node_features, edge_dict, label = None, None, None
        return node_features, edge_dict, label

    def get_n_samples(self):
        return self.__n_samples



if __name__ == "__main__":
    vocab = Vocabulary.build_from_w2v("C:/data/share/CWE119/CPG/w2v.wv")  # word_embedding
    cpg_dataset = CPGDataset("C:/data/share/CWE119/CPG/val.json", "", vocab)
    cpg_dataset[0]