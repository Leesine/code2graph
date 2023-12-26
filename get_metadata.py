import os
import glob
import re

import networkx as nx
import tqdm

def get_all_file_paths(cpg_file_root):
    cpg_dot_file_paths = []
    all_files = os.walk(cpg_file_root)
    for root, dirs, files in all_files:
        for file in files:
            if file.endswith(".dot"):
                cpg_dot_file_path = os.path.join(root, file)
                cpg_dot_file_paths.append(os.path.abspath(cpg_dot_file_path))

    return cpg_dot_file_paths


def get_all_node_edge_type(cpg_networkx_root):
    cpg_networkx_file_paths = glob.glob(os.path.join(cpg_networkx_root, "*/*"))

    node_types = set()
    edge_types = set()

    for cpg_networkx_file_path in tqdm.tqdm(cpg_networkx_file_paths):
        print(cpg_networkx_file_path)
        try:
            cpg = nx.read_gpickle(cpg_networkx_file_path)
        except EOFError:
            print('当前pickle文件为空.')
            continue
        try:
            for node_index in cpg.nodes:
                node_types.add(cpg.nodes[node_index]["node_type"])
        except:
            print(cpg_networkx_file_path, cpg.graph["file_path"], node_index, cpg.nodes[node_index])
            exit()

        for edge_index in cpg.edges:
            edge_type = cpg.edges[edge_index]["edge_type"]
            start_node_type = cpg.nodes[edge_index[0]]["node_type"]
            end_node_type = cpg.nodes[edge_index[1]]["node_type"]

            edge_types.add((start_node_type, edge_type, end_node_type))

            # 反向边
            edge_types.add((end_node_type, "reverse_" + edge_type, start_node_type))
            # self loop
            edge_types.add((start_node_type, "self_loop", start_node_type))
            edge_types.add((end_node_type, "self_loop", end_node_type))

    with open("metadata.txt", "w") as f:
        print(list(node_types), file=f)
        print(list(edge_types), file=f)



if __name__ == "__main__":

    cpg_networkx_root = "/home/zhang/share/CWE119/CPG"
    get_all_node_edge_type(cpg_networkx_root)