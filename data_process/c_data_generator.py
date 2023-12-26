import os
import glob
import re
import networkx as nx
import xml.etree.ElementTree as ET
from os.path import join, exists
from argparse import ArgumentParser

import pydot
import tqdm
from omegaconf import OmegaConf, DictConfig
from typing import List, Set, Tuple, Dict, cast

'''
    由CPG.dot文件生成networkx图
'''

# removed_edge_type_list = ["SOURCE_FILE", "POST_DOMINATE"]
removed_edge_type_list = []

def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="../../config/config.yaml",
                            type=str)
    return arg_parser


def read_cpg_file(cpg_dir):
    nodes_dict = {}
    edges_dict = {}

    cpg_file_paths = glob.glob(os.path.join(cpg_dir, "*.dot"))

    for cpg_file_path in cpg_file_paths:
        graph = pydot.graph_from_dot_file(cpg_file_path)[0]
        graph_nodes = graph.get_nodes()
        graph_edges = graph.get_edges()

        method_name = os.path.basename(cpg_file_path).split(".")[0]

        nodes = []
        edges = []  # (start_node_id, end_node_id, edge_type)
        for graph_edge in graph_edges:
            start_node_id = int(graph_edge.obj_dict["points"][0])
            end_node_id = int(graph_edge.obj_dict["points"][1])
            edge_type = graph_edge.obj_dict['attributes']["label"].strip().split(" ")[0]        # 异构图
            # edge_type = "EDGE"            # 同构图
            if edge_type in removed_edge_type_list:
                continue
            edges.append([start_node_id, end_node_id, edge_type])

        for graph_node in graph_nodes:
            node_id = int(graph_node.obj_dict["name"])
            node_type = graph_node.obj_dict['attributes']["label"].strip()
            # node_type = "NODE"

            node_code = "<empty>"
            if "CODE" in graph_node.obj_dict['attributes'].keys():
                node_code = graph_node.obj_dict['attributes']["CODE"]
            elif "FULL_NAME" in graph_node.obj_dict['attributes'].keys():
                node_code = graph_node.obj_dict['attributes'].get("FULL_NAME")
            elif "NAME" in graph_node.obj_dict['attributes'].keys():
                node_code = graph_node.obj_dict['attributes'].get("NAME")

            if "LINE_NUMBER" in graph_node.obj_dict['attributes'].keys():
                line_number = int(graph_node.obj_dict['attributes'].get("LINE_NUMBER"))
            else:
                line_number = -1

            node_signature = node_code
            if "FULL_NAME" in graph_node.obj_dict['attributes'].keys():
                node_signature = graph_node.obj_dict['attributes'].get("FULL_NAME")
            elif "NAME" in graph_node.obj_dict['attributes'].keys():
                node_signature = graph_node.obj_dict['attributes'].get("NAME")

            nodes.append([node_id, node_type, node_code, node_signature, line_number])

        nodes_dict[method_name] = nodes
        edges_dict[method_name] = edges

    return nodes_dict, edges_dict


def build_CPG(cpg_dir: str, vul_lines: set, source_path) -> nx.DiGraph:
    """
    build program dependence graph from code

    Args:
        code_path (str): source cpg file dir
        vul_lines (str): vlu lines      vul_lines 错误代码行

    Returns: List(CPG)
    """

    nodes_dict, edges_dict = read_cpg_file(cpg_dir)
    CPGs = []
    for method_name in nodes_dict.keys():
        CPG = nx.DiGraph(name=method_name)  # 为每一个方法构图
        CPG.graph["file_path"] = source_path
        nodes = nodes_dict.get(method_name)
        edges = edges_dict.get(method_name)
        row_indexs = []
        for node in nodes:
            (node_id, node_type, node_var, node_signature, row_index) = node
            row_indexs.append(int(row_index))
            CPG.add_node(node_id, node_type=node_type, node_var=node_var, node_signature=node_signature, row_index=int(row_index))

        for edge in edges:
            (start_node_id, end_node_id, edge_type) = edge
            CPG.add_edge(start_node_id, end_node_id, edge_type=edge_type)

        # set(row_indexs) 函数代码少于5行，或 nodes 节点数少于10个， 应该被过滤了。
        # if len(set(row_indexs)) < 5 or len(nodes) < 10:
        #     continue

        if len(set(row_indexs).intersection(vul_lines)) != 0:           # vul_lines 错误代码行
            CPG.graph["label"] = 1                  # 存在错误代码
        else:
            CPG.graph["label"] = 0              # 不存在错误代码

        CPGs.append(CPG)

    return CPGs


def dump_CPG(CPGs, out_root_path, testcaseid):              # 保存CPG图到磁盘
    testcase_out_root_path = join(out_root_path, testcaseid)
    if not exists(testcase_out_root_path):
        os.makedirs(testcase_out_root_path)

    for CPG in CPGs:
        method_name = CPG.name
        out_path = join(testcase_out_root_path, method_name)
        print("save cpg to {}...".format(out_path))
        nx.write_gpickle(CPG, out_path)


def process_parallel(testcases: List, codeIDtoPath: Dict, cwe_root: str,
                     source_root_path: str,             # source_path 源代码 根
                     out_root_path: str):                   # out_root_path 保存 networkx 根
    for testcase in tqdm.tqdm(testcases):
        testcaseid = testcase.attrib["id"]
        if testcaseid in codeIDtoPath:
            file_map = codeIDtoPath[testcaseid]     # testcaseid: {testcaseid:{filePath:set(vul lines)}}   vul lines xml中mix和flaw的行号
            # file_map: {filePath:set(vul lines)}       # filePath 是 从source-code出发到源代码文件的路径
            for file_path in file_map:
                # print(file_path)
                vul_lines = file_map[file_path]         # 错误代码在源代码的行号 set(vul lines)
                # 解析的.dot文件路径
                cpg_dot_dir = join(cwe_root, file_path).replace("CWE119", "CWE119-output").split(".")[0] + "_cpg"
                if os.path.exists(cpg_dot_dir) is False:
                    continue
                source_path = join(source_root_path, file_path)
                CPGs = build_CPG(cpg_dot_dir, vul_lines, source_path)
                dump_CPG(CPGs, out_root_path, testcaseid)              # 保存CPG图到磁盘


def getCodeIDtoPathDict(testcases: List,
                        sourceDir: str) -> Dict[str, Dict[str, Set[int]]]:
    codeIDtoPath: Dict[str, Dict[str, Set[int]]] = {}
    for testcase in testcases:
        files = testcase.findall("file")  # 同一个.c文件夹下的testcases
        testcaseid = testcase.attrib["id"]  # 77704
        codeIDtoPath[testcaseid] = dict()

        for file in files:
            path = file.attrib["path"]  # testcase文件的路径
            flaws = file.findall("flaw")  # flaw
            mixeds = file.findall("mixed")  # <mixed line="47" name="CWE-127: Buffer Under-read" />
            fix = file.findall("fix")  # fix
            # print(mixeds)
            VulLine = set()
            if (flaws != [] or mixeds != [] or fix != []):
                # targetFilePath = path
                # flaws 和 mixeds是 错误（脆弱）代码
                if (flaws != []):
                    for flaw in flaws:
                        VulLine.add(int(flaw.attrib["line"]))
                if (mixeds != []):
                    for mixed in mixeds:
                        VulLine.add(int(mixed.attrib["line"]))

            codeIDtoPath[testcaseid][path] = VulLine

    return codeIDtoPath  # 返回{testcaseid:{filePath:set(vul lines)}}


def generate(config_path: str):
    config = cast(DictConfig, OmegaConf.load(config_path))
    root = config.data_folder           # root = "./data"
    cweid = config.dataset.name         # cweid = CWE119
    cwe_root = join(root, cweid)
    source_root_path = join(cwe_root, "source-code")
    out_root_path = join(cwe_root, "PDG")           # PDG/AST/CFG  networkx 保存目录
    xml_path = join(source_root_path, "manifest.xml")  # manifest.xml是testcase信息   label 信息

    tree = ET.ElementTree(file=xml_path)  # 解析xml文件
    testcases = tree.findall("testcase")
    codeIDtoPath = getCodeIDtoPathDict(testcases, source_root_path)         # 返回{testcaseid:{filePath:set(vul lines)}}

    if not exists(out_root_path):
        os.makedirs(out_root_path)

    process_parallel(testcases, codeIDtoPath, cwe_root, source_root_path, out_root_path)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    generate(__args.config)

