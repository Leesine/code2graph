import glob
import os
import json
from typing import cast

import tqdm
from omegaconf import OmegaConf, DictConfig
import networkx as nx
from os.path import join
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from data_process.symbolizer import clean_gadget, tokenize_code_line
from os.path import exists

def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="../../config/config.yaml",
                            type=str)
    return arg_parser

def process_parallel(testcaseid: str, CPG_root_path: str, split_token: bool):
    """

    Args:
        testcaseid:
        queue:
        CPG_root_path:

    Returns:

    """
    testcase_root_path = join(CPG_root_path, testcaseid)
    cpg_ps = os.listdir(testcase_root_path)
    try:
        for cpg_p in cpg_ps:
            cpg_path = join(testcase_root_path, cpg_p)     # 所生成的CPG图的pkl文件
            cpg = nx.read_gpickle(cpg_path)         # 载入networkx
            for idx, n in enumerate(cpg):           # n: node_id
                if "code_sym_token" in cpg.nodes[n]:        # code token 已经symbol line   code token: {'code_sym_token': ['char', 'VAR1', '[', '100', ']', ';']}
                    return testcaseid
            file_path = cpg.graph["file_path"]      # 对应的源码文件路径
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:      # 读取源码
                file_contents = f.readlines()
            code_lines = list()
            for n in cpg.nodes:               # 只取在cpg中的源码对应行号
                if 'row_index' not in cpg.nodes[n].keys() or cpg.nodes[n]['row_index'] == -1 or cpg.nodes[n]['row_index'] > len(file_contents):
                    if 'node_var' in cpg.nodes[n].keys():
                        code_lines.append(cpg.nodes[n]['node_var'])
                    else:
                        code_lines.append("")
                    continue
                row_index = int(cpg.nodes[n]['row_index'])
                code_lines.append(file_contents[row_index - 1].strip())     # cpg中的行号从1开始，所以要n-1
            # code_lines = list(set(code_lines))
            sym_code_lines = clean_gadget(code_lines)       # 返回symbol后的字符串列表，每一个字符串对应源代码的一行， 变量归一化到VARi，函数名被归一化到FUNi,
            to_remove = list()
            for idx, n in enumerate(cpg):
                cpg.nodes[n]["code_sym_token"] = tokenize_code_line(sym_code_lines[idx], split_token)       # 将代码行的字符串拆分成一个个token
            #     if len(cpg.nodes[n]["code_sym_token"]) == 0:        # code_sym_token 为空的时候，则删去这个节点
            #         to_remove.append(n)
            # cpg.remove_nodes_from(to_remove)

            # 将归一化后的CPG替换原始的CPG
            nx.write_gpickle(cpg, cpg_path)
    except:
        print(cpg_path, n)
        exit()


def add_symlines(cweid: str, root: str, split_token: bool):
    """

    Args:
        cweid:
        root:

    Returns:

    """

    CPG_root_path = join(root, cweid, "CPG")
    testcaseids = [sub_dir for sub_dir in os.listdir(CPG_root_path) if os.path.isdir(os.path.join(CPG_root_path, sub_dir))]
    testcase_len = len(testcaseids)
    for testcaseid in tqdm.tqdm(testcaseids):
        process_parallel(testcaseid, CPG_root_path, split_token)



def split_list(cweid, root, out_root_path):
    """

    Args:
        cpg_paths:
        out_root_path:

    Returns:

    """
    CPG_root_path = join(root, cweid, "CPG")
    # testcaseids = os.listdir(CPG_root_path)
    files = glob.glob(os.path.join(CPG_root_path, "*/*"))

    X_train, X_test = train_test_split(files, test_size=0.1)
    # X_test, X_val = train_test_split(X_test, test_size=0.5)
    if not exists(f"{out_root_path}"):
        os.makedirs(f"{out_root_path}")
    with open(f"{out_root_path}/train.json", "w") as f:
        json.dump(X_train, f)
    with open(f"{out_root_path}/test.json", "w") as f:
        json.dump(X_test, f)
    # with open(f"{out_root_path}/val.json", "w") as f:
    #     json.dump(X_val, f)



if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config = cast(DictConfig, OmegaConf.load(__args.config))

    add_symlines(config.dataset.name, config.data_folder, config.split_token)       # 对源代码进行符号化（归一化），再写回networkx
    # 随机 9:1 分train, test ，并把各自的CPG file path 写入到对应的json文件中。
    split_list(config.dataset.name, config.data_folder, join(config.data_folder, config.dataset.name, "CPG"))