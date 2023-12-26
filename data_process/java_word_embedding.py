import os
from typing import cast

import tqdm
from omegaconf import OmegaConf, DictConfig
import json
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
from os import cpu_count
from multiprocessing import cpu_count

PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
BOS = "<BOS>"
EOS = "<EOS>"

SPECIAL_TOKENS = [PAD, UNK, MASK]


def process_parallel(path: str):             # 返回cpg的所有code_sym_token
    """

    Args:
        path:

    Returns:

    """
    cpg = nx.read_gpickle(path)
    tokens_list = list()
    for ln in cpg:
        code_tokens = cpg.nodes[ln]["code_sym_token"]

        if len(code_tokens) != 0:
            tokens_list.append(code_tokens)

    return tokens_list


def train_word_embedding(train_json_path: str):
    """
    train word embedding using word2vec
    """

    with open(train_json_path, "r") as f:
        paths = json.load(f)
    tokens_list = list()
    for path in tqdm.tqdm(paths):
        token_l = process_parallel(path)        # 载入CPG的code_sym_token
        tokens_list.extend(token_l)     # 得到train.json训练集里的所有code_sym_token

    print("training w2v...")
    num_workers = cpu_count()
    model = Word2Vec(sentences=tokens_list, min_count=3, size=256,
                     max_vocab_size=190000, workers=num_workers, sg=1)

    save_w2v_path = os.path.join(os.path.dirname(train_json_path), "w2v.wv")
    model.wv.save(save_w2v_path)


def load_wv(w2v_path: str):
    """

    Args:
        w2v_path:

    Returns:

    """

    model = KeyedVectors.load(w2v_path, mmap="r")

    print(model)

    return model


if __name__ == '__main__':
    w2v_path = ""
    load_wv(w2v_path)