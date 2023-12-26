from argparse import ArgumentParser
from typing import cast
from omegaconf import OmegaConf, DictConfig
import json
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
from os import cpu_count
from multiprocessing import cpu_count

PAD = '<PAD>'
UNK = '<UNK>'
MASK = '<MASK>'
BOS = '<BOS>'
EOS = '<EOS>'

SPECIAL_TOKENS = [PAD, UNK, MASK]

def process_parallel(path: str, split_token: bool): # 返回cpg的所有code_sym_token
    """

    :param path:
    :param split_token:
    :return:
    """
    cpg = nx.read_gpickle(path)
    tokens_list = list()
    for ln in cpg:
        code_tokens = cpg.nodes[ln]['code_sym_token']

        if len(code_tokens) != 0:
            tokens_list.append(code_tokens)

    return tokens_list

def train_word_embedding(config_path: str):
    """
    train word embedding using word2vec
    :param config_path:
    :return:
    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    cweid = config.dataset.name
    root = config.data_folder
    train_json = f"{root}/{cweid}/CPG/train.json"
    with open(train_json, "r") as f:
        paths = json.load(f)
    tokens_list = list()
    for path in paths:
        token_l = process_parallel(path, split_token=config.split_token)
        tokens_list.extend(token_l)

    print("training w2v...")
    num_workers = cpu_count() if config.num_workers == -1 else config.num_workers
    model = Word2Vec(sentences=tokens_list, min_count=3, vector_size=config.gnn.embed_size,
                     max_vocab_size=config.dataset.token.vocabulary_size, workers=num_workers, sg=1)
    model.wv.save(f"{root}/{cweid}/CPG/w2v.wv")

def load_wv(config_path: str):
    """

    :param config_path:
    :return:
    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    cweid = config.dataset.name

    model = KeyedVectors.load(f"{config.data_folder}/{cweid}/CPG/w2v.wv", mmap="r")
    print(model)
    return model

if __name__ == '__main__':
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c",
                              "--config",
                              help="Path to YAML configuration file",
                              default="../config/config.yaml",
                              type=str)
    __args = __arg_parser.parse_args()
    train_word_embedding(__args.config)
    load_wv(__args.config)