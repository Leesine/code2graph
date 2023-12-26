import os
from argparse import ArgumentParser
from typing import cast
from omegaconf import OmegaConf, DictConfig

from vocabulary import Vocabulary
from train import train
import torch

def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="./config/config.yaml",
                            type=str)
    return arg_parser

def run(config_path: str):
    config = cast(DictConfig, OmegaConf.load(config_path))
    vocab = Vocabulary.build_from_w2v(config.gnn.w2v_path)      # word_embedding
    vocabulary_size = vocab.get_vocab_size()         # 3214
    pad_idx = vocab.get_pad_id()        # 0
    device = torch.device("cuda:" + str(config.cuda)) if config.cuda != -1 else torch.device("cpu")
    train(config, vocab, vocabulary_size, pad_idx, device)




if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    run(__args.config)