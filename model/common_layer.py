
from torch import nn
from omegaconf import DictConfig
import torch
import numpy
from gensim.models import KeyedVectors
from os.path import exists


class RNNLayer(torch.nn.Module):
    """

    """
    __negative_value = -numpy.inf

    def __init__(self, config: DictConfig, pad_idx: int):
        super(RNNLayer, self).__init__()
        self.__pad_idx = pad_idx
        self.__config = config
        self.__rnn = nn.LSTM(
            input_size=config.gnn.embed_size,           # 256
            hidden_size=config.gnn.rnn.hidden_size,     # 256
            num_layers=config.gnn.rnn.num_layers,
            bidirectional=config.gnn.rnn.use_bi,
            dropout=config.gnn.rnn.drop_out if config.gnn.rnn.num_layers > 1 else 0,
            batch_first=True)
        self.__dropout_rnn = nn.Dropout(config.gnn.rnn.drop_out)

    def forward(self, subtokens_embed: torch.Tensor, node_ids: torch.Tensor):
        """

        Args:
            subtokens_embed: [n nodes; max parts; embed dim]
            node_ids: [n nodes; max parts]

        Returns:

        """
        with torch.no_grad():
            is_contain_pad_id, first_pad_pos = torch.max(
                node_ids == self.__pad_idx, dim=1)
            first_pad_pos[~is_contain_pad_id] = node_ids.shape[
                1]  # if no pad token use len+1 position
            first_pad_pos[first_pad_pos == 0] = node_ids.shape[
                1]
            sorted_path_lengths, sort_indices = torch.sort(first_pad_pos,
                                                           descending=True)

            _, reverse_sort_indices = torch.sort(sort_indices)
            sorted_path_lengths = sorted_path_lengths.to(torch.device("cpu"))
        subtokens_embed = subtokens_embed[sort_indices]
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            subtokens_embed, sorted_path_lengths, batch_first=True)
        # [2; N; rnn hidden]
        _, (node_embedding, _) = self.__rnn(packed_embeddings)
        # [N; rnn hidden]
        node_embedding = node_embedding.sum(dim=0)

        # [n nodes; max parts; rnn hidden]
        node_embedding = self.__dropout_rnn(
            node_embedding)[reverse_sort_indices]

        return node_embedding


class STEncoder(torch.nn.Module):           # doc2vec 将代码的每一行生成对应的embedding
    """

    encoder for statement

    """

    def __init__(self, config: DictConfig, vocab,
                 vocabulary_size: int,
                 pad_idx: int):
        super(STEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__wd_embedding = nn.Embedding(vocabulary_size,             # 3124
                                           config.gnn.embed_size,           # 256
                                           padding_idx=pad_idx)
        # Additional embedding value for masked token
        torch.nn.init.xavier_uniform_(self.__wd_embedding.weight.data)
        if exists(config.gnn.w2v_path):                 #从预训练好的word2vec模型载入权重
            self.__add_w2v_weights(config.gnn.w2v_path, vocab)
        self.__rnn_attn = RNNLayer(config, pad_idx)     # 一层双向LSTM

    def __add_w2v_weights(self, w2v_path: str, vocab):      # vocab: Vocabulary
        """
        add pretrained word embedding to embedding layer

        Args:
            w2v_path: path to the word2vec model

        Returns:

        """
        model = KeyedVectors.load(w2v_path, mmap="r")
        w2v_weights = self.__wd_embedding.weight.data
        for wd in model.index_to_key:
            w2v_weights[vocab.convert_token_to_id(wd)] = torch.from_numpy(model[wd])        # model[wd] 对每个word生成 256维向量，
        self.__wd_embedding.weight.data.copy_(w2v_weights)

    def forward(self, seq: torch.Tensor):
        """

        Args:
            seq: [n nodes (seqs); max parts (seq len); embed dim]

        Returns:

        """
        # seq:[n nodes; max parts], wd_embedding: [n nodes; max parts: 256], node_embedding: [n nodes; 256]
        wd_embedding = self.__wd_embedding(seq)
        # [n nodes; rnn hidden]
        node_embedding = self.__rnn_attn(wd_embedding, seq)
        return node_embedding