import os.path

import torch

from dataset.cpg_dataset import CPGDataset
from model.hgt import HGTClassification
from vocabulary import Vocabulary
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics


def get_optimizer(optimizer_name, model, lr=0.001):
    if optimizer_name == 'Adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)

    return optimizer


def get_model(config, vocab, vocabulary_size, pad_idx, model_name, in_channels, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata, num_categories):
    if model_name == "HGT":
        model = HGTClassification(config, vocab, vocabulary_size, pad_idx, in_channels, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata, num_categories)
    else:
        model = HGTClassification(config, vocab, vocabulary_size, pad_idx, in_channels, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata, num_categories)

    return model

def get_data_metadata(metadata_file_path):
    with open(metadata_file_path, "r") as f:
        node_types = f.readline()
        edge_types = f.readline()
    return eval(node_types), eval(edge_types)


def train(config, vocab, vocabulary_size, pad_idx, device):
    metadata_file_path = "metadata.txt"
    node_types, edge_types = get_data_metadata(metadata_file_path)
    metadata = (node_types, edge_types)
    model = get_model(config, vocab, vocabulary_size, pad_idx, "HGT", in_channels=64, hidden_channels=64, out_channels=64, num_heads=2, num_layers=2, node_types=node_types, metadata=metadata, num_categories=2)
    model = model.to(device) # gpu
    optimizer = get_optimizer(config.hyper_parameters.optimizer, model, lr=config.hyper_parameters.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-6)
    model.train()
    train_json_file = os.path.join(config.windows_data_folder, 'CWE119/CPG/train.json')

    vocab = Vocabulary.build_from_w2v(os.path.join(config.windows_data_folder, 'CWE119/CPG/w2v.wv'))
    train_dataset = CPGDataset(train_json_file, None, vocab)

    val_json_file = os.path.join(config.windows_data_folder, 'CWE119/CPG/test.json')
    val_dataset = CPGDataset(val_json_file, None, vocab)

    train_f = open("./train_hgt_info.log", "w")

    best_f_score = 0
    step = 0 # 总的迭代次数
    for epoch in range(config.hyper_parameters.n_epochs):
        print("epoch: {}".format(epoch), file=train_f)
        print("train: ", file=train_f)
        train_preds = []
        train_labels = []
        for i, (node_features, edge_dict, label) in enumerate(train_dataset):
            if node_features is None or edge_dict is None or label is None:
                continue
            optimizer.zero_grad()
            node_features = { key: node_features[key].to(device) for key in node_features.keys()}
            edge_dict = {key: edge_dict[key].to(device) for key in edge_dict.keys()}
            out = model(node_features, edge_dict)

            loss = criterion(out.view(-1, 2), label.view(-1).to(device))
            # loss = criterion(out, label.to(device))
            train_preds.append(out.argmax(0).item())
            train_labels.append(label.item())
            print("epoch: {}, iter: {}, train_loss: {}".format(epoch, i, loss.item()))
            print("epoch: {}, iter: {}, train_loss: {}".format(epoch, i, loss.item()), file=train_f)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                valid_preds = []
                valid_labels = []
                with torch.no_grad():
                    model.eval()
                    for i, (node_features, edge_dict, label) in enumerate(val_dataset):
                        if node_features is None or edge_dict is None or label is None:
                            continue
                        node_features = {key: node_features[key].to(device) for key in node_features.keys()}
                        edge_dict = {key: edge_dict[key].to(device) for key in edge_dict.keys()}
                        out = model(node_features, edge_dict)
                        loss = criterion(out.view(-1, 2), label.view(-1).to(device))
                        valid_preds.append(out.argmax(0).item())
                        valid_labels.append(label.item())
                        print("epoch: {}, iter: {}, valid loss: {}".format(epoch, i, loss.item()))
                        print("epoch: {}, iter: {}, valid loss: {}".format(epoch, i, loss.item()), file=train_f)
                        model.train()
                print('valid_preds', valid_preds)
                print('valid_labels', valid_labels)
                acc = metrics.accuracy_score(valid_labels, valid_preds)
                f1 = metrics.f1_score(valid_labels, valid_preds)
                precision = metrics.precision_score(valid_labels, valid_preds)
                recall = metrics.recall_score(valid_labels, valid_preds)
                print('acc:', acc)
                print('f1', f1)
                print('precision', precision)
                print('recall', recall)

        # train_precision, train_recall, train_f_score, _ = precision_recall_fscore_support(train_labels, train_preds, average="macro")
        # print("epoch: {}, train_precision: {}, train_recall: {}, train_f_score: {}".format(epoch, train_precision, train_recall, train_f_score))
        # print("epoch: {}, train_precision: {}, train_recall: {}, train_f_score: {}".format(epoch, train_precision, train_recall, train_f_score), file=train_f)
        #
        # print("valid: ", file=train_f)
        # valid_preds = []
        # valid_labels = []
        # with torch.no_grad():
        #     model.eval()
        #     for i, (node_features, edge_dict, label) in enumerate(val_dataset):
        #         if node_features is None or edge_dict is None or label is None:
        #             continue
        #         node_features = {key: node_features[key].to(device) for key in node_features.keys()}
        #         edge_dict = {key: edge_dict[key].to(device) for key in edge_dict.keys()}
        #         out = model(node_features, edge_dict)
        #         loss = criterion(out.view(-1, 2), label.view(-1).to(device))
        #         valid_preds.append(out.argmax(0).item())
        #         valid_labels.append(label.item())
        #         print("epoch: {}, iter: {}, valid loss: {}".format(epoch, i, loss.item()))
        #         print("epoch: {}, iter: {}, valid loss: {}".format(epoch, i, loss.item()), file=train_f)
        #     model.train()
        # valid_precision, valid_recall, valid_f_score, _ = precision_recall_fscore_support(valid_labels, valid_preds, average="macro")
        # print("epoch: {}, valid_precision: {}, valid_recall: {}, valid_f_score: {}".format(epoch, valid_precision, valid_precision, valid_f_score))
        # print("epoch: {}, valid_precision: {}, valid_recall: {}, valid_f_score: {}".format(epoch, valid_precision, valid_precision, valid_f_score), file=train_f)
        # print('valid_labels', valid_labels)
        # print('valid_preds', valid_preds)
        # if valid_f_score > best_f_score:
        #     best_f_score = valid_f_score
        #     print("save best f1 score {} model".format(best_f_score))
        #     print("save best f1 score {} model".format(best_f_score), file=train_f)
        #     torch.save(model.state_dict(), "best_f_score_{}.pth".format(best_f_score))

    torch.save(model.state_dict(), "last_epoch.pth")

    train_f.close()