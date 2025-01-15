import pickle
from datetime import datetime
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torch
from dgl.data.utils import load_graphs

class Dataset:
    def __init__(self, name='yelp', device='cpu', prefix='./datasets/'):
        self.name = name
        self.adj: list[torch.Tensor] = []
        graph = load_graphs(prefix + name)[0][0]
        type_list = graph.etypes
        self.feature: torch.Tensor = graph.ndata['feature']
        self.label: torch.Tensor = graph.ndata['label']
        self.device = device
        for e_type in type_list:
            self.adj.append(graph.edges(etype=e_type))

    def process_data(self, training_ratio, device='cpu'):
        test_ratio = 1 - training_ratio
        # test_ratio = 0.90
        labels = self.label
        index = list(range(labels.shape[0]))
        test_size = int(test_ratio * len(index))
        train_idx, test_idx, y_train, y_test = train_test_split(index, labels, stratify=labels, test_size=test_size,
                                                                random_state=2, shuffle=True)
        nodes = np.arange(labels.shape[0])
        mask = np.zeros_like(nodes, dtype=bool)
        mask[train_idx] = True  # 1是训练集，0是测试集
        mask = torch.from_numpy(mask).to(device=self.device)
        return nodes, mask, labels, train_idx, test_idx, y_train, y_test, self.feature, self.adj


def save_results(config, results, file_name):
    file_dir = f'./results/{file_name}.txt'
    f = open(file_dir, 'a+')
    f.write(f"n_estimators: {config['n_estimators']}\tlearning_rate: {config['learning_rate']}\ttheta: {config['theta']}\t"
            f"max_depth: {config['max_depth']}\tAUROC: {results['AUROC']}\t"
            f"F1-Macro: {results['f1_macro']}\tgmean: {results['gmean']}\tTime:{results['Time']}\t\n")
    f.close()
    print(f'save to file name: {file_name}')


