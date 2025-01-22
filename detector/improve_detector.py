import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
from imblearn.metrics import geometric_mean_score
import xgboost as xgb
from model.model import SpectralFeature


class GraphDetector:
    def __init__(self, graph, nodes, labels, mask, feature, train_config: dict):
        device = train_config["device"]
        n_estimators = 100 if train_config['n_estimators'] is None else train_config['n_estimators']
        learning_rate = 0.2 if train_config['learning_rate'] is None else train_config['learning_rate']
        max_depth = 20 if train_config['max_depth'] is None else train_config['max_depth']
        select_anomaly = ((labels == 1) & (mask == 1)).to(torch.bool).to(device)
        select_normal = ((labels == 0) & (mask == 1)).to(torch.bool).to(device)
        # gnn = SpectralFeature(graph, feature.shape[0], nodes[select_anomaly], nodes[select_normal],
        #                       theta=train_config['theta'])
        gnn = SpectralFeature(graph, feature.shape[0], nodes[select_anomaly], nodes[select_normal],
                              theta=train_config['theta'], device=device)
        if device == "cpu":
            self.model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           eval_metric=roc_auc_score, max_depth=max_depth, tree_method="hist",device="cpu")
        else:
            self.model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                           eval_metric=roc_auc_score, max_depth=max_depth, tree_method="hist",
                                           predictor='gpu_predictor', device="cuda")

        feature = feature.float()
        fea_list = gnn.embedding(feature)
        # fea_list = gnn.embedding(feature)
        self.feature = torch.cat(fea_list, -1)
        # self.feature = torch.mean(torch.stack(fea_list), dim=0)

    def train(self, train_idx, train_y, test_idx):
        self.model.fit(self.feature[train_idx], train_y)
        pred_y = self.model.predict_proba(self.feature[test_idx])
        pred_y = pred_y[:, 1]
        return pred_y

    def train_val(self, train_idx, train_y, val_idx, val_y, test_idx):
        val_x = self.feature[val_idx].detach().numpy()
        self.model.fit(self.feature[train_idx].detach().numpy(), train_y, eval_set=[(val_x, val_y)])
        pred_y = self.model.predict_proba(self.feature[test_idx].detach().numpy())
        pred_y = pred_y[:, 1]
        return pred_y

    def save_model(self, file_name):
        self.model.save_model(f'../save_model/{file_name}.json')


def eval(labels, probs: torch.Tensor):
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        AUROC = roc_auc_score(labels, probs)
        AUPRC = average_precision_score(labels, probs)
        labels = np.array(labels)
        k = int(labels.sum())
    RecK = sum(labels[probs.argsort()[-k:]]) / sum(labels)
    pred = probs.copy()
    pred[probs >= 0.5] = 1
    pred[probs < 0.5] = 0
    f1_micro = f1_score(labels, pred, average='micro')
    f1_macro = f1_score(labels, pred, average='macro')
    # label_pred = np.where(probs >= 0.5, 1, 0)
    # label_true = np.sum(label_pred == labels)
    recall = recall_score(labels, pred)
    g_mean = geometric_mean_score(labels, pred)
    return AUROC, AUPRC, RecK, f1_micro, f1_macro, recall, g_mean
